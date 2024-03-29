"""Main training loop"""
import argparse
import ctypes
import multiprocessing as mp
import pickle, json
from datetime import datetime
from os import environ
# Only works with POSIX-style paths
#environ["PRISMATA_INIT_AI_JSON_PATH"] = f"{'/'.join(__file__.split('/')[:-1])}/AI_config.txt"
import prismataengine

import numpy as np
import torch

from model import MLPBase, D2RLNet, CategoricalMasked
from train import train_step
from worker import GamePlayer
from utils import gae, RunningMeanStd, apply_normalizer
from tracker import WandBTracker, ConsoleTracker
start_time=datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--env_name', default="PrismataEnv-v0")
parser.add_argument('--model', default="mlp")
parser.add_argument('--gamma', default=.99, type=float)
parser.add_argument('--lam', default=.95, type=float)
parser.add_argument('--epsilon', default=.1, type=float)
parser.add_argument('--value_loss_coef', default=.5, type=float)
parser.add_argument('--entropy_coef', default=.01, type=float)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--num_iterations', default=10**10, type=int)
parser.add_argument('--num_steps', default=2048, type=int)
parser.add_argument('--ppo_epochs', default=4, type=int)
parser.add_argument('--num_batches', default=4, type=int)
parser.add_argument('--lr', default=2.5e-5, type=float)
parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--clip_rewards', default=False)
parser.add_argument('--logger', default="console")
parser.add_argument('--policy', default="RandomAI")
parser.add_argument('--cards', default="4")
parser.add_argument('--model_dir', default=".")
parser.add_argument('--load_model_dir', default=None)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--max_grad_norm', default=.5, type=float)
parser.add_argument('--model_save_interval', default=50, type=int)
#parser.add_argument('--memory_profiler', default=False, type=bool)
parser.add_argument('--player', default='p1')
parser.add_argument('--nn_opponent', default=None)
parser.add_argument('--one_hot', default=False)
args = parser.parse_args()
if args.nn_opponent:
    with open(args.nn_opponent) as f:
        args.nn_opponent = json.load(f)
assert(args.num_workers >= args.num_batches)
args.batch_size = int(args.num_workers / args.num_batches)

if args.one_hot=='True':
    args.one_hot=True
else:
    args.one_hot=False

if args.cards=="4":
    args.num_actions=14
    if args.one_hot:
        args.num_obs=670
    else:
        args.num_obs = 30
elif args.cards=="11":
    args.num_actions=32
    if args.one_hot:
        args.num_obs=1242
    else:
        args.num_obs = 82
else:
    raise ValueError('Cards Not Accepted')


if args.player not in ['p1','p2']:
    raise ValueError('Player mode not recognized')


runid=f"trainedpolicy{args.policy}_net{args.model}_cards{args.cards}_player{args.player}_onehot{args.one_hot}_time{start_time}"

device = torch.device(args.device)

# Define common shapes for convenience
scalar_shape = (args.num_workers, args.num_steps)
batch_obs_shape = (args.num_workers, args.num_steps, args.num_obs)
batch_legals_shape = (args.num_workers, args.num_steps, args.num_actions)


# Make a shared array to get observations / legals from each process
# and wrap it with Numpy
shared_obs_c = mp.Array(ctypes.c_float, int(np.prod(batch_obs_shape)))
shared_obs = np.frombuffer(shared_obs_c.get_obj(), dtype=np.float32)
shared_obs = np.reshape(shared_obs, batch_obs_shape)

shared_legals_c = mp.Array(ctypes.c_float, int(np.prod(batch_legals_shape)))
shared_legals = np.frombuffer(shared_legals_c.get_obj(), dtype=np.float32)
shared_legals = np.reshape(shared_legals, batch_legals_shape)

# Make arrays to store all other rollout info
rewards = np.zeros(scalar_shape, dtype=np.float32)
discounted_rewards = np.zeros(scalar_shape, dtype=np.float32)
episode_ends = np.zeros(scalar_shape, dtype=np.float32)
values = np.zeros(scalar_shape, dtype=np.float32)
policy_probs = np.zeros(scalar_shape, dtype=np.float32)
actions = np.zeros(scalar_shape, dtype=np.int32)

# Build the key classes
if args.logger == "wandb":
    init_args = args
    init_args.AbstractAction = "\n".join([f"{k}: {v}" for k, v in prismataengine.AbstractAction.values.items()])
    tracker = WandBTracker(args.name, init_args)
else:
    tracker = ConsoleTracker(args.name, args)

game_player = GamePlayer(args, shared_obs, shared_legals)

if args.model == "mlp":
    model = MLPBase(args.num_obs, args.num_actions, args.hidden_dim)
elif args.model == "d2rl":
    model= D2RLNet(args.num_obs, args.num_actions, args.hidden_dim, args.num_layers)
else:
    raise ValueError('Model Not Supported')
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.load_model_dir!=None:
    model.load_state_dict(torch.load(f'{args.load_model_dir}/model.h5', map_location=torch.device(args.device)))

model.to(device)

reward_normalizer = RunningMeanStd(shape=())
if not args.one_hot:
    obs_normalizer = RunningMeanStd(shape=(args.num_obs, ), path=args.load_model_dir)
else:
    obs_normalizer = None

# Main loop
i = 0
for i in range(args.num_iterations):
    if i!=0 and i%10==0:
        game_player.reset(args, shared_obs, shared_legals) #Attempt at hacky workaround to C memory leak
    # Run num_steps of the game in each worker and accumulate results in
    # the data arrays
    game_player.run_rollout(args, shared_obs, shared_legals, rewards, discounted_rewards,
                            values, policy_probs, actions, model,
                            obs_normalizer, device, episode_ends,i)

    observations = shared_obs.copy()
    legals = shared_legals.copy()

    rewards = apply_normalizer(rewards, reward_normalizer,
                                   update_data=discounted_rewards,
                                   center=False)

    # Compute advantages and future discounted rewards with GAE
    advantages = gae(rewards, values, episode_ends, args.gamma, args.lam)
    advantages = advantages.astype(np.float32)
    rewards_to_go = advantages + values

    # normalize advantages
    raw_advantages = advantages.copy()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # Split the data into batches in the num_workers dimension
    for batch in range(args.num_batches):
        start = batch * args.batch_size
        end = (batch + 1) * args.batch_size

        train_data = (advantages, rewards_to_go, values, actions, observations, legals,
                      policy_probs)
        # slice out batches from train_data
        batch_data = [x[start:end] for x in train_data]
        batch_data = [torch.tensor(x).to(device) for x in batch_data]

        # flatten (batch_size,num_steps,...) into ((batch_size*num_steps,...)
        batch_data = [x.reshape((-1, ) + x.shape[2:]) for x in batch_data]

        # Step batch
        for epoch in range(args.ppo_epochs):
            try:
                #with torch.autograd.detect_anomaly():
                train_step(model, optim, batch_data, args, i, tracker)
            except Exception as e:
                #import pdb
                print(e)
                #pdb.set_trace()
        #for reward in rewards:
        #    print(max(reward))
    tracker.log_iteration_time(args.num_workers * args.num_steps, i)
    if i % 5 == 0:
        tracker.add_histogram("episode/episode_length",
                              game_player.episode_length, i)
        tracker.add_histogram("episode/episode_rewards",
                              game_player.episode_rewards, i)
        tracker.add_histogram("episode/episode_winners",
                              game_player.episode_winners, i)
    if i % 25 == 0:
        tracker.add_histogram("training/raw_advantages",
                              raw_advantages, i)
        tracker.add_histogram("training/rewards",
                              rewards, i)
        tracker.add_histogram("training/observations",
                              observations, i)
        tracker.add_histogram("training/reward_std",
                              np.sqrt(reward_normalizer.var), i)

    if i%args.model_save_interval==0:
        torch.save(model.state_dict(), f"{args.model_dir}/{i}_model_{runid}.h5")
        if not args.one_hot:
            np.save(f"{args.model_dir}/{i}_mean_{runid}", obs_normalizer.mean)
            np.save(f"{args.model_dir}/{i}_var_{runid}", obs_normalizer.var)
