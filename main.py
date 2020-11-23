"""Main training loop"""
import argparse
import ctypes
import multiprocessing as mp
import pickle

import numpy as np
import torch

from model import MLPBase, D2RLNet, Discrete
from train import train_step
from worker import GamePlayer
from utils import get_gym_env_info, gae
from reward_norm import RunningMeanStd, apply_normalizer
from tracker import WandBTracker, ConsoleTracker

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
parser.add_argument('--num_steps', default=1024, type=int)
parser.add_argument('--ppo_epochs', default=4, type=int)
parser.add_argument('--num_batches', default=4, type=int)
parser.add_argument('--lr', default=2.5e-4, type=float)
parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--end_on_life_loss', default=False)
parser.add_argument('--clip_rewards', default=False)
parser.add_argument('--logger', default="console")
args = parser.parse_args()

args.batch_size = int(args.num_workers / args.num_batches)

args.num_actions, args.obs_shape, args.num_obs = \
    get_gym_env_info(args.env_name)

device = torch.device(args.device)

# Define common shapes for convenience
scalar_shape = (args.num_workers, args.num_steps)
batch_obs_shape = (args.num_workers, args.num_steps, args.num_obs)
batch_legals_shape = (args.num_workers, args.num_steps, args.num_actions)
args.steps_to_skip = 1


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
    tracker = WandBTracker(args.name, args)
else:
    tracker = ConsoleTracker(args.name, args)
    
game_player = GamePlayer(args, shared_obs, shared_legals)

dist = Discrete(args.num_actions)
if args.model == "mlp":
    model = MLPBase(args.num_obs, args.num_actions, dist).to(device)
elif args.model == "d2rl":
    model= D2RLNet(args.num_obs, args.num_actions, dist).to(device)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

reward_normalizer = RunningMeanStd(shape=())
obs_normalizer = RunningMeanStd(shape=(args.num_obs, ))

# Main loop
i = 0  
for i in range(args.num_iterations):
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
            train_step(model, optim, batch_data, args, i, tracker)
        #for reward in rewards:
        #    print(max(reward))
    tracker.log_iteration_time(args.num_workers * args.num_steps, i)
    if i % 5 == 0:
        tracker.add_histogram("episode/episode_length",
                              game_player.episode_length, i)
        tracker.add_histogram("episode/episode_rewards",
                              game_player.episode_rewards, i)
    if i % 25 == 0:
        tracker.add_histogram("training/raw_advantages",
                              raw_advantages, i)
        tracker.add_histogram("training/rewards",
                              rewards, i)
        tracker.add_histogram("training/observations",
                              observations, i)
        tracker.add_histogram("training/reward_std",
                              np.sqrt(reward_normalizer.var), i)
            
    if i%100==0:
        torch.save(model.state_dict(), f"model_{i}.h5")
        np.save(f"mean_{i}", obs_normalizer.mean)
        np.save(f"var_{i}", obs_normalizer.var)
        