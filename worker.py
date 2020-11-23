"""Environment Parallelization from https://github.com/zplizzi/pytorch-ppo"""

from collections import deque
import multiprocessing as mp

import gym, gym_prismata
import numpy as np
import torch
import sys
from torchvision import transforms

import traceback
from sys import exc_info

from reward_norm import apply_normalizer
class GamePlayer:
    """A manager class for running multiple game-playing processes."""
    def __init__(self, args, shared_obs, shared_legals):
        self.episode_length = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)

        # Start game-playing processes
        self.processes = []
        for i in range(args.num_workers):
            parent_conn, child_conn = mp.Pipe()
            worker = SubprocWorker(i, child_conn, args, shared_obs, shared_legals)
            ps = mp.Process(target=worker.run)
            ps.start()
            self.processes.append((ps, parent_conn))


    def run_rollout(self, args, shared_obs, shared_legals, rewards, discounted_rewards, values,
                    policy_probs, actions, model, obs_normalizer, device,
                    episode_ends,i):
        model.eval()
        # Start with the actions selected at the end of the previous iteration
        #step_actions = actions[:, -1]
        # Same with obs and legals
        #shared_obs[:, 0] = shared_obs[:, -1]
        #shared_legals[:, 0] = shared_legals[:, -1]

        for step in range(args.num_steps):
            if step==0: #Initialize the very first environments and obtain initial state
                for j, (p, pipe) in enumerate(self.processes):
                    pipe.send(("start", step, None))
                for j, (p, pipe) in enumerate(self.processes):
                    _ = pipe.recv()
            obs = shared_obs[:, step]
            legals = shared_legals[:, step]
                       
            if len(obs.shape) == 2:
                obs = apply_normalizer(obs, obs_normalizer)
                shared_obs[:, step] = obs

            # run the model
            obs_torch = torch.tensor(obs).to(device).float()
            step_values, probs = model(obs_torch)
            #select only legal actions and renormalize
            legals=torch.tensor(legals).float()
            #print(f"{i} {legals}")
            probs=probs.detach().cpu()
            probs=torch.mul(probs,legals)
            probs_sum=torch.sum(probs, dim=1)
            probs = torch.einsum('ij,i->ij', probs , 1/probs_sum)
            dist = torch.distributions.Categorical(probs=probs)
            
            # Sample actions from the policy distribution
            step_actions = dist.sample()
            step_policy_probs = dist.log_prob(step_actions)

            # Store data for use in training
            step_actions = step_actions.detach().cpu().numpy()
            values[:, step] = step_values.detach().cpu().numpy().flatten()
            policy_probs[:, step] = step_policy_probs.detach().cpu().numpy()
            actions[:, step] = step_actions
            
            # Send the selected actions to workers and request a step
            for j, (p, pipe) in enumerate(self.processes):
                pipe.send(("step", step, step_actions[j]))

            # Receive step data from workers
            for j, (p, pipe) in enumerate(self.processes):
                (reward, discounted_reward, done, info) = pipe.recv()
                rewards[j, step] = reward
                discounted_rewards[j, step] = discounted_reward
                episode_ends[j, step] = done
                try:
                    self.episode_length.append(info['final_episode_length'])
                    self.episode_rewards.append(info['final_episode_rewards'])
                except KeyError:
                    pass
                
class SubprocWorker:
    """A worker for running an environment, intended to be run on a separate
    process."""
    def __init__(self, index, pipe, args, shared_obs, shared_legals):
        self.index = index
        self.pipe = pipe
        self.episode_steps = 0
        self.episode_rewards = 0
        self.disc_ep_rewards = 0
        self.args = args
        self.shared_obs = shared_obs
        self.shared_legals = shared_legals
        #self.num_actions=4
        #self.env = gym.make(self.args.env_name)
        #self.env.reset()
        


    def run(self):
        """The worker entrypoint, will wait for commands from the main
        process and execute them."""
        try:
            while True:
                cmd, t, action = self.pipe.recv()
                if cmd == 'step':
                    try:
                        self.pipe.send(self.step(action, t))
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = exc_info()
                        print(traceback.format_exc())
                        print("PrismataEnvStep[{}]: {} {} {}".format(exc_tb.tb_lineno, type(e).__name__, e))
                        sys.exit(1)
                elif cmd == 'start':
                    self.pipe.send(self.start(t))
                elif cmd == 'close':
                    self.pipe.send(None)
                    break
                else:
                    raise RuntimeError('Got unrecognized cmd %s' % cmd)
        except KeyboardInterrupt:
            print('worker: got KeyboardInterrupt')
        finally:
            self.env.close()
            
    def start(self,t):
        self.env = gym.make(self.args.env_name)
        obs, legal = self.env.reset()
        #obs = self.env.reset()
        #legal = np.ones(self.num_actions)
        self.shared_obs[self.index, t] = obs
        self.shared_legals[self.index, t] = legal
        return None

    def step(self, action, t):
        """Perform a single step of the environment."""
        info = {}
        step_reward = 0
        obs, legal, reward, done = self.env.step(action)
        #obs, reward, done, _ = self.env.step(action)
        #legal=np.ones(self.num_actions)
        self.episode_rewards += reward
        step_reward += reward
        if self.episode_rewards>10000:
            print(self.episode_rewards)
            print(self.episode_steps)
        if done or t==self.args.num_steps-1:
            done=True
            info["final_episode_length"] = self.episode_steps
            info["final_episode_rewards"] = self.episode_rewards
            if __debug__:
                print('Game over, resetting environment')
            obs, legal = self.env.reset()
            #obs = self.env.reset()
            #legal = np.ones(self.num_actions)

            self.episode_steps = 0
            self.episode_rewards = 0

        self.episode_steps += 1

        # We store the observation in t+1 because it's really the next
        # step's observation
        if t < self.args.num_steps - 1:
            self.shared_obs[self.index, t+1] = obs
            self.shared_legals[self.index, t+1] = legal

        if self.args.clip_rewards:
            # clip reward to one of {-1, 0, 1}
            step_reward = np.sign(step_reward)

        self.disc_ep_rewards = self.disc_ep_rewards * self.args.gamma + step_reward
        last_disc_reward = self.disc_ep_rewards
        if done:
            self.disc_ep_rewards = 0
        return step_reward, last_disc_reward, done, info