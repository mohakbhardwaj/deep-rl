#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('envname', type=str)
  parser.add_argument("--max_timesteps", type=int)
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--vision_input', action = 'store_true')
  args = parser.parse_args()
  
  evaluate = False
  monitor_training = False
  common_seed = 1234
  env = Env(args.envname, 84, 84, 4, args.vision_input)
  
  assert type(env.observation_space) == gym.spaces.Box
  assert type(env.action_space) == gym.spaces.Discrete
  
  max_steps = args.max_timesteps or env.timestep_limit
  if evaluate or monitor_training:
    env.Monitor('../data/env_monitor/dqn' + args.envname)
  env.setSeed(common_seed)

  #Need to do this as tensorflow hogs the gpu sometimes
  if args.render:
    env.render()

  agent = DQAgent( env,\
           discount_factor = 0.99 ,\
           learning_rate = 0.00025,\
           max_training_steps = 10000000,\
           steps_per_epoch = 6000,\
           buffer_size = 1000000,\
           start_training_after = 50000 ,\
           batch_size = 32,\
           target_network_update = 10000 ,\
           gradient_update_frequency = 4 ,\
           clip_rewards = True,\
           save_after_episodes = 6,\
           training_params_file = "dqn_" + args.envname,\
           training_summary_file = "dqn_" + args.envname ,\
           max_epsilon = 1,\
           min_epsilon = 0.1,\
           min_epsilon_timestep = 1000000,\
           max_episode_length = max_steps ,\
           tau = 1 ,\
           vision = args.vision_input ,\
           warm_start = False ,\
           network_mode = "gpu" ,\
           seed = common_seed ,\
           render = args.render)

  if evaluate:
    agent.test(max_episodes = 200)
  else:
    agent.learn()

if __name__ == '__main__':
  main()