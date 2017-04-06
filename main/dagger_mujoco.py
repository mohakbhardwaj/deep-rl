#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('..'))

from environments.EnvWrapper import Env
from agents.DaggerAgent import DaggerAgent

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_policy_file', type=str)
  parser.add_argument('envname', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument("--max_timesteps", type=int)
  parser.add_argument('--num_episodes', type=int, default=20,
                      help='Number of training episodes')
  parser.add_argument("--vision_input", type=bool, default=False,
                      help='Whether to use images as observations or not')
  
  args = parser.parse_args()

  evaluate = True
  monitor_training = True
  common_seed = 1234
  num_testing_episodes = 150
  env = Env(args.envname, 84, 84, 1, args.vision_input)
  max_steps = args.max_timesteps or env.timestep_limit
  if evaluate or monitor_training:
    env.Monitor('../data/env_monitor/supervised/dagger' + args.envname)
  env.setSeed(common_seed)

  #Need to do this as tensorflow hogs the gpu sometimes
  if args.render:
    env.render()
  
  agent = DaggerAgent( env,\
           expert_policy_file = args.expert_policy_file,\
           max_training_episodes = args.num_episodes,\
           timesteps_per_episode = max_steps,\
           mixing_ratio = 0.1,\
           learning_rate = 0.001,\
           batch_size = 64,\
           training_epochs = 10,\
           training_params_file = "dagger_"+ args.envname,\
           training_summary_file = "dagger_"+ args.envname ,\
           seed = common_seed ,\
           render = args.render)
 
  if evaluate:
    agent.test(num_testing_episodes)
  else:
    agent.learn()

if __name__ == '__main__':
    main()
