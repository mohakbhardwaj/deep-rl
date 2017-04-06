#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('..'))

from environments.EnvWrapper import Env
from agents.BehaviorCloningAgent import BehaviorCloningAgent

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_policy_file', type=str)
  parser.add_argument('envname', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument("--max_timesteps", type=int)
  parser.add_argument('--num_episodes', type=int, default=20,
                      help='Number of expert roll outs if generating training data')
  parser.add_argument("--vision_based", type=bool, default=False,
                      help='Whether to use images as observations or not')
  # parser.add_argument("--train", type=bool, default=False ,\
  #                     help='Training')
  # parser.add_argument("--test", type=bool, default=False ,\
  #                     help = 'Testing')
  
  args = parser.parse_args()

  evaluate = False
  monitor_training = True
  common_seed = 0
  min_data_points = 40000 #Do not train if database file has less than this amount
  num_testing_episodes = 20
  env = Env(args.envname, 84, 84, 1, args.vision_based)
  max_steps = args.max_timesteps or env.timestep_limit
  if evaluate or monitor_training:
    env.Monitor('../data/env_monitor/supervised/' + args.envname)
  env.setSeed(common_seed)
  
  #Need to do this as tensorflow hogs the gpu sometimes
  if args.render:
    env.render()

  agent = BehaviorCloningAgent( env,\
           expert_policy_file = args.expert_policy_file,\
           max_training_episodes = args.num_episodes,\
           timesteps_per_episode = max_steps,\
           learning_rate = 0.001,\
           batch_size = 64,\
           training_epochs = 20,\
           training_params_file = "bc_"+ args.envname,\
           training_summary_file = "bc_"+ args.envname ,\
           database_file = "database-" + args.envname ,\
           seed = common_seed ,\
           render = args.render)

  
  if evaluate:
    agent.test(num_testing_episodes)
  else:
    agent.learn(min_data_points)

if __name__ == '__main__':
    main()
