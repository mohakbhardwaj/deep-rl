#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent

use_vision = False
render_while_training = False
evaluate = False
monitor_training = True
common_seed = 0
env = Env('CartPole-v0', 84, 84, 2, use_vision)

agent = DQAgent( env,\
         discount_factor = 0.99 ,\
         learning_rate = 0.001,\
         max_training_steps = 1000000,\
         steps_per_epoch = 6000,\
         buffer_size = 100000,\
         start_training_after = 1000,\
         batch_size = 64,\
         target_network_update = 1 ,\
         gradient_update_frequency = 1 ,\
         clip_rewards = True,\
         save_after_episodes = 10,\
         training_params_file = "dqn_cartpole",\
         training_summary_file = "dqn_cartpole" ,\
         max_epsilon = 1,\
         min_epsilon = 0.1,\
         min_epsilon_timestep = 100000,\
         tau = 1 ,\
         vision = use_vision ,\
         warm_start = False ,\
         network_mode = "cpu" ,\
         seed = common_seed)

if evaluate:
  env.start_monitor('../data/env_monitor/dqn/cartpole')
  agent.test(max_episodes = 200, render_env = True, MAX_EPISODE_LENGTH = 200)
  env.close_monitor()
else:
  env.setSeed(common_seed)
  if monitor_training:
    env.start_monitor('../data/env_monitor/dqn/cartpole')
    agent.learn(render_while_training, MAX_EPISODE_LENGTH = 200)
    env.close_monitor()
  else:
    agent.learn(render_while_training, MAX_EPISODE_LENGTH = 200)