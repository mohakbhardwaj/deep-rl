#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent

use_vision = True
render_while_training = False
evaluate = False
monitor_training = False
common_seed = 0
env = Env('Breakout-v0', 84, 84, 4, use_vision)
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
         training_params_file = "dqn_breakout",\
         training_summary_file = "dqn_breakout" ,\
         max_epsilon = 1,\
         min_epsilon = 0.1,\
         min_epsilon_timestep = 1000000,\
         tau = 1 ,\
         vision = use_vision ,\
         warm_start = False ,\
         network_mode = "cpu" ,\
         seed = common_seed)

if evaluate:
  env.start_monitor('../data/env_monitor/dqn/cartpole')
  agent.test(max_episodes = 200, render_env = True)
  env.close_monitor()
else:
  env.setSeed(common_seed)
  if monitor_training:
    env.start_monitor('../data/env_monitor/dqn/cartpole')
    agent.learn(render_while_training)
    env.close_monitor()
  else:
    agent.learn(render_while_training)