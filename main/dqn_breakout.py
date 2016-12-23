#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent

use_vision = True
render_while_training = False

env = Env('Breakout-v0', 84, 110, 4, use_vision)
env.start_monitor('../data/env_monitor/dqn/breakout')
agent = DQAgent( env,\
				 discount_factor = 0.90 ,\
				 learning_rate = 0.0001,\
				 max_training_steps = 10000000,\
				 steps_per_epoch = 6000,\
				 buffer_size = 1000000,\
				 batch_size = 32,\
				 clip_rewards = True,\
				 save_after_episodes = 3,\
				 training_params_file = "dqn_breakout",\
				 training_log_file = "dqn_breakout" ,\
				 max_epsilon = 1,\
				 min_epsilon = 0.1,\
				 min_epsilon_timestep = 1000000,\
				 vision = use_vision ,\
				 warm_start = False ,\
				 network_mode = "cpu")

agent.learn(render_while_training)

env.close_monitor()
