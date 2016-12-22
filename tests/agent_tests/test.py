#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('../..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent


env = Env('Reacher-v1', 96, 96, 2, False)
# env.start_monitor('../env_monitor/dqn-experiment-1')
agent = DQAgent( env,\
				 discount_factor = 0.90 ,\
				 learning_rate = 0.001,\
				 max_training_steps = 60000,\
				 steps_per_epoch = 1000,\
				 buffer_size = 10000,\
				 batch_size = 32,\
				 clip_rewards = False,\
				 save_after_episodes = 3,\
				 training_params_file = "dqn_lunar",\
				 max_epsilon = 1,\
				 min_epsilon = 0.1,\
				 min_epsilon_timestep = 10000,\
				 vision = False ,\
				 warm_start = True)

agent.learn()

# env.close_monitor()
