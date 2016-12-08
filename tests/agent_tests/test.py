#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('../..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent

# from RLAgent import RLAgent

# #Basic test
# learner = RLLearners()
# learner.learn()

# Class definition test
# class DummyAgent(RLAgent):

# 	def learn(self):
# 		print("Implemented")
# 		return 0


env = Env('Breakout-v0', 84, 110, 4, True)
#env.start_monitor('../env_monitor/dqn-experiment-1')
agent = DQAgent( env,\
				 discount_factor = 0.90 ,\
				 learning_rate = 0.0001,\
				 max_training_steps = 500000,\
				 steps_per_epoch = 1000,\
				 buffer_size = 40000,\
				 batch_size = 32,\
				 clip_rewards = False,\
				 save_after_episodes = 3,\
				 training_params_file = "dqn_atari",\
				 max_epsilon = 1,\
				 min_epsilon = 0.1,\
				 min_epsilon_timestep = 100000,\
				 vision = True ,\
				 warm_start = True)

agent.learn()

#env.close_monitor()
