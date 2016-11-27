#!/usr/bin/env python
"""Class implements simple Deep Q learning agent compatible
with openAI gym environment interface"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
from RLAgent import RLAgent
from networks.DQN import DQNetwork
from rl_common.ReplayBuffer import SimpleBuffer
from rl_common.NoiseModel import *
import gym
import numpy as np

class DQAgent(RLAgent):
	def __init__(self,\
				 env,\
				 discount_factor = 0.90 ,\
				 learning_rate = 0.0001,\
				 max_training_steps = 10000000,\
				 steps_per_epoch = 6000,\
				 buffer_size = 1000000,\
				 batch_size = 32,\
				 scale_rewards = True,\
				 vision = True):
		
		#Learning parameters
		self.env = env
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.max_training_steps = max_training_steps
		self.steps_per_epoch = steps_per_epoch
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.replay_buffer = SimpleBuffer(self.buffer_size)
		self.scale_rewards = scale_rewards
		self.vision = vision
		# self.exploration_strategy = EpsilonGreedy(...)
		# self.sess = tf.session()
		self.network = DQNetwork(env.num_actions,\
								env.observation_length,\
								self.discount_factor,\
								self.learning_rate,\
								1,\
								self.env.history_length,\
								self.env.h_out,\
								self.env.w_out,\
								self.vision)

	def learn(self):
		t = 0
		state = self.env.reset() #Get the initial state

		while t < self.max_training_steps:
			if t%self.steps_per_epoch == 0:
				print("Epoch Done")
				#Print out stats here
			while self.replay_buffer.
			next_state, reward, terminal, info = self.env.step 

	def clip_reward(self, reward, upper, lower):
		return np.clip(reward, lower, upper)









#[TODO:] Implement Epsilon Greedy Noise Model
#[TODO]
# env = gym.make('CartPole-v0')
# Agent = DDPGAgent(env)
# Agent.learn()
