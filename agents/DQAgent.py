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
import gym
import numpy as np

class DQAgent(RLAgent):
	def __init__(self
				, env
				, discount_factor = 0.90
				, learning_rate = 0.0001
				, buffer_size = 1000000
				, es
				, batch_size = 32
				, scale_rewards = True
				, vision = True):
		
		self.env = env
		self.discount_factor = discount_factor
		self.buffer_size = buffer_size
		self.replay_buffer = SimpleBuffer(buffer_size)
		self.sess = tf.session()
		self.network = DQNetwork(self.sess, len(env.observation_space.high), .......)

	def learn(self):
		#
		# for epoch in self.num_epochs:
		# Handle terminals wisely



#[TODO:] Implement Epsilon Greedy Noise Model
#[TODO]
# env = gym.make('CartPole-v0')
# Agent = DDPGAgent(env)
# Agent.learn()
