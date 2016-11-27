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
				 clip_rewards = True,\
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
		self.clip_rewards = clip_rewards
		self.vision = vision
		# self.exploration_strategy = EpsilonGreedy(...)
		# self.sess = tf.session()
		self.network = DQNetwork(env.num_actions,\
								env.observation_length,\
								self.discount_factor,\
								self.batch_size #Give the same batch size to NN so that tflearn regression does not impose additional batch_size
								self.learning_rate,\
								1,\
								self.env.history_length,\
								self.env.h_out,\
								self.env.w_out,\
								self.vision)

	def learn(self):
		t = 0
		curr_state = self.env.reset() #Get the initial state
		while t < self.max_training_steps:
			if t%self.steps_per_epoch == 0:
				print("Epoch Done")
				#Print out stats here
			if self.replay_buffer.size() < self.batch_size:
				#Initially just do random actions till you have enough frmaes to actually update
				action = env.sample_action() #Or you could just keep epsilon as 1
			else:
				# action = #Get from DQN + epsilon greedy
				# anneal epsilon here
			next_state, reward, terminal, info = self.env.step(action)
			#Clip the reward as it does in the DeepMind implementation
			if self.clip_reward:
				reward = self.clip_reward(reward, -1 , 1)
			#Append the experience to replay buffer
			self.replay_buffer.append(curr_state, action, reward, terminal, next_state)

			#Again check if training can be done
			if self.replay_buffer.size() >= self.batch_size:
				#Sample a batch form experience buffer
				#Calculate targets from the batch
				#While calculating targets if some state is terminal, then target must be R(s) and not 
				# R(s) + gamma x Q*(s',a')
				#Send the state batch and target batch to DQN
			if terminal:
				#Begin a new episode if reached terminal episode
				curr_state = self.env.reset()
			else:
				curr_state = next_state
			#Don't forget to update the time counter!
			t += 1

	def clip_reward(self, reward, lower, upper):
		return np.clip(reward, lower, upper)









#[TODO:] Implement Epsilon Greedy Noise Model
#[TODO]
# env = gym.make('CartPole-v0')
# Agent = DDPGAgent(env)
# Agent.learn()
