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
		self.max_epsilon = 1
		self.min_epsilon = 0.1
		self.min_epsilon_timestep = 1000000
		self.num_actions =env.num_actions
		self.exploration_strategy = EpsilonGreedy(self.max_epsilon,self.min_epsilon,self.min_epsilon_timestep,self.num_actions)
		# self.sess = tf.session()
		
		self.network = DQNetwork(self.num_actions,\
								env.observation_length,\
								self.discount_factor,\
								#Give the same batch size to NN so that tflearn regression does not impose additional batch_size
								self.batch_size, \
								self.learning_rate,\
								1,\
								self.env.history_length,\
								self.env.h_out,\
								self.env.w_out,\
								self.vision)

	def learn(self):
		timestep = 0
		curr_state = self.env.reset() #Get the initial state
		while timestep < self.max_training_steps:
			action = 0
			if timestep%self.steps_per_epoch == 0:
				print("Epoch Done")
				#Print out stats here
			#Initially just do random actions till you have enough frmaes to actually update
			if self.replay_buffer.size() < self.batch_size:
				action = self.env.sample_action() #Or you could just keep epsilon as 1
			#Otherwise follow the exploration strategy
			else:
				best_action = self.network.get_best_action(curr_state)
				action = self.exploration_strategy.get_action(timestep,best_action)

			# Apply the action in the environment
			next_state, reward, terminal, info = self.env.step(action)
			
			#Clip the reward 
			if self.clip_reward:
				reward = self.clip_reward(reward)

			#Append the experience to replay buffer
			self.replay_buffer.add(curr_state, action, reward, terminal, next_state)

			#Again check if training can be done
			if self.replay_buffer.size() >= self.batch_size:
				#Sample a batch form experience buffer
				#Calculate targets from the batch
				
				s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.batch_size)
				
				#While calculating targets if some state is terminal, then target must be R(s) and not 
				# R(s) + gamma x Q*(s',a')
				target_batch = r_batch
				for term,s2,target,action in zip(t_batch,s2_batch,target_batch,a_batch):
					if not term:
						q_vals = self.network.evaluate_values(s2)
						max_q = np.max(q_vals)
						lookahead = self.discount_factor * max_q
						target+=lookahead

				#Send the state batch and target batch to DQN
				self.network.train(s_batch, target_batch, a_batch)

			if terminal:
				#Begin a new episode if reached terminal episode
				curr_state = self.env.reset()
			else:
				curr_state = next_state
			#Don't forget to update the time counter!
			timestep += 1

	def clip_reward(self, reward):
		return np.sign(reward)









#[TODO:] Implement Epsilon Greedy Noise Model
#[TODO]
# env = gym.make('CartPole-v0')
# Agent = DDPGAgent(env)
# Agent.learn()
