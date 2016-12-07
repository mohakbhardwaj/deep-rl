#!/usr/bin/env python
"""Class implements simple Deep Q learning agent compatible
with openAI gym environment interface"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from RLAgent import RLAgent
from networks.DQN import DQNetwork
from rl_common.ReplayBuffer import SimpleBuffer
from rl_common.NoiseModel import *
import gym
import tensorflow as tf
import numpy as np
import csv

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
				 save_after_episodes = 3,\
				 training_params_file = "dqn_atari",\
				 max_epsilon = 1,\
				 min_epsilon = 0.1,\
				 min_epsilon_timestep = 1000000 ,\
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
		self.save_after_episodes = save_after_episodes
		self.training_params_file = training_params_file
		self.vision = vision
		self.max_epsilon = max_epsilon
		self.min_epsilon = min_epsilon
		self.min_epsilon_timestep = min_epsilon_timestep
		self.num_actions =env.num_actions
		self.agent_log_writer = csv.writer(open('agent_log.csv','w+'))

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

		self.print_csv_header()

	def print_csv_header(self):
		log_info= ['num_episodes_passed', 'cumulative_reward', 'episode_length',\
		 'avg_reward', 'unclipped_episode_reward','unclipped_average_reward']
		self.save_log_to_csv(log_info)

	def save_log_to_csv(self,log):
		row = []
		for log_elt in log:
			row.append(log_elt) 
		self.agent_log_writer.writerow(row)

	def learn(self):
		timestep = 0
		curr_state = self.env.reset() #Get the initial state
		cumulative_reward = 0 #Cumulative rewards in one episode
		avg_reward = 0 #Average reward per episode
		num_episodes_passed = 0
		episode_length = 0
		unclipped_episode_reward = 0
		unclipped_average_reward = 0
		print("Initiate Training")
		# Load params from a saved model
		
		print("Loading parameters")

		try:
			self.network.load_params(self.training_params_file)
			print("Weights Loaded")
		except:
			print("LOADING FAILED!")
		while timestep < self.max_training_steps:
			# print "Step: ",timestep
			action = 0
			# TODO FIX: Giving an error when rendering. 
			# self.env.render()
			if (timestep+1)%self.steps_per_epoch == 0:
				
				print("Epoch Done")
				#Print out stats here
			#Initially just do random actions till you have enough frmaes to actually update
			if self.replay_buffer.size() < self.batch_size:
				action = self.env.sample_action() #Or you could just keep epsilon as 1
			#Otherwise follow the exploration strategy
			else:
				best_action = self.network.get_best_action(curr_state)
				action = self.exploration_strategy.get_action(timestep,best_action)
				# print action, best_action, self.exploration_strategy.epsilon
				if timestep == self.min_epsilon_timestep:
					print self.exploration_strategy.epsilon, min_epsilon

			# Apply the action in the environment
			next_state, reward, terminal, info = self.env.step(action)
			unclipped_episode_reward += reward

			#Clip the reward 
			if self.clip_reward:
				reward = self.clip_reward(reward)
			
			cumulative_reward += reward
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

			#Save data for the current episode to csv
			if terminal:
				#Begin a new episode if reached terminal episode
				curr_state = self.env.reset()
				#Update the average reward
				num_episodes_passed += 1
				# avg_reward = (avg_reward*(num_episodes_passed - 1) + cumulative_reward)/num_episodes_passed
				unclipped_average_reward += (unclipped_episode_reward - unclipped_average_reward)/num_episodes_passed
				avg_reward += (cumulative_reward - avg_reward)/num_episodes_passed
				# print("[INFO], "Number",)  "Cumulative Episode Reward: ",cumulative_reward
				
				log_info= [num_episodes_passed, cumulative_reward, episode_length, avg_reward, unclipped_episode_reward,unclipped_average_reward]
				self.save_log_to_csv(log_info)

				print("[INFO]", "Episode Number: ", num_episodes_passed, "Episode Reward ", cumulative_reward, "Episode Length", episode_length,\
				  "Average Reward Per Episode ", avg_reward, "Episode Reward(unclipped) ", unclipped_episode_reward, "Average Reward Per Episode(unclipped) ",\
				  unclipped_average_reward)

					
				#If num_episodes_passed%save_after_episodes is zero, save the  current learned weights
				if (num_episodes_passed + 1)%self.save_after_episodes == 0:
					print("Saving currently learned weights")
					self.network.save_params(self.training_params_file)
				#Reset episode statistics				
				cumulative_reward = 0
				unclipped_episode_reward = 0
				episode_length = 0		
			else:
				curr_state = next_state
			#Don't forget to update the time counter!

			
			timestep += 1
			episode_length += 1

		
		#Save final model weights after traning complete
		print("Training Done. Saving final model weights")
		self.network.save_params(self.training_params_file)

	def clip_reward(self, reward):
		return np.sign(reward)
