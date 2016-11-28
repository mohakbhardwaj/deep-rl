#!/usr/bin/env python
"""Deep Q Network with target Network
	Hyperparameters and architecture from "Playing Atari with Deep Reinforcement Learning - Mnih et al."
	https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
	
	Created on: October 14, 2016
	Author: Mohak Bhardwaj
"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import tflearn
import numpy as np
from ValueNetworks import ActionValueNetwork

class DQNetwork(ActionValueNetwork):
	def __init__(self ,\
		num_actions ,\
		num_observations ,\
		batch_size ,\
		discount_factor = 0.90 ,\
		learning_rate = 0.0001 ,\
		num_epochs = 1 ,\
		frameskip = 4 ,\
		frameheight = 84 ,\
		framewidth = 84,
		vision = True):

		self.num_actions = num_actions
		self.num_observations = num_observations
		#Learning parameters
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		#Vision parameters
		self.frameskip = frameskip
		self.frameheight = frameheight
		self.framewidth = framewidth
		self.vision = vision

		self.hidden_fc = 512
		self.sess = tf.Session()

		#Change for compatibility with non-CNN
		# if self.vision:
		# 	self.state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.frameheight, self.framewidth])

		# self.model = self.create_network()
		# self.train_net = self.init_graph()
		self.graph_ops = self.init_graph()

	def create_network(self):
		"""Constructs and initializes core network architecture"""
		if self.vision:
			#Change state to correct dimensions --->Required by tfearn
			state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.frameheight, self.framewidth])
			# state_input = tflearn.input_data(shape=[None, self.frameheight, self.framewidth, self.frameskip])
			# state_input_transpose = tf.transpose(state_input, [0,2,3,1])
			# input_ = tflearn.input_data(state_input)
			net = tf.transpose(state_input, [0,2,3,1])
			net = tflearn.conv_2d(net, 32, 8, strides = 4, activation = 'relu')
			net = tflearn.conv_2d(net, 64, 4, strides = 2, activation =  'relu')
			net = tflearn.conv_2d(net, 64, 3, strides = 1, activation = 'relu')
			net = tflearn.fully_connected(net, self.hidden_fc, activation = 'relu')
			output = tflearn.fully_connected(net, self.num_actions, activation = 'linear')
			return state_input, output
		else:
			sys.stdout.println("Implement this first!")
			raise NotImplementedError

	def init_graph(self):
		"""Overall architecture including target network,
		gradient ops etc"""
		# action_input = tflearn.input_data(tf.float32,shape = [None, self.num_actions])
		state_input, q_value_output = self.create_network()
		network_params = tf.trainable_variables()
		action_input = tf.placeholder("float", [None, self.num_actions])
		target_input = tf.placeholder("float", [None])
		relevant_q_value = tf.reduce_sum(tf.mul(q_value_output, action_input), reduction_indices=1)
		cost = tflearn.mean_square(relevant_q_value, target_input)
		# d_net = tflearn.regression(correct_loss, optimizer = 'adam', loss = 'mean_square', learning_rate = self.learning_rate)
		optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
		# train_net = tflearn.DNN(d_net)
		train_net = optimizer.minimize(cost, var_list=network_params)
		graph_operations = {"s": state_input,\
							"q_value_output": q_value_output,\
							"action_input": action_input
							"target_input": target_input
							"train_net": train_net}
		return graph_operations

 	def train(self, state_batch, target_batch, action_batch):
		action_vectors = self.to_action_input(action_batch)
 		self.train_net.fit([np.transpose(state_batch, [0,2,3,1]), action_vectors], np.asarray(target_batch),\
 		 n_epoch = 1, show_metric = True, batch_size = self.batch_size, snapshot_epoch = False)

 	def get_best_action(self, state):

 		q_values = self.evaluate_values(state)
 		best_action = np.argmax(q_values)
 		return best_action

	def to_action_input(self, action_batch):
		action_vectors = []
		for action in action_batch:
			action_vector = np.zeros(self.num_actions)
			action_vector[action] = 1
			action_vectors.append(action_vector)
		return np.asarray(action_vectors)


 	def evaluate_values(self, input):
 		if self.vision:
			# n_dim_input = np.expand_dims(input, axis=0)
			n_dim_input = np.reshape(input,[-1,input.shape[0],input.shape[1],input.shape[2]])
			# n_dim_concat = np.concatenate([n_dim_input,n_dim_input])
			q_values = self.train_net.predict([np.asarray(np.transpose(n_dim_input, [0,2,3,1]))])

 		else:
 			q_values = self.train_net.predict(input) ##Check implementation in tf

 		return q_values

 	def save_params(self):
 		if self.vision:
 			self.train_net.save("../saved_models/dqn_cnn")
 		else:
 			self.train_net.save("../saved_models/dqn_mlp")

 	def load_params(self):
 		if self.vision:
 			self.train_net.load("../saved_models/dqn_cnn")
 		else:
 			self.train_net.load("../saved_models/dqn_mlp")



# sess = tf.Session()
# o = 3
# a_m = 5
# a_min = 0
# D = DQNConvNet(sess, o, a_m, a_min)
# print("Constructed)")





