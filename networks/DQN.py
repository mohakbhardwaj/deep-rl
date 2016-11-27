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
		self.discount_factor = discount
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		#Vision parameters
		self.frameskip = frameskip
		self.frameheight = frameheight
		self.framewidth = framewidth
		self.vision = vision

		#Change for compatibility with non-CNN
		if vision:
			self.state_input = tf.placeholder(tf.float32,\
				 [None, self.frameskip, self.frameheight, self.framewidth])

		self.model = self.create_network()
		self.train_net = self.init_graph()

	def create_network(self):
		"""Constructs and initializes core network architecture"""
		if self.vision:
			#Change state to correct dimensions --->Required by tfearn
			state_input_transpose = tf.transpose(self.state_input, [0,2,3,1])
			input_ = tflearn.input_data(shape = [None, self.frameskip, self,frameheight, self.framewidth])
			net = tflearn.conv_2d(net, 32, 8, stride = 4, activation = 'relu')
			net = tflearn.conv_2d(net, 64, 4, stride = 2, activation =  'relu')
			net = tflearn.conv_2d(net, 64, 3, stride = 1, activation = 'relu')
			net = tflearn.fully_connected(net, self.hidden_fc, activation = 'relu')
			output = tflearn.fully_connected(net, self.num_actions activation = 'linear')
			return output
		else:
			sys.stdout.println("Implement this first!")
			raise NotImplementedError

	def init_graph(self):
		"""Overall architecture including target network, 
		gradient ops etc"""
		d_net = tflearn.regression(self.model, optimizer = 'adam', loss = 'mean_squared', learning_rate = self.learning_rate)
		train_net = tflearn.DNN(d_net)
		return train_net
			
 	def train(self, state_batch, target_batch):
 		self.train_net.fit(np.asarray(state_batch), np.asarray(target_batch), n_epoch = None, show_metric = True, batch_size = self.batch_size, snapshot_epoch = False)

 	def evaluate_values(self, input):
 		if self.vision:
 			q_values = self.train_net.predict([tf.transpose(input, [0,2,3,1])])
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
 	


sess = tf.Session()
o = 3
a_m = 5
a_min = 0
D = DQNConvNet(sess, o, a_m, a_min)
print("Constructed)")

	



