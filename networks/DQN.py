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

class DQNConvNet(ActionValueNetwork):
	def __init__(self ,\
		sess ,\
		observation_length ,\
		action_max ,\
		action_min ,\
		action_length ,\
		discount = 0.90 ,\
		learning_rate = 0.0001 ,\
		tau = 0.001 ,\
		frameskip = 4 ,\
		frameheight = 84 ,\
		framewidth = 84):
		self.sess = sess
		self.observation_length = observation_length
		self.action_max = action_max
		self.action_min = action_min
		self.action_length = action_length
		self.discount = discount
		self.learning_rate = learning_rate
		self.tau = tau
		self.frameskip = frameskip
		self.frameheight = frameheight
		self.framewidth = framewidth

	def create_network(self):
		state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.frameheight, self.framewidth])
		#Change state to correct dimensions --->Required by tfearn
		net = tf.transpose(state_input, [0, 2, 3, 1]) 
		net = tflearn.conv_2d(net, 32, 8, stride = 4, activation = 'relu')
		net = tflearn.conv_2d(net, 64, 4, stride = 2, activation =  'relu')
		net = tflearn.conv_2d(net, 64, 3, stride = 1, activation = 'relu')
		net = tflearn.fully_connected(net, 512, activation = 'relu')
		output = tflearn.fully_connected(net, self.action_length, activation = 'linear')
		return output
		


sess = tf.Session()
o = 3
a_m = 5
a_min = 0
D = DQNConvNet(sess, o, a_m, a_min)
print("Constructed)")

	



