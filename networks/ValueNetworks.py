#!/usr/bin/env python

"""Generic class for all Value Networks
	Created on: October 14, 2016
	Author: Mohak Bhardwaj"""

import tensorflow as tf
import tflearn
import numpy as np


class ValueNetwork(object):
	def __init__(self, sess, observation_length, discount = 0.90, learning_rate = 0.0001, tau=0.001, seed = 1234):
		raise NotImplementedError
	def create_v_network(self):
		"""Constructs and initializes neural network"""
		raise NotImplementedError
	def train(self, state_batch, target_batch):
		"""Train using mini-batches as input"""
		raise NotImplementedError
	def evaluate_value(self, state_input):
		"""Return value for a state"""
		raise NotImplementedError
	def update_target_network(self):
		"""Update the target network"""
		pass
	def save_params(self):
		"""Save network parameters to file using tensorflow"""
		pass
	def restore_params(self):
		"""Load network parameters from file using tensorflow"""
		pass



class ActionValueNetwork(object):
	def __init__(self, sess, observation_length, action_max, action_min, action_length, discount = 0.90, learning_rate = 0.0001, tau = 0.001, seed = 1234):
		pass
	def create_network(self):
		"""Constructs and initializes core network architecture"""
		raise NotImplementedError
	def init_graph(self):
		"""Overall architecture includingtarget network, gradient ops etc"""
		raise NotImplementedError
	def train(self, current_state_batch,target_batch):
		"""Train using mini-batches as input"""
		raise NotImplementedError
	def evaluate_value(self, state_input, target_input):
		"""Return value for a state"""
		raise NotImplementedError
	def update_target_network(self):
		"""Update the target network"""
		pass
	def save_params(self):
		"""Save network parameters to file"""
		pass
	def load_params(self):
		"""Load network parameters from file"""
		pass