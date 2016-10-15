#!/usr/bin/env python

"""Module for critic networks for DDPG 
   Two types of criic networks will be used:
   A convolution neural network for raw image input
   A MLP for feature input """
import tensorflow as tf
import numpy as np
from ValueNetworks import ActionValueNetwork


class DDPGCriticNetwork(ActionValueNetwork):
	def __init__( self
				, env
				, discount_factor = 0.90
				, learning_rate = 0.0001):
		self.env = env
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate


class DDPGConvNetCritic(DDPGCriticNetwork):
	def __init__(self
				, env
				, discount_factor = 0.90
				, learning_rate = 0.0001):
		super(DDPGCriticNetwork, self).__init__(env, discount_factor, learning_rate)
#[TODO: Implement rest of the network]


class DDPGMLPCritic(DDPGCriticNetwork):
	def __init__(self
				, env
				, discount_factor = 0.90
				, learning_rate = 0.0001):
		super(DDPGCriticNetwork, self).__init__(env, discount_factor, learning_rate)



