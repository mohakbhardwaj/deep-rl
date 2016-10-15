#!/usr/bin/env python

"""Module for actor networks for DDPG 
   Two types of actor networks will be used:
   A convolution neural network for raw image input
   A MLP for feature input """
import tensorflow as tf
import numpy as np
import PolicyNetwork


class DDPGActorNetwork(PolicyNetwork):
	def __init__( self
				, env
				, discount_factor = 0.90
				, learning_rate = 0.0001):
		self.env = env
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate


class DDPGConvNetActor(DDPGActorNetwork):
	def __init__(self
				, env
				, discount_factor = 0.90
				, learning_rate = 0.0001):
		super(DDPGActorNetwork, self).__init__(env, discount_factor, learning_rate)
#[TODO: Implement rest of the network]


class DDPGMLPActor(DDPGActorNetwork):
	def __init__(self
				, env
				, discount_factor = 0.90
				, learning_rate = 0.0001):
		super(DDPGActorNetwork, self).__init__(env, discount_factor, learning_rate)



