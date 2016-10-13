#!/usr/bin/env python
"""Wrapper class to instantiate environment with interface similar to openai gym"""
#[TODO: current implementation returns an environment object with same name as the corresponding
# open ai gym environment. Change this to be applicable for any environment in database of 
# environments]
#[TODO: Accept environment name, params if any --> lookup in database file --> import appropriate env module --> return object]
#[TODO: Wrap all the required functions for gym environment]
import gym
import numpy as np

class Env(object):
	def __init__(self, env_name, gymEnv=True):
		self.env_name = env_name
		self.gymEnv = gymEnv
		self.seed = 1234

	def initialize(self, envSeed):
		if(self.gymEnv):
			self.env = gym.make(self.env_name)
			self.env.seed(envSeed) #This should be an input controllable by user
		

	def setSeed(self, seedVal):
		self.env.seed(seedVal)







