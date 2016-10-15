#!/usr/bin/env python
"""
	Noise models for exploration noise in action selection
	Created on: October 14, 2016
	Author: Mohak Bhardwaj
"""
import random
import numpy as np

class NoiseModel(object):
	def noisy_action(self, action):
	"""Must be overriden by derived class"""
		raise NotImplementedError
	# def plot_model():
	# """Must be overriden by derived class"""
	# 	raise NotImplementedError


class GaussianNoiseModel(NoiseModel):
	"""Adds Gaussian Noise to the action"""
	#[TODO: This API will change once the environment implementations are finalized]
	def __init__(self, env_action_dim, env_action_max, env_action_min, sigma_max = 1.0, sigma_min = 0.1, decay_rate = 1000000):
		assert sigma_max > sigma_min
		assert decay_rate > 0
		self.env_action_dim = env_action_dim
		self.env_action_max = env_action_max
		self.env_action_min = env_action_min
		self.sigma_max = sigma_max
		self.sigma_min = sigma_min
		self.decay_rate = decay_rate
	
	def noisy_action(self, action, time):
		"""Returns action perturbed by noise model with 
		variance that decays with time"""
		sigma = self.sigma_max - (self.sigma_max - self.sigma_min)*(max(1, t)/self.decay_rate)
		action_p = np.clip(action + np.random.normal(size=self.env_action_dim)*sigma, self.env_action_min, self.env_action_max)
		return action_p
	
	# def plot_model():



class OUNoiseModel(NoiseModel):
	"""Perturbs action using OU Noise which takes inspiration from Brownian motion
	Link:https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
	"""
	def __init__(self, env_action_dim, env_action_max, env_action_min, mean = 0, sigma = 0.2, theta = 0.15, dt = 1.0):
		assert theta > 0
		assert sigma > 0
		self.env_action_dim = env_action_dim
		self.env_action_max = env_action_max
		self.env_action_min = env_action_min
		self.mean = mean
		self.sigma = sigma
		self.theta = theta
		self.state = np.ones(self.env_action_dim)*self.mean
		self.dt = dt

	def state_propogate(self):
		"""Propogate internal state using OU stochastic differential equation"""
		dx = self.theta*(self.mean - self.state)*seld.dt + self.sigma*np.random.randn(len(self.state))
		self.state = self.state + dx
		return self.state

	def noisy_action(self, action):
		s = self.state_propogate()
		action_p = np.clip(action + s, self.env_action_min, self.env_action_max)
		return action_p
	# def plot_model():




