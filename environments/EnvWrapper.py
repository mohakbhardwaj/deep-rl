#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

class Env(object):
	"""Wrapper class to for Deep RL with OpenAI gym environments
	Overloads the step function, maintains a state buffer and preprocesses
	states"""
	"""Any other environments implemented must follow OpenAI Gym interfaces to be used with the wrapper"""
	#Currently supports preprocessing as described in DQN Atari paper

	def __init__(self, env_name, w_out, h_out, history_length, preprocessing=False):
		self.env_name = env_name
		self.w_out = w_out
		self.h_out = h_out	
		self.preprocessing = preprocessing #Flag can be set false for environments that do not provide camera frames
		self.history_length = history_length
		#The following commands are specific to gym environments,
		#[TODO]change when different environments are also implemented
		self.env = gym.make(env_name)
		self.num_actions = range(self.env.action_space.n)
		self.state_buffer = deque()

	def setSeed(self, seedVal):
		self.env.seed(seedVal)

	def preprocess_frame(self, obs):
		"""Preprocessing involves conversion to grayscale and then rescaling to required
		dimensions"""
		processed_frame = obs
		if self.preprocessing:
			processed_frame = resize(rgb2gray(obs), (self.h_out, self.w_out))
		return processed_frame

	def reset(self):
		"""Reset environment to initial state and clears state buffer"""
		self.state_buffer.clear()
		xt = self.preprocess_frame(self.env.reset())
		st = np.stack((xt, xt, xt, xt), axis=0)
		[self.state_buffer.append(xt) for i in xrange(self.history_length)]
		return st

	def render(self):
		self.env.render()

	def sample_action(self):
		return self.env.action_space.sample()

	def step(self, action):
		"""Execute an action in the environment and return the next reward, observation(set of history_length states)
		and update the current state_buffer"""
		observation, reward, done, info = self.env.step(action)
		if self.preprocessing:
			observation = self.preprocess_frame(observation)
		previous_frames = np.array(self.state_buffer)
		
		if len(previous_frames.shape) < 3:
			previous_frames = previous_frames.reshape(previous_frames.shape[0], 1, previous_frames.shape[1])
		# print observation
		s = np.empty((self.history_length, self.h_out, self.w_out))
		s[:self.history_length, ...] = previous_frames
		s[self.history_length-1] = observation
		#Update the state_buffer
		self.state_buffer.popleft()
		self.state_buffer.append(observation)
		return s, reward, done, info

		













