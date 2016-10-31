#!/usr/bin/env python
"""Class implements DDPG Agents compatible
with openAI gym environment interface"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from RLAgent import RLAgent
from networks.DDPGActor import DDPGConvNetActor
from networks.DDPGCritic import DDPGConvNetCritic
from rl_common.ReplayBuffer import SimpleBuffer
import gym
import numpy as np

class DQAgent(RLAgent):
	def __init__(self
				, env
				, discount_factor = 0.90
				, actor_type = "mlpactor"
				, actor_learning_rate = 0.0001
				, critic_type = "mlpcritic"
				, critic_learning_rate = 0.0001):
		self.env = env
		self.discount_factor = discount_factor
		self.actor_type = actor_type
		self.actor_learning_rate = actor_learning_rate
		self.critic_learning_rate = critic_learning_rate

	def learn(self):
		print "Implemented"


# env = gym.make('CartPole-v0')
# Agent = DDPGAgent(env)
# Agent.learn()
