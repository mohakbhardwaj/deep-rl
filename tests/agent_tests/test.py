#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('../..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent

# from RLAgent import RLAgent

# #Basic test
# learner = RLLearners()
# learner.learn()

# Class definition test
# class DummyAgent(RLAgent):

# 	def learn(self):
# 		print("Implemented")
# 		return 0


env = Env('Breakout-v0', 84, 110, 4, True)
# env.monitor.start('../env_monitor/dqn-experiment-1')
agent = DQAgent(env)

agent.learn()

# env.monitor.close()