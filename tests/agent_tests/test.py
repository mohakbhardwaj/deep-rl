#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from RLAgent import RLAgent

# #Basic test
# learner = RLLearners()
# learner.learn()

# Class definition test
class DummyAgent(RLAgent):

	def learn(self):
		print("Implemented")
		return 0

agent = DummyAgent()
agent.learn()