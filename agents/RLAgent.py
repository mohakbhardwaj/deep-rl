#!/usr/bin/env python
"""Base class for all RL learners"""

class RLAgent():
	      
    def learn(self):
		raise NotImplementedError

    def save_log_to_csv(self):
		raise NotImplementedError
