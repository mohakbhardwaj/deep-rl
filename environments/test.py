#!/usr/bin/env python
from EnvWrapper import Env


#Class creating test
env = Env('CartPole-v0', True)
env.initialize(1234)
print "done"