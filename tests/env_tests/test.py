#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from EnvWrapper import Env


#Class creating test
env = Env('CartPole-v0', True)
env.initialize(1234)
print "done"