#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from environments.EnvWrapper import Env
import cv2
from threading import Thread
import numpy as np
import time

env = Env('AirRaid-v0', 84, 110, 4, True)
si = env.reset()
# print si[-1]
img = np.zeros([84, 110], dtype=int)

working = True

def display():
	while working:
		cv2.imshow("Output", img)
		cv2.waitKey(33)

t1 = Thread(target=display)

t1.start()

for _ in range(1000):
	env.render()
	observation, reward, done, info = env.step(env.sample_action())
	img = observation[-1]
	time.sleep(0.0002)

working = False