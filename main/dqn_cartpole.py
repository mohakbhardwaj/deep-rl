#!/usr/bin/env python
import os
import sys
import gym

sys.path.insert(0, os.path.abspath('..'))

from environments.EnvWrapper import Env
from agents.DQAgent import DQAgent

use_vision = False
render_while_training = False
evaluate = True

env = Env('CartPole-v0', 84, 84, 2, use_vision)
agent = DQAgent( env,\
				 discount_factor = 0.99 ,\
				 learning_rate = 0.001,\
				 max_training_steps = 1000000,\
				 steps_per_epoch = 6000,\
				 buffer_size = 100000,\
				 start_training_after = 1000 ,\
				 batch_size = 64,\
				 target_network_update = 1 ,\
				 gradient_update_frequency = 1 ,\
				 clip_rewards = True,\
				 save_after_episodes = 10,\
				 training_params_file = "dqn_cartpole",\
				 training_log_file = "dqn_cartpole" ,\
				 max_epsilon = 1,\
				 min_epsilon = 0.1,\
				 min_epsilon_timestep = 100000,\
				 tau = 1 ,\
				 vision = use_vision ,\
				 warm_start = False ,\
				 network_mode = "cpu")

if evaluate:
	env.start_monitor('../data/env_monitor/dqn/cartpole')
	agent.test(max_episodes = 200, render_env = True)
	env.close_monitor()
else:
	agent.learn(render_while_training)

