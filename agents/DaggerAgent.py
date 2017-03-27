#!/usr/bin/env python
"""Class implements a Behvior Cloning agent that takes in an expert policy and 
learns a neural network to mimick the expert with supervised learning"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import RLAgent
from imitation_learning.load_policy import load_policy 
from imitation_learning import tf_util
from networks.SupervisedLearningNetwork import SupervisedLearningNetwork
# from netoworks
import gym
import tensorflow as tf
import numpy as np
import pickle

class DaggerAgent():
  def __init__(self,\
             env ,\
             expert_policy_file ,\
             max_training_episodes = 20 ,\
             timesteps_per_episode = 2000 ,\
             mixing_ratio = 0.5 ,\
             learning_rate = 0.001 ,\
             batch_size = 32 ,\
             training_epochs = 15,\
             training_params_file = "dagger_1" ,\
             training_summary_file = "dagger_1" ,\
             seed = 1234 ,\
             ):
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))#, log_device_placement=True))
    self.env = env
    self.max_training_episodes = max_training_episodes
    self.timesteps_per_episode = timesteps_per_episode
    self.mixing_ratio = mixing_ratio
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.training_epochs = training_epochs
    self.seed = seed
    self.training_params_file = training_params_file
    self.summary_dir = "../data/summaries/dagger/" + training_summary_file
    
    tf.set_random_seed(seed)
    np.random.seed(seed)
    # print self.env.observation_dim
    
    self.network = SupervisedLearningNetwork(self.sess ,\
                                           self.env.action_dim ,\
                                           self.env.observation_dim ,\
                                           self.learning_rate ,\
                                           self.batch_size ,\
                                           self.training_epochs ,\
                                           self.env.history_length ,\
                                           self.env.h_out ,\
                                           self.env.w_out)
    print("loading and building expert policy")
    self.expert_policy_fn = load_policy("../" + expert_policy_file)
    print('loaded and built')
    
  
  def learn(self, render_env = False):
    with self.sess:
      database = []
      returns = []
      observations = []
      actions = []      
      for i in xrange(self.max_training_episodes):
        print('Episode', i)
        obs = self.env.reset()
        done = False
        totalr = 0.
        steps = 0
        #Mixing expert and learnt policy
        if i == 0:
          beta = 0
        else:
          if np.random.sample(1) > self.mixing_ratio:
            beta = 1 #For learner
            print('Chose Learner')
          else:
            beta = 0 #For expert
            print('Chose Expert')
        while not done:
          action = self.expert_policy_fn(obs)  
          database.append((obs,action))  
          if beta == 1:
            action = self.network.get_best_action(obs)
          actions.append(action) #Chosen action stored
          observations.append(obs) 
          obs, reward, done, _ = self.env.step(action)
          totalr += reward
          steps += 1
          if render_env:
            self.env.render()
          if steps % 100 == 0: print("%i/%i"%(steps, self.timesteps_per_episode))
          if steps >= self.timesteps_per_episode:
            break
        returns.append(totalr)
        print('Training to imitate expert')
        self.network.train(database)
      print('returns', returns)
      print('mean returns', np.mean(returns))
      print('std of returns', np.std(returns))   
      print('network trained, storing results')
      self.network.save_params(self.training_params_file)
      print('learnt params saved')
      # self.test(10)
  # self.network.save_summaries()



  def test(self, max_testing_episodes = 20, render_env = True):
    returns = []
    observations = []
    actions = []
    expert_actions = []
    with self.sess:
      print "[INFO: Loading parameters]"
      try:
        self.network.load_params(self.training_params_file)
        print "[INFO: Weights Loaded]"
      except:
        print "[ERR: Loading failed!]"

      for i in xrange(max_testing_episodes):
        print('Episode', i)
        obs = self.env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
          action = self.network.get_best_action(obs)
          expert_action = self.expert_policy_fn(obs)
          observations.append(obs)
          actions.append(action)
          expert_actions.append(expert_action)
          obs, reward, done, _ = self.env.step(action)
          totalr += reward
          steps += 1
          if render_env:
            self.env.render()
          if steps % 100 == 0: print("%i/%i"%(steps, self.timesteps_per_episode))
          if steps >= self.timesteps_per_episode:
            break
        returns.append(totalr)
      print('returns', returns)
      print('mean returns', np.mean(returns))
      print('std of returns', np.std(returns))
      print('mean error in actions', self.network.calculate_error(observations , expert_actions))
      print('mean error in actions along individual dimensions', self.network.calculate_error_each_dim(actions, expert_actions))
