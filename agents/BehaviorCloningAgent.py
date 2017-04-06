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

class BehaviorCloningAgent():
  def __init__(self,\
             env ,\
             expert_policy_file ,\
             max_training_episodes = 20 ,\
             timesteps_per_episode = 2000 ,\
             learning_rate = 0.001 ,\
             batch_size = 32 ,\
             training_epochs = 15,\
             training_params_file = "bc_1" ,\
             training_summary_file = "bc_1" ,\
             database_file = "database" ,\
             seed = 1234 ,\
             render = False
             ):
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))#, log_device_placement=True))
    self.env = env
    self.max_training_episodes = max_training_episodes
    self.timesteps_per_episode = timesteps_per_episode
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.training_epochs = training_epochs
    self.seed = seed
    self.training_params_file = training_params_file
    self.summary_dir = "../data/summaries/behavior_cloning/" + training_summary_file
    self.database_file_name = "../imitation_learning/training_data/" + database_file + ".pkl"
    self.render_env = render
    # tf.set_random_seed(seed)
    np.random.seed(seed)
    # print self.env.observation_dim
    # self.sess = tf.Session(tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    
    self.network = SupervisedLearningNetwork(self.sess ,\
                                           self.env.action_dim ,\
                                           self.env.observation_dim ,\
                                           self.learning_rate ,\
                                           self.batch_size ,\
                                           self.training_epochs ,\
                                           self.env.history_length ,\
                                           self.env.h_out ,\
                                           self.env.w_out ,\
                                           mode = 'gpu')
    print("loading and building expert policy")
    self.expert_policy_fn = load_policy("../" + expert_policy_file)
    print('loaded and built')
    
  
  def learn(self, min_data_points = 20000):
    with self.sess:
      database = self.try_load_database(self.database_file_name)
      database_size = len(database) 
      print("Database size", database_size)
      if database_size < min_data_points:
        print('sufficient expert data does not exist, creating roll-outs')
        database = []
        expert_returns = []
        observations = []
        actions = []      
        for i in xrange(self.max_training_episodes):
          print('Episode', i)
          obs = self.env.reset()
          done = False
          totalr = 0.
          steps = 0
          while not done:
            action = self.expert_policy_fn(obs)
            observations.append(obs)
            actions.append(action)
            database.append((obs,action))
            obs, reward, done, _ = self.env.step(action)
            totalr += reward
            steps += 1
            if self.render_env:
              self.env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, self.timesteps_per_episode))
            if steps >= self.timesteps_per_episode:
              break
          expert_returns.append(totalr)
        print('expert returns', expert_returns)
        print('mean expert returns', np.mean(expert_returns))
        print('std of expert returns', np.std(expert_returns))
        print('storing database of expert rollouts. Number of datapoints', len(database))
        self.write_database_to_file(database, self.database_file_name)
      else:
        print('sufficient expert data already exists! Number of datapoints', len(database))    
      print('Training to imitate expert')
      self.network.train(database)
      print('network trained, storing results')
      self.network.save_params(self.training_params_file)
      print('learnt params saved')
      # self.test(10)
  # self.network.save_summaries()



  def test(self, num_testing_episodes = 20):
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
          if self.render_env:
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
  
  def try_load_database(self, file_name):
    try:
      with open(file_name, 'rb') as f:
        database = pickle.load(f) 
    except IOError:
      database = []
    return database

  def write_database_to_file(self, database, file_name):
    with open(file_name, 'wb') as f:
      pickle.dump(database, f)

  
  # def build_summaries(self):
  #   returns = 

