#!/usr/bin/env python
"""Neural network policy to be trained in a supervised way
Created on: March 25, 2017
Author: Mohak Bhardwaj"""

import os 
import sys
sys.path.insert(0, os.path.abspath('..'))


import tensorflow as tf
import tflearn
import numpy as np
import random
tf.logging.set_verbosity(tf.logging.INFO)

class SupervisedLearningNetwork():
  def __init__(self,\
             sess ,\
             action_dim,\
             observation_dim,\
             learning_rate = 0.001,\
             batch_size = 32,\
             training_epochs = 15,\
             frameskip = 1,\
             frameheight = 84,\
             framewidth = 84,\
             vision = False ,\
             mode = "cpu"):
    
    self.sess = sess
    self.action_dim = action_dim
    self.observation_dim = observation_dim
    self.learning_rate = learning_rate
    self.batch_size =  batch_size
    self.training_epochs = training_epochs
    self.frameskip = frameskip
    self.frameheight = frameheight
    self.framewidth = framewidth
    self.vision = vision
    self.display_step = 1


    if mode == "gpu":
      self.device = '/gpu:0'
    else:
      self.device = '/cpu:0'
    with tf.device(self.device):
      self.graph_ops = self.init_graph()
      self.init_op = tf.initialize_all_variables()
    self.sess.run(self.init_op)
    print('network created and initialized')

  def create_network(self):
    """Constructs and initializes core network architecture"""
    if self.vision:
      raise NotImplementedError
    else:
      state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.observation_dim])
      net = tflearn.fully_connected(state_input, 400, activation='relu')
      net = tflearn.fully_connected(net, 300, activation ='relu')
      # net = tflearn.fully_connected(net, 300, activation = 'relu')
      output = tflearn.fully_connected(net, self.action_dim, activation = 'linear')
    return state_input, output


  def init_graph(self):
    """Overall architecture including target network,
    gradient ops etc"""
    state_input, output = self.create_network()
    network_params = tf.trainable_variables()
    target = tf.placeholder(tf.float32, [None, self.frameskip, self.action_dim])
    cost = tf.reduce_sum(tf.pow(output - target, 2))/(2*self.batch_size)
    optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
    train_net = optimizer.minimize(cost, var_list = network_params)
    saver = tf.train.Saver()
    graph_operations = {"s": state_input,\
                        "output": output,\
                        "target": target,\
                        "cost": cost,\
                        "train_net": train_net,\
                        "network_params": network_params,\
                        "saver": saver}
    return graph_operations

  def train(self, database):
    #Shuffle the database
    
    for epoch in xrange(self.training_epochs):
      random.shuffle(database)
      avg_cost = 0.
      total_batch = int(len(database)/self.batch_size)
      for i in xrange(total_batch):
        batch_x, batch_y = self.get_next_batch(database, i)
        #Run optimization op(backprop) and cost op(to get loss value)
        _, c = self.sess.run([self.graph_ops['train_net'], self.graph_ops['cost']],\
                             feed_dict = {self.graph_ops['s']:batch_x,\
                                          self.graph_ops['target']:batch_y})
        #Compute Average Loss
        avg_cost+= c/total_batch
      #Display logs per epoch
      if epoch%self.display_step == 0:
        print "epoch:", '%04d' % (epoch+1), "cost=", \
              "{:.9f}".format(avg_cost)
    print('optimization finished!')

  def get_best_action(self, input):
    output = self.graph_ops['output'].eval(session = self.sess, feed_dict={self.graph_ops['s']:[input]})
    return output

  def save_params(self, file_name):
    save_path = self.graph_ops['saver'].save(self.sess, "../data/saved_models/supervised/" + file_name + ".ckpt")
    print("Model saved in file: %s" % save_path)
    

  def load_params(self, file_name):
    self.graph_ops['saver'].restore(self.sess, "../data/saved_models/supervised/" + file_name + ".ckpt")
    print("Weights loaded from file ../data/saved_models/supervised/" + file_name + ".ckpt")
  
  def get_params(self):
    raise NotImplementedError

  def set_params(self, input_params, tau):
    raise NotImplementedError

  def get_next_batch(self, database, i):
    batch = database[i*self.batch_size: (i+1)*self.batch_size]
    batch_x = np.array([_[0] for _ in batch])
    batch_y = np.array([_[1] for _ in batch])
    return batch_x, batch_y
  
  # def build_summaries(self):
  #   episode_reward = tf.Variable(0.)
  #   tf.scalar_summary("Reward", episode_reward)
  #   episode_ave_max_q = tf.Variable(0.)

  def calculate_error(self, observations, ground_truth):
    num_data_points = len(ground_truth)
    observations = np.asarray(observations)
    ground_truth = np.asarray(ground_truth)
    c = self.sess.run(self.graph_ops['cost'] ,\
                     feed_dict = {self.graph_ops['s']: observations,\
                                  self.graph_ops['target']:ground_truth})
    return (c*self.batch_size)/num_data_points
  
  def calculate_error_each_dim(self, actions, expert_actions):
    errors = [0]*self.action_dim
    for a, e_a in zip(actions, expert_actions):
      # print a[0], e_a[0]
      for i in xrange(self.action_dim):        
        e = np.absolute(e_a[0][i] - a[0][i])
        errors[i] += e
    return np.asarray(errors)/len(actions)





