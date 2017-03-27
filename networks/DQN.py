#!/usr/bin/env python
"""Deep Q Network with target Network
  Hyperparameters and architecture from "Human Level Control with Deep Reinforcement Learning - Mnih et al."
  http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
  Created on: October 14, 2016
  Author: Mohak Bhardwaj
"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import tflearn
import numpy as np
from ValueNetworks import ActionValueNetwork
#Activate INFO logs during training
tf.logging.set_verbosity(tf.logging.INFO)

class DQNetwork(ActionValueNetwork):
  def __init__(self ,\
    sess ,\
    num_actions ,\
    num_observations ,\
    batch_size ,\
    discount_factor = 0.90 ,\
    learning_rate = 0.0001 ,\
    num_epochs = 1 ,\
    tau = 0.1 ,\
    frameskip = 4 ,\
    frameheight = 84 ,\
    framewidth = 84,\
    vision = True ,\
    mode = "cpu"):

    self.sess = sess
    self.num_actions = num_actions
    self.num_observations = num_observations
    #Learning parameters
    self.batch_size = batch_size
    self.discount_factor = discount_factor
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs
    self.tau = tau
    #Vision parameters
    self.frameskip = frameskip
    self.frameheight = frameheight
    self.framewidth = framewidth
    self.vision = vision
    if mode == "gpu":
      self.device = '/gpu:0'
    else:
      self.device = '/cpu:0'
    with tf.device(self.device):
      self.graph_ops = self.init_graph()
      self.init_op = tf.initialize_all_variables()
    self.sess.run(self.init_op)
    print("Deep Q Network created and initialized")

  def create_network(self):
    """Constructs and initializes core network architecture"""
    if self.vision:
      #Change state to correct dimensions --->Required by tfearn
      state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.frameheight, self.framewidth])
      net = tf.transpose(state_input, [0,2,3,1])
      net = tflearn.conv_2d(net, 32, 8, strides = 4, activation = 'relu')
      net = tflearn.conv_2d(net, 64, 4, strides = 2, activation =  'relu')
      net = tflearn.conv_2d(net, 64, 3, strides = 1, activation = 'relu')
      net = tflearn.fully_connected(net, 512, activation = 'relu')
      output = tflearn.fully_connected(net, self.num_actions, activation = 'linear')
    else:
      state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.num_observations])
      net = tflearn.fully_connected(state_input, 400, activation = 'relu')
      net = tflearn.fully_connected(net, 300, activation = 'relu')
      output = tflearn.fully_connected(net, self.num_actions, activation = 'linear')
    return state_input, output
  
  def init_graph(self):
    """Overall architecture including target network,
    gradient ops etc"""
    state_input, q_value_output = self.create_network()
    network_params = tf.trainable_variables()
    state_input_t, q_value_output_t = self.create_network()
    network_params_t = tf.trainable_variables()[len(network_params):]
    #Add op for resetting target network parameters (soft-updates)
    reset_params_t = [network_params_t[i].assign(tf.mul(network_params[i], self.tau) +\
    tf.mul(network_params_t[i], 1. - self.tau))
    for i in range(len(network_params_t))]
    #Cost and gradient operations
    action_input = tf.placeholder(shape = [None], dtype = tf.int32)
    action_onehot = tf.one_hot(action_input, self.num_actions, dtype = tf.float32)
    target_input = tf.placeholder("float", [None])
    relevant_q_value = tf.reduce_sum(tf.mul(q_value_output, action_onehot), reduction_indices=1)
    cost = tflearn.mean_square(relevant_q_value, target_input)
    optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, decay = 0.95, momentum = 0.95, epsilon = 0.01)
    train_net = optimizer.minimize(cost, var_list=network_params)
    saver = tf.train.Saver()
    graph_operations = {"s": state_input,\
              "q_value_output": q_value_output,\
              "action_input": action_input,\
              "target_input": target_input,\
              "train_net": train_net,\
              "network_params": network_params,\
              "s_t": state_input_t,\
              "q_value_output_t": q_value_output_t,\
              "network_params_t": network_params_t,\
              "reset_params_t":reset_params_t,\
              "saver": saver}
    return graph_operations

  def train(self, state_batch, target_batch, action_batch):
    # action_vectors = self.to_action_input(action_batch)
    state_input = self.graph_ops["s"]
    target_input = self.graph_ops["target_input"]
    action_input = self.graph_ops["action_input"]
    self.sess.run(self.graph_ops['train_net'], feed_dict={state_input: state_batch, action_input: action_batch, target_input: target_batch})

  def get_best_action(self, state):
    q_values = self.evaluate_values(state)
    best_action = np.argmax(q_values)
    return best_action

  def to_action_input(self, action_batch):
    action_vectors = []
    for action in action_batch:
      action_vector = np.zeros(self.num_actions)
      action_vector[action] = 1
      action_vectors.append(action_vector)
    return np.asarray(action_vectors)

  def evaluate_values(self, input):
    state_input = self.graph_ops["s"]
    q_values= self.graph_ops['q_value_output'].eval(session = self.sess, feed_dict={state_input:[input]})
    return q_values

  def evaluate_values_target(self, input):
    state_input = self.graph_ops["s_t"]
    q_values= self.graph_ops['q_value_output_t'].eval(session = self.sess, feed_dict={state_input:[input]})
    return q_values

  def save_params(self, file_name):
    save_path = self.graph_ops['saver'].save(self.sess, "../data/saved_models/dqn/" + file_name + ".ckpt")
    print("Model saved in file: %s" % save_path)

  def load_params(self, file_name):
    self.graph_ops['saver'].restore(self.sess, "../data/saved_models/dqn/" + file_name + ".ckpt")
    print("Weights loaded from file ../data/saved_models/dqn/" + file_name + ".ckpt")

  def get_params(self):
    params = [self.sess.run(self.graph_ops['network_params'][i]) for i in range(len(self.graph_ops['network_params']))]
    return params

  def set_params(self, input_params, tau):
    assign_op = \
    [self.graph_ops['network_params'][i].assign(tf.mul(input_params[i], tau) +\
    tf.mul(self.graph_ops['network_params'][i], 1. - tau))
    for i in range(len(self.graph_ops['network_params']))]
    self.sess.run(assign_op)

  def update_target_network_params(self):
    self.sess.run(self.graph_ops['reset_params_t'])

  








