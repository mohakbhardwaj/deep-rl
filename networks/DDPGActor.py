#!/usr/bin/env python

"""DDPG Actor-Critic Network class with target networks
   In this implementation, the actor and critic networks share the initial layer parameters
  Hyperparameters and architecture from "Continuous Control with Deep Rinforcement Learning- Lillicrap et al."
  https://arxiv.org/pdf/1509.02971v2.pdf
  Created on: December 30, 2016
  Author: Mohak Bhardwaj"""

import os 
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import tflearn
import numpy as np
from ValueNetworks import ActionValueNetwork
#Activate INFO logs during training
tf.logging.set_verbosity(tf.logging.INFO)

class DDPGActorCritic():
  def __init__(self ,\
    sess ,\
    action_dim ,\
    num_observations ,\
    action_bound ,\
    batch_size ,\
    discount_factor = 0.99 ,\
    actor_learning_rate = 0.0001 ,\
    critic_learning_rate = 0.001 ,\
    critic_l2_weight_decay = 0.01 ,\
    num_epochs = 1 ,\
    tau = 0.001 ,\
    frameskip = 3 ,\
    frameheight = 64 ,\
    framewidth = 64,\
    vision = True ,\
    mode = "cpu"):

    self.sess = sess
    self.action_dim = action_dim
    self.num_observations = num_observations
    self.action_bound = action_bound
    self.w_init = w_init
    #Learning parameters
    self.batch_size = batch_size
    self.discount_factor = discount_factor
    self.actor_learning_rate = actor_learning_rate
    self.critic_learning_rate = critic_learning_rate
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
    """Constructs and initializes core network architectures
       For CNNs we share the convolutional layers between actor and critic networks
       For MLP only the first fully connected layer is sahred"""
    w_init_f = tflearn.initializations.uniform_scaling (shape=None, factor=1.0, dtype=tf.float32, seed=None)
      action_input = tf.placeholder("float", [None, self.action_dim])
    if self.vision:
      w_init = tflearn.initializations.uniform(minval=-0.0003, maxval=0.0003)
      #Change state to correct dimensions --->Required by tfearn
      state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.frameheight, self.framewidth])
      net_co = tf.transpose(state_input, [0,2,3,1])
      net_co = tflearn.batch_normalization(net_co)
      net_co = tflearn.conv_2d(net_co, 32, 8, strides = 4, activation = 'relu', weights_init = w_init_f)
      net_co = tflearn.batch_normalization(net_co)
      net_co = tflearn.conv_2d(net_co, 32, 4, strides = 2, activation =  'relu', weights_init = w_init_f)
      net_co = tflearn.batch_normalization(net_co)
      net_co = tflearn.conv_2d(net_co, 32, 3, strides = 1, activation = 'relu', weights_init = w_init_f)
      net_co = tflearn.batch_normalization(net_co)
      common_params = tf.trainable_variables()
      #Here we split the actor and critic networks
      #Remaining actor network
      net_a = tflearn.fully_connected(net_co, 200, activation = 'relu', weights_init = w_init_f)
      net_a = tflearn.batch_normalization(net_a)
      net_a = tflearn.fully_connected(net_a, 200, activation = 'relu', weights_init = w_init_f)
      net_a = tflearn.batch_normalization(net_a)
      output_a = tflearn.fully_connected(net_a, self.action_dim, activation = 'tanh', weights_init = w_init)
      output_a = tf.mul(output_a, self.action_bound)
      actor_seperate_params = tf.trainable_variables()[len(common_params):]
      #Remaining critic network
      t1 = tflearn.fully_connected(net_co, 200, weights_init = w_init_f)
      t2 = tflearn.fully_connected(action_input, 200, weights_init = w_init_f)
      net_c = tflearn.activation(tf.matmul(net_co,t1.W) + tf.matmul(action_input, t2.W) + t2.b, activation='relu')
      net_c = tflearn.fully_connected(net_c, 200, activation = 'relu', weights_init = w_init_f)
      output_c = tflearn.fully_connected(net_c, 1, weights_init = w_init)
      critic_seperate_params = tf.trainable_variables()[len(common_params) + len(actor_seperate_params):]
    else:
      w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
      state_input = tf.placeholder(tf.float32, [None, self.frameskip, self.num_observations])
      state_input = tflearn.batch_normalization(state_input)
      net_co = tflearn.fully_connected(state_input, 400, activation = 'relu', weights_init = w_init_f)
      net_co = tflearn.batch_normalization(net_co)
      common_params = tf.trainable_variables()
      #Here we split the actor and critic networks
      #Remaining actor network
      net_a = tflearn.fully_connected(net, 300, activation = 'relu', weights_init = w_init_f)
      net_a = tflearn.batch_normalization(net_a)
      output_a = tflearn.fully_connected(net_a, self.action_dim, activation = 'tanh', weights_init = w_init)
      output_a = tf.mul(output_a, self.action_bound)
      actor_seperate_params = tf.trainable_variables()[len(common_params):]
      #Remaining critic network
      t1 = tflearn.fully_connected(net_co, 300, weights_init = w_init_f)
      t2 = tflearn.fully_connected(action_input, 200, weights_init = w_init_f)
      net_c = tflearn.activation(tf.matmul(net_co,t1.W) + tf.matmul(action_input, t2.W) + t2.b, activation='relu')
      net_c = tflearn.fully_connected(net_c, 200, activation = 'relu', weights_init = w_init_f)
      output_c = tflearn.fully_connected(net_c, 1, weights_init = w_init)
      critic_seperate_params = tf.trainable_variables()[len(common_params) + len(actor_seperate_params):]
    return state_input, action_input, output_a, output_c, common_params, actor_seperate_params, critic_seperate_params
  
  def init_graph(self):
    """Overall architecture including target network,
    gradient ops etc"""
    state_input, action_input, output_a, output_c, common_params, actor_seperate_params, critic_seperate_params = self.create_network()
    state_input_t, action_input_t, output_a_t, output_c_t, common_params_t, actor_seperate_params_t, critic_seperate_params_t = self.create_network()
    common_params_t = tf.trainable_variables()[len(common_params) + len(actor_seperate_params) + len(critic_seperate_params):len(common_params)+1]
    #Add ops for resetting target network parameters (soft-updates)
    reset_common_params_t = [common_params_t[i].assign(tf.mul(common_params[i], self.tau) +\
    tf.mul(common_params_t[i], 1. - self.tau))
    for i in range(len(common_params))]
    reset_actor_seperate_params_t = [actor_seperate_params_t[i].assign(tf.mul(actor_sperate_params[i], self.tau) +\
    tf.mul(actor_seperate_params_t[i], 1. - self.tau))
    for i in range(len(actor_seperate_params))]
    reset_critic_seperate_params_t = [critic_seperate_params_t[i].assign(tf.mul(critic_seperate_params[i], self.tau) +\
    tf.mul(critic_seperate_params_t[i], 1. - self.tau))
    for i in range(len(critic_seperate_params))]
    #[-----Cost and gradient operations-----]
    #Actor Network
    action_gradient_input = tf.placeholder(tf.float32, [None, self.a_dim])
    actor_ddpg_gradient = tf.gradients(output_a, [common_params, actor_seperate_params], -action_gradient_input)
    train_actor = tf.train.AdamOptimizer(self.actor_learning_rate).\
      apply_gradients(zip(actor_ddpg_gradient, [common_params, actor_seperate_params]))
    #Critic Netowrk
    #[TODO: Add L2 weight decay for critic]
    target_input = tf.placeholder("float", [None])
    cost = tflearn.mean_square(output_c, target_input)
    optimizer = tf.train.AdamOptimizer(learning_rate = self.critic_learning_rate)
    train_critic = optimizer.minimize(cost, var_list=[common_params, critic_seperate_params])
    action_grads = tf.gradients(output_c, action_input)

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
    action_vectors = self.to_action_input(action_batch)
    state_input = self.graph_ops["s"]
    target_input = self.graph_ops["target_input"]
    action_input = self.graph_ops["action_input"]
    self.sess.run(self.graph_ops['train_net'], feed_dict={state_input: state_batch, action_input:action_vectors, target_input:target_batch})

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







