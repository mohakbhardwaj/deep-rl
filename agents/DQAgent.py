#!/usr/bin/env python
"""Class implements simple Deep Q learning agent compatible
with openAI gym environment interface"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from RLAgent import RLAgent
from networks.DQN import DQNetwork
from rl_common.ReplayBuffer import SimpleBuffer
from rl_common.NoiseModel import *
import gym
import tensorflow as tf
import numpy as np

class DQAgent(RLAgent):
  def __init__(self,\
               env,\
               discount_factor = 0.99 ,\
               learning_rate = 0.0001,\
               max_training_steps = 10000000,\
               steps_per_epoch = 6000,\
               buffer_size = 40000,\
               start_training_after = 400000 ,\
               batch_size = 32,\
               target_network_update = 10000 ,\
               gradient_update_frequency = 4 ,\
               clip_rewards = True,\
               save_after_episodes = 3,\
               training_params_file = "dqn_atari",\
               training_summary_file = "dqn_atari" ,\
               max_epsilon = 1,\
               min_epsilon = 0.1,\
               min_epsilon_timestep = 400000 ,\
               max_episode_length = 200 ,\
               tau = 1 ,\
               vision = True ,\
               warm_start = False ,\
               network_mode = "cpu",\
               seed = 1234 ,\
               render = False):
      
      #Learning parameters
      self.env = env
      self.discount_factor = discount_factor
      self.learning_rate = learning_rate
      self.max_training_steps = max_training_steps
      self.steps_per_epoch = steps_per_epoch
      self.buffer_size = buffer_size
      self.start_training_after = start_training_after
      self.batch_size = batch_size
      self.target_network_update = target_network_update
      self.gradient_update_frequency = gradient_update_frequency
      self.clip_rewards = clip_rewards
      self.save_after_episodes = save_after_episodes
      self.training_params_file = training_params_file
      self.summary_dir = "../data/summaries/dqn/" + training_summary_file
      self.vision = vision
      self.warm_start = warm_start
      self.max_epsilon = max_epsilon
      self.min_epsilon = min_epsilon
      self.min_epsilon_timestep = min_epsilon_timestep
      self.tau = tau
      self.render_env = render
      self.num_actions = env.action_dim
      self.max_episode_length = max_episode_length
      
      tf.set_random_seed(seed)
      np.random.seed(seed)
      
      self.replay_buffer = SimpleBuffer(self.buffer_size, self.env.history_length)
      self.exploration_strategy = EpsilonGreedy(self.num_actions, self.num_actions, 0,  self.max_epsilon, self.min_epsilon, self.min_epsilon_timestep)
      # self.heldout_states = SimpleBuffer(self.buffer_size)
      self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))# log_device_placement=True))
      self.network = DQNetwork(self.sess ,\
                              self.env.action_dim,\
                              self.env.observation_dim,\
                              self.discount_factor,\
                              self.batch_size, \
                              self.learning_rate,\
                              1,\
                              self.tau,\
                              self.env.history_length,\
                              self.env.h_out,\
                              self.env.w_out,\
                              self.vision ,\
                              network_mode)

  def learn(self):
      timestep = 0
      curr_state = self.env.reset() #Get the initial state
      episode_reward = 0
      avg_reward_per_episode = 0 #Average reward per episode
      num_episodes_passed = 0
      episode_length = 0
      episode_average_max_q = 0
      print "[INFO: Initiate Training]"
      # Load params from a saved model
      if self.warm_start:
        try:
            self.network.load_params(self.training_params_file)
            print "[INFO: Weights Loaded]"
        except:
            print "[ERR: Loading failed!]"
      summary_ops, summary_vars = self.build_summaries()
      writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

      while timestep < self.max_training_steps:
        action = 0
        # print curr_state.shape
        if self.render_env: 
          self.env.render()
        if (timestep+1)%self.steps_per_epoch == 0:
          print '[INFO: Epoch Done | Timestep: %.2i '% int(timestep) , '| Episodes Passed:  %.2i' % int(num_episodes_passed),\
            ' | Average Reward Per Episode: %.2i' % int(avg_reward_per_episode), ' | Current Epsilon: %.4f' % float(self.exploration_strategy.epsilon), ']'
        #Initially execute uniform random policy and populate the experience buffer
        if timestep < self.start_training_after:
          print("Random action")
          action = self.env.sample_action()         
        #Otherwise follow the exploration strategy
        else:
          best_action = self.network.get_best_action(curr_state)
          action = self.exploration_strategy.get_action(timestep,best_action)
        
        next_state, reward, terminal, info = self.env.step(action)  
        episode_reward += reward
        q_value_pred = self.network.evaluate_values(curr_state)
        episode_average_max_q += np.max(q_value_pred)
        #Clip the reward 
        if self.clip_reward:
          reward = self.clip_reward(reward)
        
        #Append the experience to replay buffer
        self.replay_buffer.add(curr_state[-1,:,:], action, reward, terminal)
        #Train the network
        if (self.replay_buffer.size() > self.start_training_after) and ((timestep%self.gradient_update_frequency) == 0):
          #Sample a batch form experience buffer
          #Calculate targets from the batch
          s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.batch_size) 
          #While calculating targets if some state is terminal, then target must be R(s) and not 
          # R(s) + gamma x Q*(s',a')  
          target_batch = r_batch
          for idx, (term,s2,target) in enumerate(zip(t_batch,s2_batch,target_batch)):
              lookahead = 0
              if not term:
                  #Target q values are obtained from target network
                  q_vals = self.network.evaluate_values_target(s2)
                  max_q = np.max(q_vals)
                  lookahead = self.discount_factor * max_q
              target_batch[idx] += lookahead 
          #Send the state batch and target batch to DQN
          self.network.train(s_batch, target_batch, a_batch)
        # Periodically update target network
        if (timestep % self.target_network_update) == 0:  
          self.network.update_target_network_params()
        
        if terminal or (episode_length+1)%self.max_episode_length == 0:
          #Begin a new episode if reached terminal episode
          curr_state = self.env.reset()
          #Update the average reward
          num_episodes_passed += 1
          avg_reward_per_episode += (episode_reward - avg_reward_per_episode)/num_episodes_passed
          episode_length += 1
          #Write tensorflow summaries
          summary_str = self.sess.run(summary_ops, feed_dict={
                  summary_vars[0]: episode_reward,
                  summary_vars[1]: episode_average_max_q / float(episode_length)})
          writer.add_summary(summary_str, num_episodes_passed)
          writer.flush()
          print '[INFO: Timestep: %.2i' % int(timestep),'| Reward: %.2i' % int(episode_reward), ' | Episode: ', num_episodes_passed, \
          ' | Episode Length: ', episode_length, '| Qmax: %.4f' % (episode_average_max_q / float(episode_length)), ']'
          
          if (num_episodes_passed + 1)%self.save_after_episodes == 0:
              print "[INFO: Saving currently learned weights]" 
              self.network.save_params(self.training_params_file)
          #Reset episode statistics               
          episode_reward = 0
          episode_average_max_q = 0
          episode_length = 0
        else:
          curr_state = next_state
          episode_length += 1
        timestep += 1           
      #Save final model weights after traning complete
      print "[INFO: Training Done. Saving final model weights]"
      self.network.save_params(self.training_params_file)

  def test(self, max_episodes):
      timestep = 0
      curr_state = self.env.reset() #Get the initial state
      episode_reward = 0 #Cumulative rewards in one episode
      avg_reward = 0 #Average reward per episode
      episode_length = 0
      curr_episode = 0
      episode_average_max_q = 0
      #Load params from file
      print "[INFO: Loading parameters]"
      try:
        self.network.load_params(self.training_params_file)
        print "[INFO: Weights Loaded]"
      except:
        print "[ERR: Loading failed!]"
      
      while curr_episode < max_episodes:
          if render_env: 
            self.env.render()
          best_action = self.network.get_best_action(curr_state)
          q_value_pred = self.network.evaluate_values(curr_state)
          episode_average_max_q += np.max(q_value_pred)

          #Do best action only without exploration (testing!)
          next_state, reward, terminal, info = self.env.step(best_action)
          episode_reward += reward
          if terminal or (episode_length+1)%self.max_episode_length == 0:
            curr_episode += 1
            curr_state = self.env.reset()
            avg_reward += (episode_reward - avg_reward)/curr_episode
            episode_length += 1
            print '[INFO: | Reward: %.2i' % int(episode_reward), ' | Episode: ', curr_episode, \
            'Episode Length: ', episode_length, '| Qmax: %.4f' % (episode_average_max_q / float(episode_length)),\
            'Average Reward Per Episode: ', avg_reward, ']'
            episode_length = 0
            episode_reward = 0
            episode_average_max_q = 0
          else:
            episode_length += 1
            curr_state = next_state
          timestep += 1

  def clip_reward(self, reward):
    return np.sign(reward)

  def build_summaries(self): 
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars

