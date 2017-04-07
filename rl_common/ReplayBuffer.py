#!/usr/bin/env python

""" 
    Replay Buffer Module for Deep Q Network 
    Author: Mohak Bhardwaj
    Based of off Berkeley Deep RL course's dqn implementation which can be found 
    at https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
    This implementation is optimized as it only keeps one copy of the frame in the buffer,
    hence saving RAM which can blow up.
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    """Base class for SimpleBuffer and PrioritizedBuffer that implements add, size and clear
       methods""" 
       #[TODO: Works only for discrete action spaces]
    def __init__(self, buffer_size, frame_history_length):
        self.buffer_size = buffer_size
        self.frame_history_length = frame_history_length
        # self.buffer = deque()
        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

        self.next_idx = 0
        self.curr_buffer_size = 0

    def add(self, s, a, r, t):
        # experience = (s, a, r, t, s2)
        # if self.count < self.buffer_size:
        #     self.buffer.append(experience)
        #     self.count += 1
        # else:
        #     self.buffer.popleft()
        #     self.buffer.append(experience)
        if self.obs is None:
          self.obs    = np.empty([self.buffer_size] + list(s.shape), dtype=np.uint8) #Change to uint8
          self.action = np.empty([self.buffer_size]                , dtype=np.int32)
          self.reward = np.empty([self.buffer_size]                , dtype=np.float32)
          self.done   = np.empty([self.buffer_size]                , dtype=np.bool)
        self.obs[self.next_idx]    = s
        self.action[self.next_idx] = a
        self.reward[self.next_idx] = r
        self.done[self.next_idx]   = t
        self.next_idx = (self.next_idx+ 1)%self.buffer_size
        self.curr_buffer_size = min(self.buffer_size, self.curr_buffer_size + 1)

    def size(self):
        return self.curr_buffer_size

    def sample_batch(self, batch_size):
      '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it. 
        
        Note that whenever there are missing frames mostly due to insufficient 
        data at the start of the episode, additional frames will be added which are
        all zeros.
        '''    
    def clear(self):
        self.obs = None
        self.action = None
        self.reward = None
        self.done = None
        self.curr_buffer_size = 0
        self.next_idx  = 0

    def can_sample(self, batch_size):
      return batch_size < self.curr_buffer_size

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_length
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.curr_buffer_size != self.buffer_size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.buffer_size]:
                start_idx = idx + 1
        missing_context = self.frame_history_length - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.buffer_size])
            return np.asarray(frames)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            # return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)
            # print self.obs[start_idx:end_idx].shape
            return np.asarray(self.obs[start_idx:end_idx])


class SimpleBuffer(ReplayBuffer):
    """Implements simple experience replay buffer that samples batches uniformly
        from the buffer without any prioritization"""
    def sample_batch(self, batch_size):
      assert self.can_sample(batch_size)
      idxs = random.sample(xrange(0, self.curr_buffer_size - 2), batch_size)
      s_batch  = np.concatenate([self._encode_observation(idx)[None] for idx in idxs], 0)
      a_batch  = np.asarray(self.action[idxs])
      r_batch  = np.asarray(self.reward[idxs])
      t_batch  = np.asarray(self.done[idxs])
      s2_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxs], 0)
      # print s_batch.shape
      return s_batch, a_batch, r_batch, t_batch, s2_batch

class PrioritizedBuffer(ReplayBuffer):
    """Implements prioritized experience replay, where experiences are 
        prioritized based on TD error and stoachastic prioritization 
        with annealing. See https://arxiv.org/pdf/1511.05952v4.pdf for details"""
    # [TODO: Implement]




        


