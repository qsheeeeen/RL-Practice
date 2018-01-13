# coding: utf-8

from collections import deque
from random import sample as random_sample

from numpy import array, float32


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        if len(self.buffer) > batch_size:
            samples = random_sample.sample(self.buffer, batch_size)

            last_state_batch_array = array([sample[0] for sample in samples], dtype=float32)
            last_action_batch_array = array([sample[1] for sample in samples], dtype=float32)
            last_reward_batch_array = array([sample[2] for sample in samples], dtype=float32)
            state_batch_array = array([sample[4] for sample in samples], dtype=float32)

            return last_state_batch_array, last_action_batch_array, last_reward_batch_array, state_batch_array

        else:
            return None

    def store(self, state, action, reward, new_state):
        self.buffer.append((state, action, reward, new_state))

    def erase(self):
        self.buffer.clear()
