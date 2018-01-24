# coding: utf-8

from collections import deque
import random

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        """

        Args:
            buffer_size (int):
        """
        self.buffer = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        """

        Args:
            batch_size (int):

        Returns:
            tuple:


        """
        if len(self.buffer) >= batch_size:
            samples = random.sample(self.buffer, batch_size)

            last_state_batch_array = np.array([sample[0] for sample in samples], dtype=np.float32)
            last_action_batch_array = np.array([sample[1] for sample in samples], dtype=np.float32)
            last_reward_batch_array = np.array([sample[2] for sample in samples], dtype=np.float32)
            state_batch_array = np.array([sample[4] for sample in samples], dtype=np.float32)

            return last_state_batch_array, last_action_batch_array, last_reward_batch_array, state_batch_array

        else:
            return None

    def store(self, state, action, reward, new_state):
        """
        TODO:
            Change input to *items. Make input suitable to 3 or 4 items.

        Args:
            state (np.ndarray):
            action (np.ndarray):
            reward (float):
            new_state (np.ndarray):

        Returns:
            None

        """
        self.buffer.append((state, action, reward, new_state))

    def clear(self):
        self.buffer.clear()
