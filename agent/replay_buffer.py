# coding: utf-8

import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        if len(self.buffer) > batch_size:
            samples = random.sample(self.buffer, batch_size)

        else:
            samples = random.sample(self.buffer, len(self.buffer))

        last_state_batch = [sample[0] for sample in samples]
        last_action_batch = [sample[1] for sample in samples]
        last_reward_batch = [sample[2] for sample in samples]
        state_batch = [sample[4] for sample in samples]

        return last_state_batch, last_action_batch, last_reward_batch, state_batch

    def store(self, state, action, reward, new_state):
        self.buffer.append((state, action, reward, new_state))

    def erase(self):
        self.buffer.clear()
