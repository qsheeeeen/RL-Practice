# coding: utf-8

import h5py
import numpy as np
from pyglet.window import key

from agent.core import Agent


class KeyboardAgent(Agent):
    def __init__(self):
        self.file = h5py.File('./data.h5', 'w')

        self.state_data_set = self.file.create_dataset('state', (10000, 96, 96, 3), np.float32, chunks=(1, 96, 96, 3))
        self.action_data_set = self.file.create_dataset('action', (10000, 2), np.float32, chunks=(1, 2))

        self.action_array = np.zeros((3,), np.float32)

        self.count = 0

    def act(self, state_array, reward=0, done=False):
        self.action_array = np.zeros((3,), np.float32)

        self.state_data_set[self.count] = state_array
        self.action_data_set[self.count] = self.action_array

        self.count += 1

        return self.action_array

    def close(self):
        self.file.close()

    def load(self):
        raise NotImplementedError

    def key_press(self, k, mod):
        if k == key.LEFT:
            self.action_array[0] = -1.0
        if k == key.RIGHT:
            self.action_array[0] = +1.0
        if k == key.UP:
            self.action_array[1] = +1.0
        if k == key.DOWN:
            self.action_array[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(self, k, mod):
        if k == key.LEFT and self.action_array[0] == -1.0:
            self.action_array[0] = 0
        if k == key.RIGHT and self.action_array[0] == +1.0:
            self.action_array[0] = 0
        if k == key.UP:
            self.action_array[1] = 0
        if k == key.DOWN:
            self.action_array[2] = 0


if __name__ == '__main__':
    import gym

    agent = KeyboardAgent()

    env = gym.make('CarRacing-v0')
    env.render()
    env.unwrapped.viewer.window.on_key_press = agent.key_press
    env.unwrapped.viewer.window.on_key_release = agent.key_release

    ob = env.reset()
    env.render()
    action = agent.act(ob)

    for x in range(100000):
        ob, r, d, _ = env.step(action)
        env.render()
        action = agent.act(ob, r, d)

        if d:
            print('Done x:{} '.format(x))
            print()
            d = False
            agent.close()
            break
