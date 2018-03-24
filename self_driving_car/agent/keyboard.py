import h5py
import numpy as np
from pyglet.window import key


class KeyboardAgent(object):
    def __init__(self, input_shape, output_shape, data_path='./data.h5', num_sample=10000):
        self._num_sample = num_sample
        self.file = h5py.File(data_path, 'w')

        self.state_dataset = self.file.create_dataset('state',
                                                      (num_sample,) + input_shape,
                                                      np.float32,
                                                      chunks=(1,) + input_shape)

        self.action_dataset = self.file.create_dataset('action',
                                                       (num_sample,) + output_shape,
                                                       np.float32,
                                                       chunks=(1,) + output_shape)

        self.reward_dataset = self.file.create_dataset('reward', (num_sample, 1), np.float32, chunks=(1, 1))

        self.action_array = np.zeros((3,), np.float32)

        self.count = 0

    def act(self, state, reward=0, done=False):
        self.state_dataset[self.count] = state
        self.action_dataset[self.count] = self.action_array
        self.reward_dataset[self.count] = reward

        self.count += 1

        if self.count == self._num_sample:
            self.close()

        return self.action_array

    def close(self):
        self.file.close()
        quit()

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
