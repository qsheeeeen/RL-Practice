import h5py
import numpy as np


class Recoder(object):
    def __init__(self, data_path, state_shape, action_shape):
        self.num_sample = 1e4

        self.file = h5py.File(data_path, 'w')

        self.state_dataset = self.file.create_dataset('states',
                                                      (self.num_sample,) + state_shape,
                                                      np.uint8,
                                                      chunks=(1,) + state_shape,
                                                      maxshape=(None,) + state_shape)

        self.action_dataset = self.file.create_dataset('actions',
                                                       (self.num_sample,) + action_shape,
                                                       np.float32,
                                                       chunks=(1,) + action_shape,
                                                       maxshape=(None,) + action_shape)

        self.reward_dataset = self.file.create_dataset('rewards',
                                                       (self.num_sample, 1),
                                                       np.float32,
                                                       chunks=(1, 1),
                                                       maxshape=(None, 1))

        self.done_dataset = self.file.create_dataset('dones',
                                                     (self.num_sample, 1),
                                                     np.bool,
                                                     chunks=(1, 1),
                                                     maxshape=(None, 1))
        self.count = 0

    def store(self, state, action, reward, done):
        self.state_dataset[self.count] = state
        self.action_dataset[self.count] = action
        self.reward_dataset[self.count] = reward
        self.done_dataset[self.count] = done
        self.count += 1

        if self.count == self.num_sample:
            self.num_sample += 1e4
            self.state_dataset.resize(self.num_sample, 0)
            self.action_dataset.resize(self.num_sample, 0)
            self.reward_dataset.resize(self.num_sample, 0)
            self.done_dataset.resize(self.num_sample, 0)

    def close(self):
        self.state_dataset.resize(self.count, 0)
        self.action_dataset.resize(self.count, 0)
        self.reward_dataset.resize(self.count, 0)
        self.done_dataset.resize(self.count, 0)
