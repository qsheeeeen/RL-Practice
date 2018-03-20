# coding: utf-8

import h5py
import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

class Trainer(object):
    def __init__(self, model, data_path, lr=3e-4):
        self.file = h5py.File('./data.h5', 'w')

        self.model = model()

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr, eps=1e-5)

        self.state_dataset = self.file.create_dataset('state', (10000, 96, 96, 3), np.float32, chunks=(1, 96, 96, 3))
        self.action_dataset = self.file.create_dataset('action', (10000, 3), np.float32, chunks=(1, 3))
        self.reward_dataset = self.file.create_dataset('reward', (10000, 1), np.float32, chunks=(1, 3))

    def fit(self,batch_size=32, epochs=10):
        dataset_1 = TensorDataset(states, actions_old)
        dataset_2 = TensorDataset(advantages, values_target)
        dataset_3 = TensorDataset(log_probs_old, values_target)

        data_loader_1 = DataLoader(dataset_1, batch_size)
        data_loader_2 = DataLoader(dataset_2, batch_size)

        for _ in range(epochs):
            for ((states, actions_old),
                 (advantages, values_target),
                 (log_probs_old, _)) in zip(data_loader_1,
                                            data_loader_2,
                                            data_loader_3):

                states_var = Variable(states)
                actions_old_var = Variable(actions_old)
                reward_old_var = Variable()

                means_var, stds_var, values_var = self._policy(states_var)
                total_loss = pessimistic_surrogate + value_loss

                self._policy_optimizer.zero_grad()
                total_loss.backward()
                self._policy_optimizer.step()

    def save
