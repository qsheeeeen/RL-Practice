import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from visualdl import LogWriter


class Trainer(object):
    def __init__(self, model, input_shape, output_shape, data_path='./data.h5', num_sample=10000, lr=3e-4):
        self._model = model(output_shape[0])

        self._model_optimizer = Adam(self._model.parameters(), lr=lr, eps=1e-5)
        self._model_criterion = SmoothL1Loss().cuda()

        self._file = h5py.File(data_path, 'r')

        self._state_dataset = self._file.require_dataset('state', (num_sample,) + input_shape, np.float32)
        self._action_dataset = self._file.require_dataset('action', (num_sample,) + output_shape, np.float32)
        self._reward_dataset = self._file.require_dataset('reward', (num_sample, 1), np.float32)

        self._logger = LogWriter('./workspace', sync_cycle=100)

        with self._logger.mode("train"):
            self._scalar_train_loss = self._logger.scalar("/scalar_pytorch_train_loss")

    def fit(self, batch_size=32, epochs=10):
        reward_history = []
        states_array = np.zeros_like(self._state_dataset) + self._state_dataset
        actions_array = np.zeros_like(self._action_dataset) + self._action_dataset

        states = torch.from_numpy(states_array).cuda()
        states = states.permute(0, 3, 1, 2)
        states = states / 128 - 1
        actions = torch.from_numpy(actions_array).cuda()
        # rewards = torch.from_numpy().cuda()

        dataset = TensorDataset(states, actions)

        data_loader = DataLoader(dataset, batch_size, shuffle=True)

        for _ in range(epochs):
            for states, actions in data_loader:
                states_var = Variable(states)
                actions_var = Variable(actions)

                means_var, stds_var, values_var = self._model(states_var)

                total_loss = self._model_criterion(means_var, actions_var)
                reward_history.append(total_loss.data.cpu().numpy()[0])

                self._model_optimizer.zero_grad()
                total_loss.backward()
                self._model_optimizer.step()

        plt.plot(reward_history)

        reward_history = []

        for _ in range(epochs):
            for states, actions in data_loader:
                states_var = Variable(states)
                actions_var = Variable(actions)

                means_var, stds_var, values_var = self._model(states_var)

                total_loss = self._model_criterion(means_var, actions_var)
                reward_history.append(total_loss.data.cpu().numpy()[0])

        name = 'test_loss_history'

        plt.plot(reward_history)
        plt.title(name)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.savefig(name + '.png')

    def save(self, weight_path='./ppo_weights.pth'):
        torch.save(self._model.state_dict(), weight_path)
