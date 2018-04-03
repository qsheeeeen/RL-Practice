import h5py
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from visualdl import LogWriter

from self_driving_car.util import TensorDataset


class Trainer(object):
    def __init__(self, model, input_shape, output_shape, data_path='./data.hdf5', lr=3e-4):
        self.model = model(input_shape, output_shape)

        self.model_optimizer = Adam(self.model.parameters(), lr=lr)
        self.model_criterion = SmoothL1Loss().cuda()

        self.file = h5py.File(data_path, 'r')

        self.logger = LogWriter('./logdir', sync_cycle=100)

        with self.logger.mode('train'):
            self.train_loss = self.logger.scalar('loss/')

        with self.logger.mode('test'):
            self.test_loss = self.logger.scalar('loss/')

    def get_dataset(self, test_split):
        state_dataset = self.file['states']
        action_dataset = self.file['actions']
        reward_dataset = self.file['rewards']

        states_array = np.array(state_dataset)
        actions_array = np.array(action_dataset)
        reward_array = np.array(reward_dataset)

        states_t = torch.from_numpy(states_array)
        actions_t = torch.from_numpy(actions_array)
        reward_t = torch.from_numpy(reward_array)

        # TODO: precessing image.
        # states_t = states_t.per

        dataset = TensorDataset(states_t, actions_t, reward_t)

        test_length = int(test_split * len(states_t))
        train_length = len(states_t) - test_length

        training_dataset, testing_dataset = random_split(dataset, [train_length, test_length])

        return training_dataset, testing_dataset

    def fit(self, batch_size=32, epochs=100, test_split=0.2):
        training_dataset, testing_dataset = self.get_dataset(test_split)

        training_loader = DataLoader(training_dataset, batch_size, shuffle=True)
        testing_loader = DataLoader(testing_dataset, batch_size, shuffle=True)

        training_step = 0
        testing_step = 0

        for _ in range(epochs):
            for states_t, actions_t, rewards_t in training_loader:
                states_v = Variable(states_t)
                actions_v = Variable(actions_t)
                rewards_v = Variable(rewards_t)

                means_v, stds_v, values_v = self.model(states_v)

                loss1 = self.model_criterion(means_v, actions_v)
                loss2 = self.model_criterion(values_v, rewards_v)

                total_loss = loss1 + loss2

                self.model_optimizer.zero_grad()
                total_loss.backward()
                self.model_optimizer.step()

                self.train_loss.add_record(training_step, float(total_loss))
                training_step += 1

            for states, actions in testing_loader:
                states_v = Variable(states)
                actions_v = Variable(actions)

                means_v, stds_v, values_v = self.model(states_v)

                total_loss = self.model_criterion(means_v, actions_v)

                self.test_loss.add_record(testing_step, float(total_loss))
                testing_step += 1

    def save(self, weight_path='./ppo_weights.pth'):
        torch.save(self.model.state_dict(), weight_path)
