import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from visualdl import LogWriter


class Trainer(object):
    def __init__(self, model, input_shape, output_shape, data_path='./data.hdf5', lr=3e-4):
        self._model = model(input_shape, output_shape)

        self._model_optimizer = Adam(self._model.parameters(), lr=lr)
        self._model_criterion = SmoothL1Loss().cuda()

        self._file = h5py.File(data_path, 'r')

        self._state_dataset = self._file['state']
        self._action_dataset = self._file['action']
        self._reward_dataset = self._file['reward']

        self._logger = LogWriter('./logdir', sync_cycle=100)

        with self._logger.mode('train'):
            self._train_loss = self._logger.scalar('loss/')

        with self._logger.mode('test'):
            self._test_loss = self._logger.scalar('loss/')

    def fit(self, batch_size=32, epochs=100):
        states_array = np.array(self._state_dataset)
        actions_array = np.array(self._action_dataset)

        (states_array_train,
         states_array_test,
         actions_array_train,
         actions_array_test) = train_test_split(states_array, actions_array)

        del states_array, actions_array

        states_train = self._processing_state(states_array_train)
        actions_train = torch.from_numpy(actions_array_train).cuda()

        training_dataset = TensorDataset(states_train, actions_train)
        training_data_loader = DataLoader(training_dataset, batch_size, shuffle=True)

        states_test = self._processing_state(states_array_test)
        actions_test = torch.from_numpy(actions_array_test).cuda()

        testing_dataset = TensorDataset(states_test, actions_test)
        testing_data_loader = DataLoader(testing_dataset, batch_size, shuffle=True)

        training_step = 0
        testing_step = 0

        for _ in range(epochs):
            for states, actions in training_data_loader:
                states_var = Variable(states)
                actions_var = Variable(actions)

                means_var, stds_var, values_var = self._model(states_var)

                total_loss = self._model_criterion(means_var, actions_var)

                self._model_optimizer.zero_grad()
                total_loss.backward()
                self._model_optimizer.step()

                self._train_loss.add_record(training_step, float(total_loss))
                training_step += 1

            for states, actions in testing_data_loader:
                states_var = Variable(states)
                actions_var = Variable(actions)

                means_var, stds_var, values_var = self._model(states_var)

                total_loss = self._model_criterion(means_var, actions_var)

                self._test_loss.add_record(testing_step, float(total_loss))
                testing_step += 1

    def save(self, weight_path='./ppo_weights.pth'):
        torch.save(self._model.state_dict(), weight_path)

    @staticmethod
    def _processing_state(x):
        x = torch.from_numpy(x).cuda()
        x = x.permute(0, 3, 1, 2)
        x = x / 128 - 1
        return x
