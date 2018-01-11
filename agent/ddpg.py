# coding: utf-8
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional

from .core import Agent
from .replay_buffer import ReplayBuffer


class ActorNetwork(torch.nn.Module):
    """
    TODO: uniform distribution
    """

    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))
        x = functional.relu(self.fc1(x))
        x = functional.tanh(self.fc2(x))

        return x


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        assert len(x) == 1, 'Incorrect input'
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))


class Critic(object):
    def __init__(self):
        self.model = CriticNetwork()

        self.loss = nn.functional.mse_loss()

        if torch.cuda.is_available():
            self.model.cuda()


class DDPG(Agent):
    """
    TODO: weights decay
    """

    def __init__(self,
                 batch_size=16,
                 buffer_size=1e6,
                 discounting_factor=0.99,
                 learning_factor=0.001,
                 train=True, ):

        self._actor = ActorNetwork()
        self._target_actor = self._actor.copy()

        self._actor_optimizer = optim.Adam(self._critic.parameters(), lr=1e-4)
        self._actor_criterion = nn.MSELoss()

        self._critic = CriticNetwork()
        self._target_critic = self._critic.copy()

        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=1e-3)
        self._critic_criterion = nn.MSELoss()

        self._batch_size = batch_size
        self._discounting_factor = discounting_factor
        self._train = train
        self._learning_factor = learning_factor

        if torch.cuda.is_available():
            # TODO
            pass

        self._replay_buffer = ReplayBuffer(buffer_size)

        self._last_state = None
        self._last_action = None
        self._last_reward = None

    def act(self, state, reward, done):
        """

        Args:
            state (np.ndarray):
            reward (float):
            done (bool):

        Returns:
            np.ndarray

        """

        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        if self._train and self._last_state and self._last_action and self._last_reward:
            # Store transition.
            self._replay_buffer.store(self._last_state, self._last_action, self._last_reward, state)

            # Sample a random minibatch.
            (last_state_batch,
             last_action_batch,
             last_reward_batch,
             state_batch) = self._replay_buffer.sample(self._batch_size)

            last_state_batch = Variable(torch.from_numpy(last_state_batch))
            last_action_batch = Variable(torch.from_numpy(last_action_batch))
            last_reward_batch = Variable(torch.from_numpy(last_reward_batch))
            state_batch = Variable(torch.from_numpy(state_batch))

            # Gey y.
            y = self._target_actor(state_batch)
            y = last_reward_batch + self._discounting_factor * self._target_critic(state_batch, y)

            # Update critic:
            self._critic_optimizer.zero_grad()

            loss = self._critic_criterion(self._critic(last_state_batch, last_action_batch), y)
            loss.backward()

            self._critic_optimizer.step()

            # Update actor.
            self._actor.zero_grad()
            loss = -self._critic([state_batch, self._actor(state_batch)])
            loss = loss.mean()
            loss.backward()
            self._actor_optimizer.step()

            # Update target critic.
            for f_t, f in self._target_critic.parameters(), self._critic.parameters():
                f_t.data = self._learning_factor * f.data + (1 - self._learning_factor) * f_t.data

            # Update target actor.
            for f_t, f in self._target_actor.parameters(), self._actor.parameters():
                f_t.data = self._learning_factor * f.data + (1 - self._learning_factor) * f_t.data

        # Select action using actor.
        action = self._actor(state)

        # Execute Action.
        self._last_state = state
        self._last_action = action
        self._last_reward = reward

        return action.numpy()

    def close(self):
        pass
