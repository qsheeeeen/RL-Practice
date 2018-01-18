# coding: utf-8

import numpy as np
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.autograd import Variable

from .core import Agent
from .replay_buffer import ReplayBuffer


# TODO:
def init_weights(m):
    print(m)
    m.weight.data.fill_(1.0)
    print(m.weight)


class DDPGAgent(Agent):
    def __init__(self,
                 actor_net,
                 critic_net,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 batch_size=16,
                 buffer_size=10**6,
                 discount_factor=0.99,
                 tau=0.001,
                 train=True,
                 load=False,
                 weight_folder='./weights', ):

        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._tau = tau
        self._train = train
        self._load = load
        self._weight_folder = weight_folder

        self._actor = actor_net
        self._critic = critic_net

        self._actor.float()
        self._critic.float()

        if cuda.is_available():
            print('Using GPU.')
            self._actor.cuda()
            self._critic.cuda()

        else:
            print('Using CPU.')
            self._actor.cpu()
            self._critic.cpu()

        # TODO:
        if self._load:
            self.load()

        self._target_actor = self._actor.copy()
        self._target_critic = self._critic.copy()

        self._target_actor.eval()
        self._target_critic.eval()

        if self._train:
            self._actor.train()
            self._critic.train()
        else:
            self._actor.eval()
            self._critic.eval()

        self._actor_optimizer = optim.Adam(self._critic.parameters(), lr=actor_lr)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=critic_lr)

        if cuda.is_available():
            self._critic_criterion = nn.MSELoss().cuda()
        else:
            self._critic_criterion = nn.MSELoss().cuda()

        self._replay_buffer = ReplayBuffer(buffer_size)

        self._last_state_array = None
        self._last_action_array = None
        self._last_reward_array = None

    def act(self, state_array, reward, done):
        """
        TODO:
            Handle 'done'.
            Parameter level noise.

        Args:
            state_array (np.ndarray):
            reward (float):
            done (bool):

        Returns:
            np.ndarray:

        """

        if self._train:
            # Store transition.
            if self._last_state_array and self._last_action_array and self._last_reward_array:
                self._replay_buffer.store(self._last_state_array,
                                          self._last_action_array,
                                          self._last_reward_array,
                                          state_array)

            # Sample a random mini-batch.
            samples = self._replay_buffer.sample(self._batch_size)

            if samples:
                (last_state_batch_array, last_action_batch_array, last_reward_batch_array, state_batch_array) = samples

                # Gey y.
                y = self._target_actor(state_batch_array)
                y = last_reward_batch_array + self._discount_factor * self._target_critic(state_batch_array, y)

                # Update critic:
                self._critic_optimizer.zero_grad()

                critic_loss = self._critic_criterion(self._critic(last_state_batch_array, last_action_batch_array), y)
                critic_loss.backward()

                self._critic_optimizer.step()

                # Update actor.
                self._actor.zero_grad()
                actor_loss = (-self._critic([state_batch_array, self._actor(state_batch_array)])).mean()
                actor_loss.backward()
                self._actor_optimizer.step()

                # Update target critic.
                for f_t, f in self._target_critic.parameters(), self._critic.parameters():
                    f_t.data = self._tau * f.data + (1 - self._tau) * f_t.data

                # Update target actor.
                for f_t, f in self._target_actor.parameters(), self._actor.parameters():
                    f_t.data = self._tau * f.data + (1 - self._tau) * f_t.data

            # Select action using actor.
            state = Variable(torch.from_numpy(state_array).float, requires_grad=False).unsqueeze(0)

        else:
            state = Variable(torch.from_numpy(state_array).float, volatile=True).unsqueeze(0)

        action_array = self._actor(state).numpy()

        # Execute Action.
        self._last_state_array = state_array
        self._last_action_array = action_array
        self._last_reward_array = reward

        return action_array

    def save(self):
        torch.save(self._actor.state_dict(), self._weight_folder)

    def load(self):
        self._critic.load_state_dict(torch.load(self._weight_folder))
        self._actor.load_state_dict(torch.load(self._weight_folder))

        self._target_actor = self._actor.copy()
        self._target_critic = self._critic.copy()
