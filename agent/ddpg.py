# coding: utf-8

import numpy as np
import torch
from torch import cuda
from torch import nn
from torch import optim
from torch.autograd import Variable

from agent.core import Agent
from agent.replay_buffer import ReplayBuffer


# TODO: Init weights.


class DDPGAgent(Agent):
    def __init__(
            self,
            num_outputs,
            actor_net,
            critic_net,
            actor_lr=1e-4,
            critic_lr=1e-3,
            batch_size=16,
            buffer_size=10 ** 6,
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
        self._target_actor = actor_net
        self._target_critic = critic_net

        # TODO:
        if self._load:
            self.load()

        self._target_actor.eval()
        self._target_critic.eval()

        if self._train:
            self._actor.train()
            self._critic.train()
        else:
            self._actor.eval()
            self._critic.eval()

        self._target_critic.load_state_dict(self._critic.state_dict())
        self._target_actor.load_state_dict(self._actor.state_dict())

        self._actor_optimizer = optim.Adam(self._critic.parameters(), actor_lr)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), critic_lr)

        self._critic_criterion = nn.MSELoss().cuda()

        self._replay_buffer = ReplayBuffer(buffer_size)

        self._stored = None

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
                self._replay_buffer.store(self._stored + [state_array])

            # Sample a random mini-batch.
            samples = self._replay_buffer.random_sample(self._batch_size)

            if samples:
                last_states, last_actions, last_rewards, states = samples

                # Gey y.
                y = self._target_actor(states)
                y = last_rewards + self._discount_factor * self._target_critic(states, y)

                # Update critic:
                self._critic_optimizer.zero_grad()
                critic_loss = self._critic_criterion(self._critic(last_states, last_actions), y)
                critic_loss.backward()
                self._critic_optimizer.step()

                # Update actor.
                self._actor.zero_grad()
                actor_loss = -torch.mean(self._critic([states, self._actor(states)]))
                actor_loss.backward()
                self._actor_optimizer.step()

                # Update target critic.
                self._target_critic.load_state_dict(self._critic.state_dict())

                # Update target actor.
                self._target_actor.load_state_dict(self._actor.state_dict())

        # Select action using actor.
        state = Variable(torch.from_numpy(state_array).float, volatile=True).unsqueeze(0)

        action_array = self._actor(state).numpy()

        self._stored = [state_array, action_array, reward]  # TODO: Wrong.

        return action_array

    def close(self):
        raise NotImplementedError

    def save(self):
        torch.save(self._actor.state_dict(), self._weight_folder)

    def load(self):
        self._critic.load_state_dict(torch.load(self._weight_folder))
        self._actor.load_state_dict(torch.load(self._weight_folder))

        self._target_actor = self._actor.copy()
        self._target_critic = self._critic.copy()


if __name__ == '__main__':
    def convert_action(action):
        x = np.array([0, 0, 0])
        x[0] = action[0]
        if action[1] > 0:
            x[1] = action[1]
        else:
            x[2] = -action[1]
        return x


    import gym
    import time
    import numpy as np

    agent = DDPGAgent(3, load=False)
    env = gym.make('CarRacing-v0')
    for i in range(1000):
        ob = env.reset()
        env.render()
        action = agent.act(ob)

        # action = convert_action(action)

        for x in range(10000):
            ob, r, d, _ = env.step(action)
            env.render()
            action = agent.act(ob, r, d)

            # action = convert_action(action)

            if d:
                print()
                print(time.ctime())
                print('Done i:{},x:{} '.format(i, x))
                d = False
                agent.save()
                break
