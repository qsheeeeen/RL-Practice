# coding: utf-8

import copy

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from .core import Agent
from .policy.actor import CNNActor, MLPActor
from .policy.critic import CNNCritic, MLPCritic
from .replay_buffer import ReplayBuffer


class DDPGAgent(Agent):
    def __init__(
            self,
            num_inputs,
            num_outputs,
            actor_lr=1e-4,
            critic_lr=1e-3,
            batch_size=16,
            buffer_size=10 ** 6,
            discount_factor=0.99,
            tau=0.001,
            train=True,
            load=False,
            weight_folder='./weights', ):

        self._num_inputs = num_inputs
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._tau = tau
        self._train = train
        self._load = load
        self._weight_folder = weight_folder

        if num_inputs is None:
            self._actor = CNNActor(num_outputs)
            self._critic = CNNCritic(num_outputs)
        else:
            self._actor = MLPActor(num_inputs, num_outputs)
            self._critic = MLPCritic(num_inputs, num_outputs)

        if self._load:
            self._critic.load_state_dict(torch.load(self._weight_folder))
            self._actor.load_state_dict(torch.load(self._weight_folder))

        if self._train:
            self._actor.train()
            self._critic.train()

            self._target_actor = copy.deepcopy(self._actor)
            self._target_critic = copy.deepcopy(self._critic)

            self._target_actor.eval()
            self._target_critic.eval()

            self._actor_optimizer = Adam(self._critic.parameters(), actor_lr, eps=1e-5)
            self._critic_optimizer = Adam(self._critic.parameters(), critic_lr, eps=1e-5)

            self._critic_criterion = nn.MSELoss().cuda()

            self._replay_buffer = ReplayBuffer(buffer_size)

            self._stored = None

        else:
            self._actor.eval()
            self._critic.eval()

    def act(self, state, reward=0, done=False):

        state = self._processing(state)

        action_var = self._actor(Variable(state, volatile=True))

        if self._train:
            if self._stored:
                self._replay_buffer.store(self._stored + [reward, state])

            samples = self._replay_buffer.random_sample(self._batch_size)

            if samples:
                last_states, last_actions, last_rewards, states = samples

                y = last_rewards + self._discount_factor * self._target_critic(states, self._target_actor(states))

                self._critic_optimizer.zero_grad()
                critic_loss = self._critic_criterion(self._critic(last_states, last_actions), y)
                critic_loss.backward()
                self._critic_optimizer.step()

                self._actor.zero_grad()
                actor_loss = -torch.mean(self._critic([states, self._actor(states)]))
                actor_loss.backward()
                self._actor_optimizer.step()

                self._target_critic.load_state_dict(self._critic.state_dict())

                self._target_actor.load_state_dict(self._actor.state_dict())

            self._stored = [state, action_var.data]

        return action_var.data.cpu().numpy()[0]

    def save(self):
        torch.save(self._actor.state_dict(), self._weight_folder)

    def _processing(self, array):
        try:
            tensor = torch.from_numpy(array).float()
        except RuntimeError:
            import numpy as np

            a = np.zeros_like(array) + array
            tensor = torch.from_numpy(a).float()

        if self._num_inputs is None:
            tensor = tensor.permute(2, 0, 1)
            tensor /= 256.

        return tensor.unsqueeze(0).cuda()


if __name__ == '__main__':
    import gym
    import time
    import numpy as np

    env = gym.make('CarRacing-v0')
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]
    agent = DDPGAgent(None, outputs, load=False)
    for i in range(10000):
        ob = env.reset()
        env.render()
        action = agent.act(ob)
        for x in range(10000):
            ob, r, d, _ = env.step(action)
            env.render()
            action = agent.act(ob, r, d)
            if d:
                print()
                print(time.ctime())
                print('Done i:{},x:{} '.format(i, x))
                d = False
                # agent.save()
                break
