# coding: utf-8
import copy

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader

from agent.core import Agent
from agent.replay_buffer import ReplayBuffer
from policy.shared import MLPPolicy, CNNPolicy


class PPOAgent(Agent):
    def __init__(
            self,
            num_inputs=None,
            num_outputs=3,
            horizon=2048,
            lr=3e-4,
            num_epoch=10,
            batch_size=64,
            discount_factor=0.99,
            gae_parameter=0.95,
            clip_range=0.2,
            train=True,
            load=False,
            weight_path='./ppo_weights.pth'):

        self.num_inputs = num_inputs
        self.horizon = horizon
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.gae_parameter = gae_parameter
        self.clip_range = clip_range
        self.train = train
        self.load = load
        self.weight_path = weight_path

        if num_inputs is not None:
            self.policy_old = MLPPolicy(num_inputs, num_outputs)
        else:
            self.policy_old = CNNPolicy(num_outputs)

        self.policy_old.eval()

        if self.load:
            print('Load weights.')
            self.policy_old.load_state_dict(torch.load(self.weight_path))

        if self.train:
            self.policy = copy.deepcopy(self.policy_old)
            self.policy.train()

            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr)

            self.replay_buffer = ReplayBuffer(horizon)

            self.stored = None

    def act(self, state_array, reward=0., done=False):
        state = self._preprocessing(state_array)

        mean_var, std_var, value_var = self.policy_old(Variable(state, volatile=True))

        if self.train:
            m = Normal(mean_var, std_var)
            action_var = m.sample()

            value = value_var.data

            reward = torch.zeros_like(value) + reward

            if self.stored:
                self.replay_buffer.store(self.stored + [reward])

            if len(self.replay_buffer) == self.horizon:
                self._finish_iteration()

            self.stored = [state, value, action_var.data]

        else:
            action_var = mean_var

        return action_var.data.cpu().numpy()[0]

    def close(self):
        raise NotImplementedError

    def save(self):
        torch.save(self.policy_old.state_dict(), self.weight_path)

    def _preprocessing(self, array):
        try:
            tensor = torch.from_numpy(array).float()
        except RuntimeError:
            import numpy as np

            a = np.zeros_like(array)
            a += array
            tensor = torch.from_numpy(a).float()

        if self.num_inputs:
            pass
        else:
            tensor = tensor.permute(2, 0, 1)
            tensor /= 256.

        return tensor.unsqueeze(0).cuda()

    def _calculate_advantage(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        advantages[-1] = rewards[-1] - values[-1]

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.discount_factor * values[t + 1] - values[t]
            advantages[t] = delta + self.discount_factor * self.gae_parameter * advantages[t + 1]

        return advantages

    def _finish_iteration(self):
        samples = self.replay_buffer.get_all()

        states, values_old, actions_old, rewards = samples

        advantages = self._calculate_advantage(rewards, values_old)

        values_target = advantages + values_old

        advantages = (advantages - advantages.mean()) / advantages.std()

        dataset_1 = TensorDataset(states, actions_old)
        dataset_2 = TensorDataset(advantages, values_target)

        data_loader_1 = DataLoader(dataset_1, self.batch_size)
        data_loader_2 = DataLoader(dataset_2, self.batch_size)

        # Update policy network.
        for _ in range(self.num_epoch):
            for (states, actions_old), (advantages, values_target) in zip(data_loader_1, data_loader_2):
                states_var = Variable(states)
                actions_old_var = Variable(actions_old)
                advantages_var = Variable(advantages)
                values_target_var = Variable(values_target)

                means_old_var, stds_old_var, values_old_var = self.policy_old(states_var)
                m_old = Normal(means_old_var, stds_old_var)
                log_probs_old_var = m_old.log_prob(actions_old_var)

                means_var, stds_var, values_var = self.policy(states_var)
                m = Normal(means_var, stds_var)
                log_probs_var = m.log_prob(actions_old_var)

                ratio = torch.exp(log_probs_var - log_probs_old_var)

                surrogate_1 = ratio * advantages_var
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_var
                pessimistic_surrogate = -torch.mean(torch.min(surrogate_1, surrogate_2))

                value_loss = torch.mean(torch.pow((values_var - values_target_var), 2))

                total_loss = pessimistic_surrogate + value_loss

                self.policy_optimizer.zero_grad()
                total_loss.backward()
                self.policy_optimizer.step()

        # Update old policy net.
        self.policy_old.load_state_dict(self.policy.state_dict())


if __name__ == '__main__':
    import gym
    import time
    import numpy as np


    def fun(x):
        return np.minimum(np.maximum(x, -1), 1)


    env = gym.make('CarRacing-v0')
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]
    agent = PPOAgent(None, outputs, load=True)
    for i in range(10000):
        ob = env.reset()
        env.render()
        action = agent.act(ob)
        # action = fun(action)
        for x in range(10000):
            ob, r, d, _ = env.step(action)
            env.render()
            action = agent.act(ob, r, d)
            # action = fun(action)
            if d:
                print()
                print(time.ctime())
                print('Done i:{},x:{} '.format(i, x))
                d = False
                agent.save()
                break
