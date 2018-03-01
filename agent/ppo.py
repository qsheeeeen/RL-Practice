# coding: utf-8

import torch
from torch import optim
from torch.autograd import Variable
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader

from agent.core import Agent
from agent.replay_buffer import ReplayBuffer
from net import SharedNetwork


# from visualdl import LogWriter


class PPOAgent(Agent):
    def __init__(
            self,
            horizon=2048,
            learning_rate=3e-4,
            num_epoch=10,
            batch_size=64,
            discount_factor=0.99,
            gae_parameter=0.95,
            clip_range=0.2,
            train=True,
            load=False,
            weight_path='./weights'):
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.gae_parameter = gae_parameter
        self.clip_range = clip_range
        self.train = train
        self.load = load
        self.weight_path = weight_path

        self.policy_old = SharedNetwork()
        self.policy_old.eval()

        if self.load:
            print('Load weights.')
            self.policy_old.load_state_dict(torch.load(self.weight_path))

        if self.train:
            self.policy = SharedNetwork()
            self.policy.load_state_dict(self.policy_old.state_dict())
            self.policy.train()

            self.policy_optimizer = optim.Adam(self.policy.parameters(), learning_rate)

            self.replay_buffer = ReplayBuffer(horizon)

        self.stored = None

    def act(self, state_array, reward=0., done=False):
        state = self._preprocessing(state_array)
        reward = torch.FloatTensor([reward])

        if self.train:
            if self.stored:
                self.replay_buffer.store((self.stored + (reward,)))

                if (len(self.replay_buffer) == self.horizon) or done:
                    self._finish_iteration()

        mean_var, std_var, value_var = self.policy_old(Variable(state.unsqueeze(0).cuda(), volatile=True))

        m = Normal(mean_var, std_var)
        action_var = m.sample()

        if self.train:
            self.stored = (state,
                           value_var.data.cpu()[0],
                           m.log_prob(action_var).data.cpu()[0])

        return action_var.data.cpu().numpy()[0]

    def close(self):
        raise NotImplementedError

    def save(self):
        torch.save(self.policy_old.state_dict(), self.weight_path)

    @staticmethod
    def _preprocessing(array):
        import numpy as np
        a = np.zeros_like(array)
        a += array
        array = torch.from_numpy(a).float()
        array = array.permute(2, 0, 1)
        array /= 256.

        return array

    def _calculate_advantage(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        advantages[-1] = rewards[-1] - values[-1]

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.discount_factor * values[t + 1] - values[t]

            advantages[t] = delta + self.discount_factor * self.gae_parameter * advantages[t + 1]

        return advantages

    def _finish_iteration(self):
        samples = self.replay_buffer.get_all()

        states, values_old, log_probs_old, rewards = samples

        advantages = self._calculate_advantage(rewards, values_old)

        values_target = advantages + values_old

        advantages = (advantages - advantages.mean()) / advantages.std()

        data_set_1 = TensorDataset(states, log_probs_old)
        data_set_2 = TensorDataset(advantages, values_target)

        data_loader_1 = DataLoader(data_set_1, self.batch_size, shuffle=True)
        data_loader_2 = DataLoader(data_set_2, self.batch_size, shuffle=True)

        # Update policy network.
        for _ in range(self.num_epoch):
            for (states, log_probs_old), (advantages, values_target) in zip(data_loader_1, data_loader_2):
                states_var = Variable(states.cuda())
                log_probs_old_var = Variable(log_probs_old.cuda())
                advantages_var = Variable(advantages.cuda())
                values_target_var = Variable(values_target.cuda())

                self.policy_optimizer.zero_grad()

                means_var, stds_var, values_var = self.policy(states_var)
                m = Normal(means_var, stds_var)
                actions_var = m.sample()
                log_probs_var = m.log_prob(actions_var)

                ratio = torch.exp(log_probs_var - log_probs_old_var)

                surrogate_1 = ratio * advantages_var
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_var
                pessimistic_surrogate = -torch.mean(torch.min(surrogate_1, surrogate_2))

                value_loss = torch.mean(torch.pow((values_var - values_target_var), 2))

                total_loss = pessimistic_surrogate + value_loss

                total_loss.backward()
                self.policy_optimizer.step()

        # Update old policy net.
        self.policy_old.load_state_dict(self.policy.state_dict())
        print('Finished update.')


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

    agent = PPOAgent(load=False)
    env = gym.make('CarRacing-v0')
    for i in range(1000):
        print('Begin.')
        ob = env.reset()
        env.render()
        action = agent.act(ob)

        action = convert_action(action)

        for x in range(10000):
            ob, r, d, _ = env.step(action)
            env.render()
            action = agent.act(ob, r, d)

            action = convert_action(action)

            if d:
                print('Done i:{},x:{} '.format(i, x))
                print(time.ctime())
                print()
                d = False
                agent.save()
                break
