# coding: utf-8

import torch
from torch import optim
from torch.autograd import Variable

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
            use_cuda=True,
            weight_folder='./weights'):
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.gae_parameter = gae_parameter
        self.clip_range = clip_range
        self.train = train
        self.load = load
        self.use_cuda = use_cuda

        self.policy_net_old = SharedNetwork(use_cuda=use_cuda)
        self.policy_net_old.eval()

        if self.train:
            self.policy_net = SharedNetwork(use_cuda=use_cuda)
            self.policy_net.load_state_dict(self.policy_net_old.state_dict())
            self.policy_net.train()

            self.policy_net_optimizer = optim.Adam(self.policy_net.prameters(), learning_rate)

            self.replay_buffer = ReplayBuffer(horizon)

        self.last_state_tensor = None
        self.last_action_tensor = None
        self.last_value_tensor = None
        self.last_log_prob = None

    def act(self, *inputs):
        if len(inputs) == 1:
            state_array = inputs[0]
            reward = 0
            done = False
        elif len(inputs) == 3:
            state_array, reward, done = inputs
        else:
            raise NotImplementedError

        state_tensor = torch.from_numpy(state_array).float()
        state_tensor = state_tensor.permute(2, 0, 1)
        state_tensor /= 256.
        reward_tensor = torch.from_numpy(reward)

        if self.train:
            if self.last_state_tensor and self.last_action_tensor:
                self.replay_buffer.store(
                    self.last_state_tensor,
                    self.last_action_tensor,
                    self.last_value_tensor,
                    self.last_log_prob,
                    reward_tensor)

            if (len(self.replay_buffer) == self.horizon) or done:
                self._finish_iteration()

        action_variable, value_variable = self.policy_net_old(Variable(state_tensor, volatile=True).unsqueeze(0))

        self.last_state_tensor = state_tensor
        self.last_action_tensor = action_variable
        self.last_value_tensor = value_variable

        return action_variable.data.numpy()[0]

    def close(self):
        raise NotImplementedError

    def _finish_iteration(self):
        samples = self.replay_buffer.pop(self.batch_size)

        states_tensor, actions_tensor, values_tensor, rewards_tensor = samples

        if self.use_cuda:
            states_tensor.cuda()
            actions_tensor.cuda()
            rewards_tensor.cuda()
            values_tensor.cuda()

        # Compute advantage estimates.
        advantages = torch.FloatTensor(len(self.replay_buffer) - 1)

        for t in range(len(self.replay_buffer) - 1):
            delta = rewards_tensor[t] + self.discount_factor * values_tensor[t + 1] - values_tensor[t]
            advantages[t] = last_gae_lam = delta + self.discount_factor * self.gae_parameter * last_gae_lam

        advantages = (advantages - advantages.mean()) / advantages.std()

        states_variable = Variable(states_tensor)
        actions_variable = Variable(actions_tensor)
        rewards_variable = Variable(rewards_tensor)
        values_variable = Variable(values_tensor)

        for _ in range(self.num_epoch):
            for _ in range(int(len(self.replay_buffer) / self.batch_size)):

                # Update actor network.
                for _ in range(self.num_epoch):
                    self.policy_net_optimizer.zero_grad()

                    ratio = torch.exp(
                        self.policy_net.log_prob(actions_variable) - self.policy_net_old.log_prob(actions_variable))

                    surrogate_1 = ratio * advantages
                    surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                    pessimistic_surrogate = -torch.mean(torch.min(surrogate_1, surrogate_2))

                    value_loss = torch.mean(torch.pow((values_variable -), 2))

                    total_loss = pessimistic_surrogate + value_loss

                    total_loss.backward()
                    self.policy_net_optimizer.step()

                # Update old actor net.
                self.policy_net_old.load_state_dict(self.policy_net.state_dict())
