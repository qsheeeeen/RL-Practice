import copy

import torch
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam
from torch.utils.data import DataLoader

from .core import Agent
from ..util import ReplayBuffer
from ..util.common import TensorDataset, preprocessing_state


class PPOAgent(Agent):
    def __init__(self, policy, train=True, **kwargs):
        if not len(kwargs):
            kwargs = {
                'abs_utput_limit': 1,
                'horizon': 128,
                'lr': 2.5e-4,
                'num_epoch': 3,
                'batch_size': 32,
                'clip_range': 0.1,
                'vf_coeff': 1,
                'discount_factor': 0.99,
                'gae_parameter': 0.95,
                'max_grad_norm': 0.5}

            # kwargs = {
            #     'abs_utput_limit': 1,
            #     'horizon': 2048,
            #     'lr': 3e-4,
            #     'num_epoch': 10,
            #     'batch_size': 32,
            #     'clip_range': 0.2,
            #     'vf_coeff': 1,
            #     'discount_factor': 0.99,
            #     'gae_parameter': 0.95,
            #     'max_grad_norm': 0.5}

        self.abs_utput_limit = kwargs['abs_utput_limit']
        self.horizon = kwargs['horizon']
        self.lr = kwargs['lr']
        self.num_epoch = kwargs['num_epoch']
        self.batch_size = kwargs['batch_size']
        self.clip_range = kwargs['clip_range']
        self.vf_coeff = kwargs['vf_coeff']
        self.discount_factor = kwargs['discount_factor']
        self.gae_parameter = kwargs['gae_parameter']
        self.max_grad_norm = kwargs['max_grad_norm']

        self.policy_old = policy
        self.policy_old.eval()
        self.train = train

        if self.train:
            self.policy = copy.deepcopy(self.policy_old)
            self.policy.train()

            self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

            self.replay_buffer = ReplayBuffer(self.horizon)

            self.stored = None

        torch.backends.cudnn.benchmark = True

    def act(self, state, reward=0., done=False):
        state_t = preprocessing_state(state).cuda()

        mean_v, std_v, value_v = self.policy_old(Variable(state_t, volatile=True))

        if self.train:
            m_v = Normal(mean_v, std_v)
            action_v = m_v.sample()

            action_t = action_v.data

            value_t = value_v.data
            reward_t = torch.zeros_like(value_t) + reward

            if self.stored is not None:
                self.replay_buffer.store(self.stored + [reward_t])

            if self.replay_buffer.full():
                self.update()
                self.replay_buffer.clear()

            self.stored = [state_t, value_t, action_t, m_v.log_prob(action_v).data]

        else:
            action_t = mean_v.data

        return torch.clamp(action_t, -self.abs_utput_limit, self.abs_utput_limit).cpu().numpy()[0]

    def _calculate_advantage(self, rewards_t, values):
        advantages_t = torch.zeros_like(rewards_t)
        advantages_t[-1] = rewards_t[-1] - values[-1]

        for t in reversed(range(len(rewards_t) - 1)):
            delta = rewards_t[t] + self.discount_factor * values[t + 1] - values[t]
            advantages_t[t] = delta + self.discount_factor * self.gae_parameter * advantages_t[t + 1]

        return advantages_t

    def loss(self, log_probs_v, log_probs_old_v, advantages_v, values_v, values_old_v, values_target_v):
        ratio = torch.exp(log_probs_v - log_probs_old_v)

        pg_losses1 = advantages_v * ratio
        pg_losses2 = advantages_v * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        pg_loss = -torch.mean(torch.min(pg_losses1, pg_losses2))

        values_clipped = values_old_v + torch.clamp(values_v - values_old_v, - self.clip_range, self.clip_range)

        vf_losses1 = torch.pow((values_v - values_target_v), 2)
        vf_losses2 = torch.pow((values_clipped - values_target_v), 2)
        vf_loss = torch.mean(torch.max(vf_losses1, vf_losses2))

        return pg_loss + self.vf_coeff * vf_loss

    def update(self):
        states_t, values_old_t, actions_old_t, log_probs_old_t, rewards_t = self.replay_buffer.get_all()

        advantages_t = self._calculate_advantage(rewards_t, values_old_t)

        values_target_t = advantages_t + values_old_t

        dataset = TensorDataset(states_t, actions_old_t, advantages_t, values_target_t, log_probs_old_t, values_old_t)

        data_loader = DataLoader(dataset, self.batch_size, shuffle=True)

        for _ in range(self.num_epoch):
            for states_t, actions_old_t, advantages_t, values_target_t, log_probs_old_t, values_old_t in data_loader:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

                states_v = Variable(states_t)
                actions_old_v = Variable(actions_old_t)
                advantages_v = Variable(advantages_t)
                values_target_v = Variable(values_target_t)
                log_probs_old_v = Variable(log_probs_old_t)
                values_old_v = Variable(values_old_t)

                means_v, stds_v, values_v = self.policy(states_v)
                m_v = Normal(means_v, stds_v)
                log_probs_v = m_v.log_prob(actions_old_v)

                loss = self.loss(log_probs_v, log_probs_old_v, advantages_v, values_v, values_old_v, values_target_v)

                self.policy_optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
