import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from .core import Agent
from ..util import ReplayBuffer
from ..util.common import preprocessing_state


class PPOAgent(Agent):
    def __init__(self, policy, **kwargs):
        default_kwargs = {
            'train': True,
            'use_gpu': True,
            'horizon': 256,
            'buffer_size': 8,
            'lr': 2.5e-4,
            'num_epoch': 3,
            'batch_size': 32 * 8,
            'clip_range': 0.1,
            'vf_coeff': 1,
            'discount_factor': 0.99,
            'gae_parameter': 0.95,
            'max_grad_norm': 0.5}

        for kwarg in kwargs:
            if kwarg not in default_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        self.train = default_kwargs.get('train') if kwargs.get('train') is None else kwargs.get('train')
        self.use_gpu = default_kwargs.get('use_gpu') if kwargs.get('use_gpu') is None else kwargs.get('use_gpu')

        self.horizon = kwargs.get('horizon') or default_kwargs.get('horizon')
        self.buffer_size = kwargs.get('buffer_size') or default_kwargs.get('buffer_size')
        self.lr = kwargs.get('lr') or default_kwargs.get('lr')
        self.num_epoch = kwargs.get('num_epoch') or default_kwargs.get('num_epoch')
        self.batch_size = kwargs.get('batch_size') or default_kwargs.get('batch_size')
        self.clip_range = kwargs.get('clip_range') or default_kwargs.get('clip_range')
        self.vf_coeff = kwargs.get('vf_coeff') or default_kwargs.get('vf_coeff')
        self.discount_factor = kwargs.get('discount_factor') or default_kwargs.get('discount_factor')
        self.gae_parameter = kwargs.get('gae_parameter') or default_kwargs.get('gae_parameter')
        self.max_grad_norm = kwargs.get('max_grad_norm') or default_kwargs.get('max_grad_norm')

        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.use_gpu) else 'cpu')
        self.policy_old = policy.to(self.device)

        if self.train:
            self.policy_old.train()
            self.policy = copy.deepcopy(self.policy_old)
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

            self.horizon_buffer = ReplayBuffer(self.horizon)
            self.training_buffer = ReplayBuffer(self.buffer_size)

            self.stored = None
        else:
            self.policy_old.eval()

    def act(self, state, reward=0., done=False):
        state = preprocessing_state(state).to(self.device)

        with torch.no_grad():
            action, value = self.policy_old(state)

        if self.train:
            reward = torch.full_like(value, reward)

            if self.stored is not None:
                self.horizon_buffer.store(self.stored + [reward])

            if self.horizon_buffer.full():
                states, values_old, actions_old, log_probs_old, rewards = self.horizon_buffer.get_all()

                advantages = self._calculate_advantage(rewards, values_old)

                values_target = advantages + values_old

                self.training_buffer.store([states, actions_old, advantages, values_target, log_probs_old, values_old])
                self.horizon_buffer.clear()

            if self.training_buffer.full():
                self._update_policy()
                self.training_buffer.clear()

            self.stored = [
                state.detach(),
                value.detach(),
                action.detach(),
                self.policy_old.pd.log_prob(action).detach()]

        return action.to('cpu').squeeze(0).numpy()

    def _calculate_advantage(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        advantages[-1] = rewards[-1] - values[-1]

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.discount_factor * values[t + 1] - values[t]
            advantages[t] = delta + self.discount_factor * self.gae_parameter * advantages[t + 1]

        return advantages

    def _update_policy(self):
        states, actions_old, advantages, values_target, log_probs_old, values_old = self.training_buffer.get_all()

        dataset = TensorDataset(states, actions_old, advantages, values_target, log_probs_old, values_old)

        data_loader = DataLoader(dataset, self.batch_size, shuffle=not self.policy.recurrent)

        for _ in range(self.num_epoch):
            for states, actions_old, advantages, values_target, log_probs_old, values_old in data_loader:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                actions, values = self.policy(states)
                log_probs = self.policy.pd.log_prob(actions_old)

                ratio = (log_probs - log_probs_old).exp()

                pg_losses1 = advantages * ratio
                pg_losses2 = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = -torch.min(pg_losses1, pg_losses2).mean()

                values_clipped = values_old + torch.clamp(values - values_old, - self.clip_range, self.clip_range)

                vf_losses1 = (values - values_target).pow(2)
                vf_losses2 = (values_clipped - values_target).pow(2)
                vf_loss = torch.max(vf_losses1, vf_losses2).mean()

                total_loss = pg_loss + self.vf_coeff * vf_loss

                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
