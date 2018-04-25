import copy

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .core import Agent
from ..util import ReplayBuffer
from ..util.common import preprocessing_state


class PPOAgent(Agent):
    def __init__(self, policy, train=True, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_old = policy.to(self.device)
        self.train = train

        default_kwargs = {
            'abs_output_limit': 1,
            'horizon': 2048,
            'lr': 3e-4,
            'num_epoch': 10,
            'batch_size': 32,
            'clip_range': 0.2,
            'vf_coeff': 1,
            'discount_factor': 0.99,
            'gae_parameter': 0.95,
            'max_grad_norm': 0.5}

        for kwarg in kwargs:
            if kwarg not in default_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        self.abs_output_limit = kwargs.get('abs_output_limit') or default_kwargs.get('abs_output_limit')
        self.horizon = kwargs.get('horizon') or default_kwargs.get('horizon')
        self.lr = kwargs.get('lr') or default_kwargs.get('lr')
        self.num_epoch = kwargs.get('num_epoch') or default_kwargs.get('num_epoch')
        self.batch_size = kwargs.get('batch_size') or default_kwargs.get('batch_size')
        self.clip_range = kwargs.get('clip_range') or default_kwargs.get('clip_range')
        self.vf_coeff = kwargs.get('vf_coeff') or default_kwargs.get('vf_coeff')
        self.discount_factor = kwargs.get('discount_factor') or default_kwargs.get('discount_factor')
        self.gae_parameter = kwargs.get('gae_parameter') or default_kwargs.get('gae_parameter')
        self.max_grad_norm = kwargs.get('max_grad_norm') or default_kwargs.get('max_grad_norm')

        if self.train:
            self.policy_old.train()
            self.policy = copy.deepcopy(self.policy_old)

            self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

            self.replay_buffer = ReplayBuffer(self.horizon)

            self.stored = None
        else:
            self.policy_old.eval()

        torch.backends.cudnn.benchmark = True

    def act(self, state, reward=0., done=False):
        state = preprocessing_state(state).to(self.device)

        with torch.no_grad():
            action, value = self.policy_old(state)

        if self.train:
            reward = torch.zeros_like(value) + reward

            if self.stored is not None:
                self.replay_buffer.store(self.stored + [reward])

            if self.replay_buffer.full():
                self._update_policy()
                self.replay_buffer.clear()

            self.stored = [state.detach(), value.detach(), action.detach(), self.policy_old.log_prob(action).detach()]

        return torch.clamp(action, -self.abs_output_limit, self.abs_output_limit).to('cpu').numpy()[0]

    def _calculate_advantage(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        advantages[-1] = rewards[-1] - values[-1]

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.discount_factor * values[t + 1] - values[t]
            advantages[t] = delta + self.discount_factor * self.gae_parameter * advantages[t + 1]

        return advantages

    def _update_policy(self):
        states, values_old, actions_old, log_probs_old, rewards = self.replay_buffer.get_all()

        advantages = self._calculate_advantage(rewards, values_old)

        values_target = advantages + values_old

        dataset = TensorDataset(states, actions_old, advantages, values_target, log_probs_old, values_old)

        data_loader = DataLoader(dataset, self.batch_size, shuffle=not self.policy.recurrent)

        self.policy.load_state_dict(self.policy_old.state_dict())

        for _ in range(self.num_epoch):
            for states, actions_old, advantages, values_target, log_probs_old, values_old in data_loader:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                states = states.to(self.device)
                actions_old = actions_old.to(self.device)
                advantages = advantages.to(self.device)
                values_target = values_target.to(self.device)
                log_probs_old = log_probs_old.to(self.device)
                values_old = values_old.to(self.device)

                _, values = self.policy(states)
                log_probs = self.policy.log_prob(actions_old)

                ratio = torch.exp(log_probs - log_probs_old)

                pg_losses1 = advantages * ratio
                pg_losses2 = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = -torch.mean(torch.min(pg_losses1, pg_losses2))

                values_clipped = values_old + torch.clamp(values - values_old, - self.clip_range, self.clip_range)

                vf_losses1 = torch.pow((values - values_target), 2)
                vf_losses2 = torch.pow((values_clipped - values_target), 2)
                vf_loss = torch.mean(torch.max(vf_losses1, vf_losses2))

                total_loss = pg_loss + self.vf_coeff * vf_loss

                self.policy_optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
