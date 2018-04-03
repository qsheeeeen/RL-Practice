import copy

import torch
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import SmoothL1Loss
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..util import ReplayBuffer, TensorDataset, processing_image


class PPOAgent(object):
    def __init__(
            self,
            policy,
            input_shape,
            output_shape,
            output_limit=1,
            horizon=2048,
            lr=3e-4,
            num_epoch=10,
            batch_size=32,
            clip_range=0.2,
            vf_coeff=1,
            discount_factor=0.99,
            gae_parameter=0.95,
            max_grad_norm=0.5,
            train=True,
            load_weight=False,
            save_weight=True,
            weight_path='./ppo_weights.pth'):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_limit = output_limit
        self.horizon = horizon
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.vf_coeff = vf_coeff
        self.discount_factor = discount_factor
        self.gae_parameter = gae_parameter

        self.max_grad_norm = max_grad_norm
        self.train = train
        self.load_weight = load_weight
        self.save_weight = save_weight
        self.weight_path = weight_path

        self.policy_old = policy(self.input_shape, self.output_shape)
        self.policy_old.eval()

        if self.load_weight:
            print('Load weights.')
            self.policy_old.load_state_dict(torch.load(self.weight_path))

        if self.train:
            self.policy = copy.deepcopy(self.policy_old)
            self.policy.train()

            self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
            self.policy_criterion = SmoothL1Loss().cuda()

            self.replay_buffer = ReplayBuffer(self.horizon)

            self.stored = None

        torch.backends.cudnn.benchmark = True

    def act(self, state, reward=0., done=False):
        state_t = processing_image(state)

        mean_v, std_v, value_v = self.policy_old(Variable(state_t.cuda(), volatile=True))

        if self.train:
            m_v = Normal(mean_v, std_v)
            action_v = m_v.sample()

            action_t = action_v.data.cpu()

            value_t = value_v.data.cpu()
            reward_t = torch.zeros_like(value_t) + reward

            if self.stored is not None:
                self.replay_buffer.store(self.stored + [reward_t])

            if self.replay_buffer.full():
                self._update_policy()
                self.replay_buffer.clear()

            self.stored = [state_t, value_t, action_t, m_v.log_prob(action_v).data.cpu()]

        else:
            action_t = mean_v.data.cpu()

        return torch.clamp(action_t, -self.output_limit, self.output_limit).numpy()[0]

    def save(self):
        torch.save(self.policy_old.state_dict(), self.weight_path)

    def close(self):
        pass

    def _calculate_advantage(self, rewards_t, values):
        advantages_t = torch.zeros_like(rewards_t)
        advantages_t[-1] = rewards_t[-1] - values[-1]

        for t in reversed(range(len(rewards_t) - 1)):
            delta = rewards_t[t] + self.discount_factor * values[t + 1] - values[t]
            advantages_t[t] = delta + self.discount_factor * self.gae_parameter * advantages_t[t + 1]

        return advantages_t

    def _update_policy(self):
        states_t, values_old_t, actions_old_t, log_probs_old_t, rewards_t = self.replay_buffer.get_all()

        advantages_t = self._calculate_advantage(rewards_t, values_old_t)

        values_target_t = advantages_t + values_old_t

        dataset = TensorDataset(states_t, actions_old_t, advantages_t, values_target_t, log_probs_old_t, values_old_t)

        data_loader = DataLoader(dataset, self.batch_size, shuffle=True)

        for _ in range(self.num_epoch):
            for states_t, actions_old_t, advantages_t, values_target_t, log_probs_old_t, values_old_t in data_loader:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

                states_v = Variable(states_t.cuda())
                actions_old_v = Variable(actions_old_t.cuda())
                advantages_v = Variable(advantages_t.cuda())
                values_target_v = Variable(values_target_t.cuda())
                log_probs_old_v = Variable(log_probs_old_t.cuda())
                values_old_v = Variable(values_old_t.cuda())

                means_v, stds_v, values_v = self.policy(states_v)
                m_v = Normal(means_v, stds_v)
                log_probs_v = m_v.log_prob(actions_old_v)

                ratio = torch.exp(log_probs_v - log_probs_old_v)

                pg_losses1 = advantages_v * ratio
                pg_losses2 = advantages_v * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = -torch.mean(torch.min(pg_losses1, pg_losses2))

                values_clipped = values_old_v + torch.clamp(values_v - values_old_v, - self.clip_range, self.clip_range)

                vf_losses1 = torch.pow((values_v - values_target_v), 2)
                vf_losses2 = torch.pow((values_clipped - values_target_v), 2)
                vf_loss = torch.mean(torch.max(vf_losses1, vf_losses2))

                total_loss = pg_loss + self.vf_coeff * vf_loss

                self.policy_optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        if self.save_weight:
            self.save()
