import copy

import torch
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from .noise import AdaptiveNoise
from .policy.shared import MLPPolicy, CNNPolicy
from .replay_buffer import ReplayBuffer


class PPOAgent(object):
    def __init__(
            self,
            input_shape,
            output_shape,
            output_limit=(-1, 1),
            horizon=2048,
            lr=3e-4,
            num_epoch=10,
            batch_size=64,
            discount_factor=0.99,
            gae_parameter=0.95,
            clip_range=0.2,
            train=True,
            load=False,
            save=True,
            weight_path='./ppo_weights.pth'):

        self._input_shape = input_shape
        self._output_limit = output_limit
        self._horizon = horizon
        self._lr = lr
        self._num_epoch = num_epoch
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._gae_parameter = gae_parameter
        self._clip_range = clip_range
        self._train = train
        self._load = load
        self._save = save
        self._weight_path = weight_path

        if len(input_shape) == 3:
            self._policy_old = CNNPolicy(output_shape[0])
        else:
            self._policy_old = MLPPolicy(input_shape[0], output_shape[0])

        self._policy_old.eval()

        if self._load:
            print('Load weights.')
            self._policy_old.load_state_dict(torch.load(self._weight_path))

        if self._train:
            self._policy = copy.deepcopy(self._policy_old)
            self._policy.train()

            self._policy_optimizer = Adam(self._policy.parameters(), lr=lr, eps=1e-5)
            self._policy_criterion = SmoothL1Loss().cuda()

            self._noise = AdaptiveNoise([self._policy.mean_fc])

            self._replay_buffer = ReplayBuffer(horizon)

            self._stored = None

        torch.backends.cudnn.benchmark = True

    def act(self, state, reward=0., done=False):
        state = self._processing(state)

        mean_var, std_var, value_var = self._policy_old(Variable(state, volatile=True))

        if self._train:
            m = Normal(mean_var, std_var)

            action_var = m.sample()

            value = value_var.data

            reward = torch.zeros_like(value) + reward

            if self._stored:
                self._replay_buffer.store(self._stored + [reward])

            if len(self._replay_buffer) == self._horizon:
                self._update_policy()

            self._stored = [state, value, action_var.data, m.log_prob(action_var).data]

        else:
            action_var = mean_var

        return torch.clamp(action_var.data, self._output_limit[0], self._output_limit[1]).cpu().numpy()[0]

    def save(self):
        torch.save(self._policy_old.state_dict(), self._weight_path)

    def _processing(self, array):
        try:
            tensor = torch.from_numpy(array).float()
        except RuntimeError:
            import numpy as np

            a = np.zeros_like(array) + array
            tensor = torch.from_numpy(a).float()

        if len(self._input_shape) == 3:
            tensor = tensor.permute(2, 0, 1)
            tensor = tensor / 128 - 1

        return tensor.unsqueeze(0).cuda()

    def _calculate_advantage(self, rewards, values, ):
        advantages = torch.zeros_like(rewards)
        advantages[-1] = rewards[-1] - values[-1]

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self._discount_factor * values[t + 1] - values[t]
            advantages[t] = delta + self._discount_factor * self._gae_parameter * advantages[t + 1]

        return advantages

    def _update_policy(self):
        states, values_old, actions_old, log_probs_old, rewards = self._replay_buffer.get_all()

        self._replay_buffer.clear()

        advantages = self._calculate_advantage(rewards, values_old)

        values_target = advantages + values_old

        # advantages = (advantages - advantages.mean()) / advantages.std()

        dataset_1 = TensorDataset(states, actions_old)
        dataset_2 = TensorDataset(advantages, values_target)
        dataset_3 = TensorDataset(log_probs_old, values_target)

        data_loader_1 = DataLoader(dataset_1, self._batch_size)
        data_loader_2 = DataLoader(dataset_2, self._batch_size)
        data_loader_3 = DataLoader(dataset_3, self._batch_size)

        for _ in range(self._num_epoch):
            for ((states, actions_old),
                 (advantages, values_target),
                 (log_probs_old, _)) in zip(data_loader_1,
                                            data_loader_2,
                                            data_loader_3):
                states_var = Variable(states)
                actions_old_var = Variable(actions_old)
                advantages_var = Variable(advantages)
                values_target_var = Variable(values_target)
                log_probs_old_var = Variable(log_probs_old)

                means_var, stds_var, values_var = self._policy(states_var)
                m = Normal(means_var, stds_var)
                log_probs_var = m.log_prob(actions_old_var)

                ratio = torch.exp(log_probs_var - log_probs_old_var)

                surrogate_1 = ratio * advantages_var
                surrogate_2 = torch.clamp(ratio, 1.0 - self._clip_range, 1.0 + self._clip_range) * advantages_var
                pessimistic_surrogate = -torch.mean(torch.min(surrogate_1, surrogate_2))

                value_loss = self._policy_criterion(values_var, values_target_var)

                total_loss = pessimistic_surrogate + value_loss

                self._policy_optimizer.zero_grad()
                total_loss.backward()
                self._policy_optimizer.step()

        self._policy_old.load_state_dict(self._policy.state_dict())

        if self._save:
            self.save()
