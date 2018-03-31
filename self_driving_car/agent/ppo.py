import copy

import torch
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import SmoothL1Loss
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor

from .replay_buffer import ReplayBuffer


class PPOAgent(object):
    def __init__(
            self,
            policy,
            input_shape,
            output_shape,
            output_limit=1,
            horizon=2048,
            lr=3e-4,
            num_epoch=100,
            batch_size=64,
            clip_range=0.2,
            vf_coeff=0.5,
            discount_factor=0.99,
            gae_parameter=0.95,
            max_grad_norm=0.5,
            train=True,
            load=False,
            save=True,
            weight_path='./ppo_weights.pth'):

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._output_limit = output_limit
        self._horizon = horizon
        self._lr = lr
        self._num_epoch = num_epoch
        self._batch_size = batch_size
        self._clip_range = clip_range
        self._vf_coeff = vf_coeff
        self._discount_factor = discount_factor
        self._gae_parameter = gae_parameter

        self._max_grad_norm = max_grad_norm
        self._train = train
        self._load = load
        self._save = save
        self._weight_path = weight_path

        self._preprocessing = Compose([
            ToPILImage(),
            CenterCrop(72),
            ToTensor(),
        ])

        self._policy_old = policy(self._input_shape, self._output_shape)
        self._policy_old.eval()

        if self._load:
            print('Load weights.')
            self._policy_old.load_state_dict(torch.load(self._weight_path))

        if self._train:
            self._policy = copy.deepcopy(self._policy_old)
            self._policy.train()

            self._policy_optimizer = Adam(self._policy.parameters(), lr=self._lr, eps=1e-5)
            self._policy_criterion = SmoothL1Loss().cuda()

            self._replay_buffer = ReplayBuffer(self._horizon)

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

            if self._stored is not None:
                self._replay_buffer.store(self._stored + [reward])

            if len(self._replay_buffer) == self._horizon:
                self._update_policy()

            self._stored = [state, value, action_var.data, m.log_prob(action_var).data]

        else:
            action_var = mean_var

        return torch.clamp(action_var.data.cpu(), -self._output_limit, self._output_limit).numpy()[0]

    def save(self):
        torch.save(self._policy_old.state_dict(), self._weight_path)

    def _processing(self, array):
        if len(array.shape) == 3:
            tensor = self._preprocessing(array).float()
        elif len(array.shape) == 1:
            tensor = torch.from_numpy(array).float()
        else:
            raise NotImplementedError

        return tensor.unsqueeze(0).cuda()

    def _calculate_advantage(self, rewards, values):
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

        dataset_1 = TensorDataset(states, actions_old)
        dataset_2 = TensorDataset(advantages, values_target)
        dataset_3 = TensorDataset(log_probs_old, values_old)

        data_loader_1 = DataLoader(dataset_1, self._batch_size)
        data_loader_2 = DataLoader(dataset_2, self._batch_size)
        data_loader_3 = DataLoader(dataset_3, self._batch_size)

        for _ in range(self._num_epoch):
            for ((states, actions_old),
                 (advantages, values_target),
                 (log_probs_old, values_old)) in zip(data_loader_1,
                                                     data_loader_2,
                                                     data_loader_3):

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                states_var = Variable(states)
                actions_old_var = Variable(actions_old)
                advantages_var = Variable(advantages)
                values_target_var = Variable(values_target)
                log_probs_old_var = Variable(log_probs_old)
                values_old_var = Variable(values_old)

                means_var, stds_var, values_var = self._policy(states_var)
                m = Normal(means_var, stds_var)
                log_probs_var = m.log_prob(actions_old_var)

                ratio = torch.exp(log_probs_var - log_probs_old_var)

                surrogate_1 = advantages_var * ratio
                surrogate_2 = advantages_var * torch.clamp(ratio, 1.0 - self._clip_range, 1.0 + self._clip_range)
                pessimistic_surrogate = -torch.mean(torch.min(surrogate_1, surrogate_2))

                values_clipped = values_old_var + torch.clamp(values_var - values_old_var,
                                                              - self._clip_range,
                                                              self._clip_range)

                vf_losses1 = torch.pow((values_var - values_target_var), 2)
                vf_losses2 = torch.pow((values_clipped - values_target_var), 2)
                value_function_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))

                total_loss = pessimistic_surrogate + self._vf_coeff * value_function_loss

                self._policy_optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm(self._policy.parameters(), self._max_grad_norm)
                self._policy_optimizer.step()

        self._policy_old.load_state_dict(self._policy.state_dict())

        if self._save:
            self.save()
