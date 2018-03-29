import copy

import torch
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Grayscale

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
            num_epoch=10,
            batch_size=64,
            discount_factor=0.99,
            gae_parameter=0.95,
            vf_coeff=0.5,
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
        self._vf_coeff = vf_coeff
        self._clip_range = clip_range
        self._train = train
        self._load = load
        self._save = save
        self._weight_path = weight_path

        self._preprocessing = Compose([
            ToPILImage(),
            Grayscale(),
            CenterCrop(72),
            ToTensor()])

        self._input_buffer = ReplayBuffer(3)

        self._policy_old = policy(input_shape, output_shape)
        self._policy_old.eval()

        if self._load:
            print('Load weights.')
            self._policy_old.load_state_dict(torch.load(self._weight_path))

        if self._train:
            self._policy = copy.deepcopy(self._policy_old)
            self._policy.train()

            self._policy_optimizer = Adam(self._policy.parameters(), lr=lr, eps=1e-5)
            self._policy_criterion = SmoothL1Loss().cuda()

            self._replay_buffer = ReplayBuffer(self._horizon)

            self._stored = None

        torch.backends.cudnn.benchmark = True

    def act(self, single_state, reward=0., done=False):
        single_state = self._processing(single_state)

        self._input_buffer.store(single_state)

        if len(self._input_buffer) == 3:
            state_sequence = self._input_buffer.get_all()
        else:
            state_sequence = single_state.expand(-1, 3, -1, -1)

        mean_var, std_var, value_var = self._policy_old(Variable(state_sequence, volatile=True))

        if self._train:
            m = Normal(mean_var, std_var)

            action_var = m.sample()

            value = value_var.data

            reward = torch.zeros_like(value) + reward

            if self._stored:
                self._replay_buffer.store(self._stored + [reward])

            if len(self._replay_buffer) == self._horizon:
                self._update_policy()

            self._stored = [state_sequence, value, action_var.data, m.log_prob(action_var).data]

        else:
            action_var = mean_var

        return torch.clamp(action_var.data, -self._output_limit, self._output_limit).cpu().numpy()[0]

    def save(self):
        torch.save(self._policy_old.state_dict(), self._weight_path)

    def _processing(self, array):
        tensor = self._preprocessing(array)

        return tensor.unsqueeze(0).cuda()

    @staticmethod
    def _calculate_advantage(rewards, values):
        advantages = rewards - values

        return advantages

    def _update_policy(self):
        states, values_old, actions_old, log_probs_old, rewards = self._replay_buffer.get_all()

        self._replay_buffer.clear()

        advantages = self._calculate_advantage(rewards, values_old)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_1 = TensorDataset(states, actions_old)
        dataset_2 = TensorDataset(advantages, rewards)
        dataset_3 = TensorDataset(log_probs_old, values_old)

        data_loader_1 = DataLoader(dataset_1, self._batch_size)
        data_loader_2 = DataLoader(dataset_2, self._batch_size)
        data_loader_3 = DataLoader(dataset_3, self._batch_size)

        for _ in range(self._num_epoch):
            for ((states, actions_old),
                 (advantages, rewards),
                 (log_probs_old, values_old)) in zip(data_loader_1,
                                                     data_loader_2,
                                                     data_loader_3):
                states_var = Variable(states)
                actions_old_var = Variable(actions_old)
                advantages_var = Variable(advantages)
                rewards_var = Variable(rewards)
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

                vf_losses1 = torch.pow((values_var - rewards_var), 2)
                vf_losses2 = torch.pow((values_clipped - rewards_var), 2)
                value_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))

                total_loss = pessimistic_surrogate + self._vf_coeff * value_loss

                self._policy_optimizer.zero_grad()
                total_loss.backward()
                self._policy_optimizer.step()

        self._policy_old.load_state_dict(self._policy.state_dict())

        if self._save:
            self.save()
