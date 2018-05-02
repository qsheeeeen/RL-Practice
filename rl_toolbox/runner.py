import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

from .util import Recoder


class Runner(object):
    def __init__(
            self,
            env_name,
            agent_fn,
            policy_fn,
            record_data=False,
            data_path=None,
            save=True,
            load=False,
            weight_path='./weights/',
            image_path='./image/',
            seed=1):
        self.env_name = env_name
        self.agent_fn = agent_fn
        self.policy_fn = policy_fn
        self.record_data = record_data
        self.data_path = data_path
        self.save = save
        self.load = load
        self.image_path = image_path
        self.seed = seed

        torch.manual_seed(self.seed)

        if mp.get_start_method() != 'spawn':
            mp.set_start_method('spawn', True)

        inputs, outputs = self.get_env_shape(env_name)

        self.policy = policy_fn(inputs, outputs)
        self.policy.share_memory()

        self.weight_path = weight_path + self.policy.name + '_weights.pth'

        if self.load:
            self.policy.load_state_dict(torch.load(self.weight_path))

    def run(self, agent_kwargs=None, num_episode=2000, draw_result=True, continue_plot=False):
        save_interval = 10

        torch.manual_seed(self.seed)

        env = gym.make(self.env_name)
        env.seed(self.seed)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape

        if self.data_path is not None:
            recoder = Recoder(self.data_path + self.env_name + '-{}.hdf5'.format(self.policy.name), inputs, outputs)
        else:
            recoder = None

        if agent_kwargs is not None:
            agent = self.agent_fn(self.policy, **agent_kwargs)
        else:
            agent = self.agent_fn(self.policy)

        reward_history = []
        for episode in range(num_episode):
            ob = env.reset()
            env.render()
            action = agent.act(np.copy(ob))
            total_reword = 0

            for step in range(1000):
                ob, r, d, _ = env.step(action)
                total_reword += r
                env.render()
                action = agent.act(np.copy(ob), r, d)

                if recoder is not None:
                    recoder.store(ob, action, r, d)

                if d:
                    reward_history.append(total_reword)
                    print('- PID:{}'.format(os.getpid()))
                    print('- Done.\tEpisode:{}\tStep:{}'.format(episode, step))
                    print('- Total reward:\t{:.6}'.format(total_reword))
                    if self.save and ((episode + 1) % save_interval == 0):
                        torch.save(self.policy.state_dict(), self.weight_path)
                        print('- Weights Saved.')
                    print('--------------------------------------')
                    break

        env.close()

        if recoder is not None:
            recoder.close()

        if self.save:
            torch.save(self.policy.state_dict(), self.weight_path)

        if draw_result:
            print('Draw result.')
            plt.plot(reward_history)

            plt.title('{}-{}-{}Process(es)'.format(self.env_name, self.policy.name, num_worker))
            plt.xlabel('episode')
            plt.ylabel('total reward')
            plt.grid(True)
            plt.savefig(self.image_path + '{}-{}-{}Process(es).png'.format(self.env_name, self.policy.name, num_worker))
            if not continue_plot:
                plt.close()

    @staticmethod
    def get_env_shape(env_name):
        env = gym.make(env_name)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape
        env.close()
        return inputs, outputs
