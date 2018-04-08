import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

from .util import Recoder


class Runner(object):
    def __init__(self, env_name,
                 agent_fn,
                 policy_fn,
                 seed=123,
                 save=True,
                 load=False,
                 data_path=None,
                 weight_path='./weights/weights.pth'):
        self.env_name = env_name
        self.agent_fn = agent_fn
        self.save = save
        self.data_path = data_path
        self.weight_path = weight_path

        env = gym.make(env_name)
        env.seed(seed)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape
        del env

        mp.set_start_method('spawn')

        torch.manual_seed(seed)

        self.policy = policy_fn(inputs, outputs)
        self.policy.share_memory()

        if load:
            self.policy.load_state_dict(torch.load(weight_path))

    def run(self, num_episode=1000, num_worker=1, train=True):
        processes = []
        for i in range(num_worker):
            args = (self.env_name, self.agent_fn, self.policy, num_episode, self.data_path, train)
            p = mp.Process(target=self.process, args=args)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        if self.save:
            torch.save(self.policy.state_dict(), self.weight_path)

    @staticmethod
    def process(env_name, agent_fn, policy, num_episode, data_path, train):
        process_name = '{}'.format(os.getpid())
        env = gym.make(env_name)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape

        if data_path is not None:
            recoder = Recoder(data_path + process_name + '.hdf5', inputs, outputs)

        agent = agent_fn(policy, train=train)

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

                if data_path is not None:
                    recoder.store(ob, action, r, d)

                if d:
                    reward_history.append(total_reword)
                    print('- PID:{}'.format(os.getpid()))
                    print('- Done.\tEpisode:{}\tStep:{}'.format(episode, step))
                    print('- Total reward: {:.6}'.format(total_reword))
                    print('--------------------------------------')
                    break

        if data_path is not None:
            recoder.close()

        plt.plot(reward_history)
        plt.title(env_name)
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.grid(True)
        plt.savefig('./image/' + env_name + '-{}.png'.format(process_name))
