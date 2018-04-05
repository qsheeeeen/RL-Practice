import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp


class Runner(object):  # TODO
    def __init__(self, env_name, agent_fn, policy_fn, num_episode=1000, num_worker=1, seed=123):
        self.env_name = env_name
        self.agent_fn = agent_fn
        self.num_episode = num_episode
        self.num_worker = num_worker
        self.seed = seed

        env = gym.make(env_name)

        inputs = env.observation_space.shape
        outputs = env.action_space.shape

        del env

        mp.set_start_method('spawn')

        self.policy = policy_fn(inputs, outputs)
        self.policy.share_memory()

        torch.manual_seed(self.seed)

    def run(self):
        processes = []
        for i in range(self.num_worker):
            p = mp.Process(target=self.process, args=(self.env_name, self.agent_fn, self.policy, self.num_episode))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    @staticmethod
    def process(env_name, agent_fn, policy, num_episode):
        env = gym.make(env_name)

        agent = agent_fn(policy)

        reward_history = []
        for i in range(num_episode):
            ob = env.reset()
            env.render()
            action = agent.act(np.copy(ob))
            total_reword = 0

            while True:
                ob, r, d, _ = env.step(action)
                total_reword += r
                env.render()
                action = agent.act(np.copy(ob), r, d)
                if d:
                    reward_history.append(total_reword)
                    print('- Processing :{}'.format(os.getpid()))
                    print('- Done i:{}'.format(i))
                    print('- Total reward: {:.6}'.format(total_reword))
                    print('--------------------------------------')
                    break

        plt.plot(reward_history)
        plt.title('LunarLanderContinuous-v2')
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.grid(True)
        plt.savefig('history-{}.png'.format(os.getpid()))
