import os
import time

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
            seed=123):
        self.env_name = env_name
        self.agent_fn = agent_fn
        self.policy_fn = policy_fn
        self.record_data = record_data
        self.data_path = data_path
        self.save = save
        self.load = load
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

    def run(self, num_episode=1000, num_worker=1, draw_result=True):
        processes = []
        reward_queue = mp.Queue()

        for i in range(num_worker):
            time.sleep(i)

            args = (
                self.env_name,
                self.agent_fn,
                self.policy,
                num_episode,
                self.data_path,
                self.save,
                self.weight_path,
                reward_queue,
                self.seed + i)

            p = mp.Process(target=self.process, args=args)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if self.save:
            torch.save(self.policy.state_dict(), self.weight_path)

        if draw_result:
            while not reward_queue.empty():
                plt.plot(reward_queue.get())

            plt.title(self.env_name + '-{}-{}Process(es)'.format(self.policy.name, num_worker))
            plt.xlabel('episode')
            plt.ylabel('total reward')
            plt.grid(True)
            plt.savefig('./image/' + self.env_name + '-{}-{}Process(es).png'.format(self.policy.name, num_worker))

    @staticmethod
    def get_env_shape(env_name):
        env = gym.make(env_name)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape
        env.close()
        return inputs, outputs

    @staticmethod
    def process(env_name, agent_fn, policy, num_episode, data_path, save, weight_path, reward_queue, seed):
        save_interval = 10

        env = gym.make(env_name)
        env.seed(seed)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape

        torch.manual_seed(seed)

        if data_path is not None:
            recoder = Recoder(data_path + env_name + '-{}.hdf5'.format(policy.name), inputs, outputs)
        else:
            recoder = None

        agent = agent_fn(policy)

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
                    if save and (episode % save_interval == 0):
                        torch.save(policy.state_dict(), weight_path)
                        print('- Weights Saved.')
                    print('--------------------------------------')
                    break

        reward_queue.put(reward_history)

        if recoder is not None:
            recoder.close()
