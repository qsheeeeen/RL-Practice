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
            weight_path='./weights/ppo_weights.pth',
            seed=1):
        self.env_name = env_name
        self.agent_fn = agent_fn
        self.policy_fn = policy_fn
        self.record_data = record_data
        self.data_path = data_path
        self.save = save
        self.load = load
        self.weight_path = weight_path
        self.seed = seed

        env = gym.make(env_name)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape
        env.close()
        del env

        torch.manual_seed(self.seed)
        mp.set_start_method('spawn')

        self.policy = policy_fn(inputs, outputs)
        self.policy.share_memory()

        if self.load:
            self.policy.load_state_dict(torch.load(self.weight_path))

    def run(self, num_episode=1000, num_worker=1, train=True):
        processes = []

        for i in range(num_worker):
            self.seed += 1

            args = (
                self.env_name,
                self.agent_fn,
                self.policy,
                num_episode,
                train,
                self.data_path,
                self.save,
                self.weight_path,
                self.seed)

            p = mp.Process(target=self.process, args=args)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if self.save:
            torch.save(self.policy.state_dict(), self.weight_path)

    @staticmethod
    def process(env_name, agent_fn, policy, num_episode, train, data_path, save, weight_path, seed):
        save_interval = 5

        process_name = '{}'.format(os.getpid())
        env = gym.make(env_name)
        env.seed(seed)
        inputs = env.observation_space.shape
        outputs = env.action_space.shape

        torch.manual_seed(seed)

        if data_path is not None:
            recoder = Recoder(data_path + process_name + '.hdf5', inputs, outputs)
        else:
            recoder = None

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

                if recoder is not None:
                    recoder.store(ob, action, r, d)

                if d:
                    reward_history.append(total_reword)
                    print('- PID:{}'.format(os.getpid()))
                    print('- Done.\tEpisode:{}\tStep:{}'.format(episode, step))
                    print('- Total reward: {:.6}'.format(total_reword))
                    if save and (episode % save_interval == 0):
                        torch.save(policy.state_dict(), weight_path)
                        print('- Weights Saved.')
                    print('--------------------------------------')
                    break

        if recoder is not None:
            recoder.close()

        plt.plot(reward_history)
        plt.title(env_name)
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.grid(True)
        plt.savefig('./image/' + env_name + '-{}.png'.format(process_name))
