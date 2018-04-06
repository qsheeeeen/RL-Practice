import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp


class Runner(object):
    def __init__(self, env_name,
                 agent_fn,
                 policy_fn,
                 seed=123,
                 save=True,
                 load=False,
                 weight_path='./weights.pth'):
        self.env_name = env_name
        self.agent_fn = agent_fn
        self.save = save
        self.weight_path = weight_path

        env = gym.make(env_name)
        env.seed(seed)

        inputs = env.observation_space.shape
        outputs = env.action_space.shape

        del env

        mp.set_start_method('spawn')

        self.policy = policy_fn(inputs, outputs)
        self.policy.share_memory()

        if load:
            self.policy.load_state_dict(torch.load(weight_path))

        torch.manual_seed(seed)

    def run(self, num_episode=1000, num_worker=1):
        processes = []
        for i in range(num_worker):
            p = mp.Process(target=self.process, args=(self.env_name, self.agent_fn, self.policy, num_episode))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        if self.save:
            torch.save(self.policy.state_dict(), self.weight_path)

    @staticmethod
    def process(env_name, agent_fn, policy, num_episode):
        env = gym.make(env_name)

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
                if d:
                    reward_history.append(total_reword)
                    print('- PID:{}'.format(os.getpid()))
                    print('- Done.\tEpisode:{}\tStep:{}'.format(episode, step))
                    print('- Total reward: {:.6}'.format(total_reword))
                    print('--------------------------------------')
                    break

        plt.plot(reward_history)
        plt.title('LunarLanderContinuous-v2')
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.grid(True)
        plt.savefig('history-{}.png'.format(os.getpid()))
