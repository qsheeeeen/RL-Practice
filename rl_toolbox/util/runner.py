import gym
import matplotlib.pyplot as plt
import numpy as np
import torch


class Runner(object):
    def __init__(self, env_name, agent_fn, policy_fn, seed=123, agent_args=None):
        self.env = gym.make(env_name)
        self.env.seed(seed)

        inputs = self.env.observation_space.shape
        outputs = self.env.action_space.shape

        torch.manual_seed(seed)

        if agent_args is None:
            self.agent = agent_fn(policy_fn, inputs, outputs)
        else:
            self.agent = agent_fn(policy_fn, inputs, outputs, *agent_args)

    def run(self, ):
        reward_history = []
        for i in range(1000):
            ob = self.env.reset()
            self.env.render()
            action = self.agent.act(np.copy(ob))
            total_reword = 0

            while True:
                ob, r, d, _ = self.env.step(action)
                total_reword += r
                self.env.render()
                action = self.agent.act(np.copy(ob), r, d)
                if d:
                    reward_history.append(total_reword)
                    print('- Done i:{}'.format(i))
                    print('- Total reward: {:.6}'.format(total_reword))
                    print('--------------------------------------')
                    break

        plt.plot(reward_history)
        plt.title('LunarLanderContinuous-v2')
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.grid(True)
        plt.savefig('history.png')
