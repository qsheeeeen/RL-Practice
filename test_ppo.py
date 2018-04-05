import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy.shared import MLPPolicy


def main():
    seed = 123
    env = gym.make('LunarLanderContinuous-v2')
    env.seed(seed)

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    torch.manual_seed(seed)

    policy = MLPPolicy(inputs, outputs)

    agent = PPOAgent(policy)

    reward_history = []
    for i in range(1000):
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
                print('- Done i:{}'.format(i))
                print('- Total reward: {:.6}'.format(total_reword))
                print('--------------------------------------')
                break
    env.close()

    plt.plot(reward_history)
    plt.title('LunarLanderContinuous-v2')
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.grid(True)
    plt.savefig('history.png')


if __name__ == '__main__':
    main()
