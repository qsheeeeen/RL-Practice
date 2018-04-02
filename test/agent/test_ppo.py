import gym
import matplotlib.pyplot as plt
import torch

from self_driving_car.agent import PPOAgent
from self_driving_car.policy.shared import CNNPolicy
# from self_driving_car.policy.shared import MLPPolicy
import numpy as np
SEED = 123


def main():
    env = gym.make('CarRacing-v0')
    # env = gym.make('LunarLanderContinuous-v2')
    env.seed(SEED)

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    torch.manual_seed(SEED)

    agent = PPOAgent(CNNPolicy, inputs, outputs, horizon=128, lr=2.5e-4, num_epoch=4, batch_size=4, clip_range=0.2)
    # agent = PPOAgent(CNNPolicy, inputs, outputs)
    # agent = PPOAgent(MLPPolicy, inputs, outputs)

    reward_history = []
    for i in range(1000):
        ob = env.reset()

        ob = np.copy(ob)

        env.render()
        action = agent.act(ob)
        total_reword = 0
        while True:
            ob, r, d, _ = env.step(action)
            ob = np.copy(ob)
            total_reword += r
            env.render()
            action = agent.act(ob, r, d)
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


if __name__ == '__main__':
    main()
