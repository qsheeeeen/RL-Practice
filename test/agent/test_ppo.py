import time

import gym
import matplotlib.pyplot as plt

from self_driving_car.agent import PPOAgent
from self_driving_car.policy.shared import CNNPolicy


def main():
    total_reword = 0
    reward_history = []

    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    agent = PPOAgent(CNNPolicy, inputs, outputs, output_limit=(-1, 1), load=True)

    for i in range(2500):
        ob = env.reset()
        env.render()
        for x in range(1000):
            a = agent.act(ob)
            ob, r, d, _ = env.step(a)
            total_reword += r
            env.render()

        print()
        print(time.ctime())
        print('Done i:{},x:{} '.format(i, x))
        print('Total reward: {}'.format(total_reword))
        reward_history.append(total_reword)
        total_reword = 0

    name = 'car_baseline'

    plt.plot(reward_history)
    plt.title(name)
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.savefig('./img/' + name + '.png')


if __name__ == '__main__':
    main()
