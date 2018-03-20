import time

import gym
import matplotlib.pyplot as plt

from self_driving_car.agent import PPOAgent


def main():
    total_reword = 0
    reward_history = []

    env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('CarRacing-v0')
    inputs = env.observation_space.shape
    outputs = env.action_space.shape
    agent = PPOAgent(inputs, outputs, load=False)
    for i in range(2500):
        ob = env.reset()
        env.render()
        action = agent.act(ob)
        for x in range(1000):
            ob, r, d, _ = env.step(action)
            total_reword += r
            env.render()
            action = agent.act(ob, r, d)
            if d:
                print()
                print(time.ctime())
                print('Done i:{},x:{} '.format(i, x))
                print('Total reward: {}'.format(total_reword))
                agent.save()
                reward_history.append(total_reword)
                total_reword = 0
                break

    plt.plot(reward_history)
    plt.savefig('./img/baseline.png')


if __name__ == '__main__':
    main()
