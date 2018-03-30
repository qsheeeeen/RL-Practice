import gym
import matplotlib.pyplot as plt
import numpy as np

from self_driving_car.agent import PPOAgent
from self_driving_car.policy.shared import CNNPolicy


def main():
    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    agent = PPOAgent(CNNPolicy, inputs, outputs, load=False)

    reward_history = []
    for i in range(1000):
        ob = env.reset()
        env.render()
        action = agent.act(ob)
        total_reword = 0
        for x in range(1000):
            ob, r, d, _ = env.step(action)
            total_reword += r
            env.render()
            action = agent.act(ob, r, d)
            if d:
                reward_history.append(total_reword)
                print('-------------------------------------------------------')
                print('- Done i:{},x:{} '.format(i, x))
                print('- Total reward: {:.4}'.format(total_reword))
                print('- Average reward: {:.4}'.format(np.mean(reward_history)))
                print('-------------------------------------------------------')
                break

    plt.imshow(reward_history)
    plt.imsave('.history.png')


if __name__ == '__main__':
    main()
