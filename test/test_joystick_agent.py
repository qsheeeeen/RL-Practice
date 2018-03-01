# coding: utf-8

import gym
import time
import numpy as np

from ..agent import JoystickAgent


def convert_action(action):
    x = np.array([0, 0, 0])
    x[0] = action[0]
    if action[1] > 0:
        x[1] = action[1]
    else:
        x[2] = -action[1]
    return x


if __name__ == '__main__':
    agent = JoystickAgent()
    env = gym.make('CarRacing-v0')
    for i in range(1000):
        print('Begin.')
        ob = env.reset()
        env.render()
        action = agent.act(ob)

        action = convert_action(action)

        for x in range(10000):
            ob, r, d, _ = env.step(action)
            env.render()
            action = agent.act(ob, r, d)

            action = convert_action(action)

            if d:
                print('Done i:{},x:{} '.format(i, x))
                print(time.ctime())
                print()
                d = False
                agent.save()
                break
