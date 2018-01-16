# coding: utf-8

import gym
import numpy as np


def main():
    # env = gym.make('Enduro-v0')
    env = gym.make('MountainCarContinuous-v0')
    observation = env.reset()

    # agent = AIAgent()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        print(action)
        print()
        observation, reward, done, info = env.step(action)

    print("Episode finished")


if __name__ == '__main__':
    main()
