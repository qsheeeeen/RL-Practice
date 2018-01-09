# coding: utf-8

import gym
import numpy as np


# TODO: usd Enduro in gym
def main():
    env = gym.make('Enduro-v0')
    observation = env.reset()

    # agent = AIAgent()
    while True:
        env.render()
        print(observation)
        action = np.random.randint(0, 2, (9,), np.bool)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished")
            break


if __name__ == '__main__':
    main()
