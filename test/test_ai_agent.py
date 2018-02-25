# coding: utf-8

import gym
from ..agent import PPOAgent


def main():
    agent = PPOAgent()
    env = gym.make('CarRacing-v0')

    ob = env.reset()
    env.render()
    action = agent.act(ob)
    while True:
        ob, reward, done, _ = env.step(action)
        env.render()
        action = agent.act(ob, reward, done)

        if done:
            break


if __name__ == '__main__':
    main()
