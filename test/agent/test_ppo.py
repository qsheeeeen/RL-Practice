import time

import gym

from self_driving_car.agent import PPOAgent


def main():
    env = gym.make('CarRacing-v0')
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]
    agent = PPOAgent(None, outputs, load=False)
    for i in range(10000):
        ob = env.reset()
        env.render()
        action = agent.act(ob)
        for x in range(10000):
            ob, r, d, _ = env.step(action)
            env.render()
            action = agent.act(ob, r, d)
            if d:
                print()
                print(time.ctime())
                print('Done i:{},x:{} '.format(i, x))
                agent.save()
                break


if __name__ == '__main__':
    main()
