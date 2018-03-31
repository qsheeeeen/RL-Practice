import time

import gym

from self_driving_car.agent import JoystickAgent


def main():
    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    agent = JoystickAgent(inputs, outputs)

    for i in range(20):
        ob = env.reset()
        env.render()
        while True:
            a = agent.act(ob)
            ob, r, d, info = env.step(a)
            env.render()
            if d:
                print()
                print(time.ctime())
                print('Done i:{}'.format(i))
                break

    env.close()
    agent.close()


if __name__ == "__main__":
    main()
