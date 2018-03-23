import time

import gym

from self_driving_car.agent import KeyboardAgent


def main():
    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    agent = KeyboardAgent(inputs, outputs)

    env.render()
    env.unwrapped.viewer.window.on_key_press = agent.key_press
    env.unwrapped.viewer.window.on_key_release = agent.key_release

    for i in range(20):
        ob = env.reset()
        env.render()
        for _ in range(1000):
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
