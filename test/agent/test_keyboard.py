import time

import gym

from self_driving_car.agent import KeyboardAgent


def main():
    env = gym.make('CarRacing-v0')
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]
    agent = KeyboardAgent()
    env.render()
    env.unwrapped.viewer.window.on_key_press = agent.key_press
    env.unwrapped.viewer.window.on_key_release = agent.key_release
    for i in range(1):
        s = env.reset()
        while True:
            s, r, d, info = env.step(agent.act(s))
            env.render()
            if d:
                print()
                print(time.ctime())
                print('Done i:{}'.format(i))
                # agent.save()
                break
    env.close()
    agent.close()


if __name__ == "__main__":
    main()
