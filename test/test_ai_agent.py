# coding: utf-8

import gym


# TODO: Test DDPG
def main():
    env = gym.make('CarRacing-v0')
    observation = env.reset()

    # agent = AIAgent()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)


print("Episode finished")

if __name__ == '__main__':
    main()
