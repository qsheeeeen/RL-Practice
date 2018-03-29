import time

import gym

from self_driving_car.agent import PPOAgent
from self_driving_car.policy.shared import CNNPolicy


def main():
    reward_history = []

    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    agent = PPOAgent(CNNPolicy, inputs, outputs, output_limit=(-0.4, 0.4))

    for i in range(100):
        ob = env.reset()
        env.render()
        action = agent.act(ob)
        total_reword = 0
        for x in range(1000):
            ob, r, d, _ = env.step(action)
            total_reword += r
            if total_reword < -10:
                d = True
            env.render()
            action = agent.act(ob, r, d)
            if d:
                print()
                print(time.ctime())
                print('Done i:{},x:{} '.format(i, x))
                print('Total reward: {}'.format(total_reword))
                agent.save()
                reward_history.append(total_reword)
                break


if __name__ == '__main__':
    main()
