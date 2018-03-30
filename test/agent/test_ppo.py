import gym
import matplotlib.pyplot as plt
import numpy as np

from self_driving_car.agent import PPOAgent
# from self_driving_car.policy.shared import CNNPolicy
from self_driving_car.policy.shared import MLPPolicy


def main():
    # env = gym.make('CarRacing-v0')
    env = gym.make('LunarLanderContinuous-v2')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    # agent = PPOAgent(CNNPolicy, inputs, outputs, horizon=128, lr=2.5e-4, num_epoch=4, batch_size=4, clip_range=0.1)
    agent = PPOAgent(MLPPolicy, inputs, outputs, horizon=2048, lr=3e-4, num_epoch=10, batch_size=32, clip_range=0.2)

    reward_history = []
    for i in range(1000):
        ob = env.reset()
        env.render()
        action = agent.act(ob)
        total_reword = 0
        for x in range(1000):
            ob, r, d, _ = env.step(action)
            total_reword += r
            env.render()
            action = agent.act(ob, r, d)
            if d:
                reward_history.append(total_reword)
                print('--------------------------------------')
                print('- Done i:{}, x:{} '.format(i, x))
                print('- Total reward:\t{:.6}'.format(total_reword))
                print('- Average reward:\t{:.6}'.format(np.mean(reward_history)))
                print('--------------------------------------')
                break

    plt.plot(reward_history)
    plt.savefig('history.png')


if __name__ == '__main__':
    main()
