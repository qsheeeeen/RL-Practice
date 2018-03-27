import gym

from self_driving_car.agent import PPOAgent
from self_driving_car.policy.shared import CNNPolicy
from self_driving_car.tool import Com


def main():
    server = Com('server')

    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    del env

    agent = PPOAgent(CNNPolicy, inputs, outputs)

    for _ in range(2500):
        print('wait for ob.')
        data = server.receive_data()
        ob = data[0]

        action = agent.act(ob)
        server.send_data([action])
        print('data sent.')

        for _ in range(1000):
            print('wait for ob, r, d, info')
            data = server.receive_data()
            ob, r, d, info = data

            action = agent.act(ob, r, d)
            server.send_data([action])
            print('data sent.')


if __name__ == '__main__':
    main()
