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
        print('\\\t Wait for ob.', end='\r')
        data = server.receive_data()
        ob = data[0]

        action = agent.act(ob)
        server.send_data([action])
        print('|\t Data sent.', end='\r')

        for _ in range(1000):
            print('/\t Wait for ob, r, d, info.', end='\r')
            data = server.receive_data()
            ob, r, d, info = data

            action = agent.act(ob, r, d)
            server.send_data([action])
            print('-\t Data sent.', end='\r')


if __name__ == '__main__':
    main()
