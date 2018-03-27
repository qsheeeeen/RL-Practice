import gym

from .self_driving_car.agent import PPOAgent
from .self_driving_car.policy.shared import CNNPolicy
from .self_driving_car.tool import Com, Dashboard


def main():
    server = Com('server')

    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    del env

    agent = PPOAgent(CNNPolicy, inputs, outputs, output_limit=(-1, 1), load=True)

    # dashboard = Dashboard()

    while True:
        data = server.receive_data()

        action = agent.act(*data[:3])

        server.send_data(action)

        # dashboard.update(data[0], data[-1])


if __name__ == '__main__':
    main()
