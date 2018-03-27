from .self_driving_car.agent import PPOAgent, JoystickAgent
from .self_driving_car.policy.shared import CNNPolicy
from .self_driving_car.tool import Com, Dashboard


def main():
    server = Com('server')
    # agent = PPOAgent(CNNPolicy, )
    agent = JoystickAgent((96, 96, 3), (2,))
    dashboard = Dashboard()

    while True:
        data = server.receive_data()

        action = agent.act(*data[:3])

        server.send_data(action)

        dashboard.update(data[0], data[-1])


if __name__ == '__main__':
    main()
