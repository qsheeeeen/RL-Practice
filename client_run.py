from .self_driving_car.environment import RacingCar
from .self_driving_car.tool import Com


def main():
    client = Com('client', '127.0.0.1')
    env = RacingCar()
    data = env.reset()

    while True:
        client.send_data(data)

        action = client.receive_data()

        env.step(action)


if __name__ == '__main__':
    main()
