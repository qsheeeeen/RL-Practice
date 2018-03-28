from self_driving_car.tool import Com
from self_driving_car.environment import RacingCar


def main():
    print('Init com...', end='')
    client = Com('client', '192.168.1.100')
    print('Done')

    print('Init env...', end='')
    env = RacingCar()
    print('Done')

    for _ in range(2500):
        ob = env.reset()

        client.send_data([ob])
        print('\\ \t Send ob', end='\r')

        print('| \t Wait for action', end='\r')
        data = client.receive_data()
        action = data[0]

        for _ in range(1000):
            ob, r, d, info = env.step(action)

            client.send_data([ob, r, d, info])
            print('/ \t Sent ob, r, d, info', end='\r')

            print('- \t Wait for action', end='\r')
            data = client.receive_data()
            action = data[0]


if __name__ == '__main__':
    main()
