import gym

from self_driving_car.tool import Com


def main():
    print('Init com...', end='')
    client = Com('client', '127.0.0.1')
    print('Done')

    print('Init env...', end='')
    env = gym.make('CarRacing-v0')
    print('Done')

    for _ in range(2500):
        ob = env.reset()
        env.render()

        client.send_data([ob])
        print('\\ \t Send ob', end='\r')

        print('| \t Wait for action', end='\r')
        data = client.receive_data()
        action = data[0]

        for _ in range(1000):
            ob, r, d, info = env.step(action)
            env.render()

            client.send_data([ob, r, d, info])
            print('/ \t Sent ob, r, d, info', end='\r')

            print('- \t Wait for action', end='\r')
            data = client.receive_data()
            action = data[0]


if __name__ == '__main__':
    main()
