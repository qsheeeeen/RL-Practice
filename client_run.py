import gym

from self_driving_car.tool import Com


def main():
    client = Com('client', '127.0.0.1')

    env = gym.make('CarRacing-v0')

    for _ in range(2500):
        ob = env.reset()
        env.render()

        client.send_data([ob])
        print('send ob')

        print('wait for action')
        data = client.receive_data()
        action = data[0]

        for _ in range(1000):
            ob, r, d, info = env.step(action)
            env.render()

            client.send_data([ob, r, d, info])
            print('sent ob, r, d, info')

            print('wait for action')
            data = client.receive_data()
            action = data[0]


if __name__ == '__main__':
    main()
