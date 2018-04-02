from self_driving_car.tool import Com


def main(virtual=False):
    print('Init com...', end='\t')
    client = Com('client', '192.168.1.103')
    print('Done.')

    print('Init env...', end='\t')
    if virtual:
        import gym
        env = gym.make('CarRacing-v0')

    else:
        from self_driving_car.environment import RacingCar
        env = RacingCar()
    print('Done.')

    try:
        for _ in range(2500):
            ob = env.reset()
            if virtual:
                env.render()
            client.send_data([ob])
            print('\\ \t Send ob.', end='\r')

            print('| \t Wait for action.', end='\r')
            data = client.receive_data()
            action = data[0]

            for _ in range(1000):
                ob, r, d, info = env.step(action)
                if virtual:
                    env.render()
                client.send_data([ob, r, d, info])
                print('/ \t Sent ob, r, d, info.', end='\r')

                print('- \t Wait for action.', end='\r')
                data = client.receive_data()
                action = data[0]
    finally:
        env.close()
        print('Closed.')


if __name__ == '__main__':
    main()
