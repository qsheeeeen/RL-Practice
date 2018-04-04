from self_driving_car.agent import RandomAgent
from self_driving_car.tool import Com
from self_driving_car.tool.dashboard import Dashboard


def main(virtual=False):
    print('Init dashboard', end='\t')
    dashboard = Dashboard()
    print('Done.')

    print('Init com...', end='\t')
    server = Com('server')
    print('Done.')

    inputs = (96, 96, 3)
    if virtual:
        outputs = (3,)
    else:
        outputs = (2,)

    print('Init agent...', end='\t')
    agent = RandomAgent(inputs, outputs)
    print('Done.')

    try:
        for _ in range(2500):
            print('\\\t Wait for ob.             ', end='\r')
            data = server.receive_data()
            ob = data[0]

            action = agent.act(ob)
            print('|\t Data sent.                ', end='\r')
            server.send_data([action])

            for _ in range(1000):
                print('/\t Wait for ob, r, d, info.  ', end='\r')
                data = server.receive_data()
                ob, r, d, info = data

                if virtual:
                    info = {'action': action}

                dashboard.update(ob, info)

                action = agent.act(ob, r, d)
                server.send_data([action])
                print('-\t Data sent.                ', end='\r')
                if d:
                    break
    finally:
        agent.close()


if __name__ == '__main__':
    main()
