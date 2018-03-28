from self_driving_car.agent import JoystickAgent
from self_driving_car.tool import Com


def main():
    print('Init com...', end='')
    server = Com('server')
    print('Done')

    inputs = (96, 96, 3)
    outputs = (2,)

    print('Init agent...', end='')
    agent = JoystickAgent(inputs, outputs)
    print('Done')

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
