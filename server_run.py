# coding: utf-8

import numpy as np
from zmq import Context, REP

from agent import JoystickAgent
from tools import Dashboard
from tools import send_array, receive_array


def server_run():
    print('Run as server.')

    port = str(input('Enter the server port (default: 5555)...'))

    if not port:
        port = '5555'

    print('Init Dashboard.')
    dashboard = Dashboard()

    print('Init communication.')
    context = Context()
    socket = context.socket(REP)
    socket.bind('tcp://*:' + port)

    print('Wait for the first result...')
    data = receive_array(socket)
    print('Com success.')

    print('Init Agent')  # TODO
    sample_state = np.random.randint(0, 256, (320, 240, 3), dtype=np.uint8)
    sample_action = np.random.random_sample(2)

    agent = JoystickAgent(sample_state, sample_action)

    input('Press "Enter" to start...')

    close = False
    while not close:
        ob, reward, done, info = data

        close = dashboard.update(ob, info)

        action = agent.act(ob, reward, done)

        print('Send action...', end='\t')
        send_array(socket, action)
        print('Done.')

        print('Wait for result...', end='\t')
        data = receive_array(socket)
        print('Received.')

    agent.close()


if __name__ == '__main__':
    server_run()
