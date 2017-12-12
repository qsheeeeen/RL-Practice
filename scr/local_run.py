# coding: utf-8

# Think car as local.

import zmq

from environment import RacingCar
from tools.communication import send_array, receive_array


def main():
    print('Init communication.')

    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.2:5555")

    print('Init environment.')
    env = RacingCar()
    data = env.reset()

    print('Send data for first time...', end='\t')
    send_array(socket, data)
    print('Com success.')

    while True:
        print('Wait for action...', end='\t')
        action = receive_array(socket)
        print('Received.')

        data = env.step(action)

        print('Send result...', end='\t')
        send_array(socket, data)
        print('Done.')


if __name__ == '__main__':
    main()
