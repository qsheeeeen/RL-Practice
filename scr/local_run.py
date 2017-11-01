# coding: utf-8

# Think of car as local.

import zmq

from environment import RacingCar
from tools.communication import send_zipped_pickle, receive_zipped_pickle


def main():
    print('Init communication.')

    context = zmq.Context()

    result_sender = context.socket(zmq.PUSH)
    result_sender.connect("tcp://192.168.1.2:5558")

    order_receiver = context.socket(zmq.PULL)
    order_receiver.bind('tcp://*:5555')

    print('Init environment.')
    env = RacingCar()
    data = env.reset()

    print('Send data for first time...', end='')
    send_zipped_pickle(result_sender, data)
    print('Com estiblished.')

    while True:
        print('Wait for order...', end='')
        action = receive_zipped_pickle(order_receiver)
        print('Received.')

        data = env.step(action)

        print('Send result...', end='')
        send_zipped_pickle(result_sender, data)
        print('Done.')


if __name__ == '__main__':
    main()
