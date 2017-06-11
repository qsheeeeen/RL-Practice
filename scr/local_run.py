# coding: utf-8

# Think of car as local.

import zmq

from environment import RacingCar
from tools.communication import send_zipped_pickle, receive_zipped_pickle


def main():
    env = RacingCar()

    context = zmq.Context()

    data_sender = context.socket(zmq.PUSH)
    data_sender.connect("tcp://192.168.1.2:5558")

    result_receiver = context.socket(zmq.PULL)
    result_receiver.connect("tcp://192.168.1.2:5555")

    data = env.reset()

    send_zipped_pickle(data_sender, data)

    while True:
        action = receive_zipped_pickle(result_receiver)

        data = env.step(action)

        send_zipped_pickle(data_sender, data)


if __name__ == '__main__':
    main()
