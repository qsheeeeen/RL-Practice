# coding: utf-8

# Think of server as remote.

import zmq

from agent import ControllerAgent
from tools.communication import send_zipped_pickle, receive_zipped_pickle


def main():
    context = zmq.Context()

    data_receiver = context.socket(zmq.PULL)
    data_receiver.bind("tcp://*:5555")

    result_sender = context.socket(zmq.PUSH)
    result_sender.bind("tcp://*:5558")

    agent = ControllerAgent()

    while True:
        data = receive_zipped_pickle(data_receiver)

        ob, reward, done, info = data
        print('Data:\t{}\t{}\t{}'.format(reward, done, info))

        result = agent.act(ob, reward, done)

        send_zipped_pickle(result_sender, result)


if __name__ == '__main__':
    main()
