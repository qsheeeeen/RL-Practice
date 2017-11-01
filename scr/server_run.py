# coding: utf-8

# Think of server as remote.

import zmq

from agent import ControllerAgent
from tools.communication import send_zipped_pickle, receive_zipped_pickle
from tools.dashboard import Dashboard


def main():
    print('Init communication.')

    context = zmq.Context()

    result_receiver = context.socket(zmq.PULL)
    result_receiver.bind("tcp://*:5558")

    order_sender = context.socket(zmq.PUSH)
    order_sender.connect("tcp://192.168.1.2:5555")

    print('Init agent.')
    dashboard = Dashboard()
    agent = ControllerAgent()

    while True:
        print('Wait for result...', end='')
        data = receive_zipped_pickle(result_receiver)
        print('Received.')

        ob, reward, done, info = data

        result = agent.act(ob, reward, done)

        print('Send .order..', end='')
        send_zipped_pickle(order_sender, result)
        print('Done.')

        dashboard.update(ob, (reward,) + info)


if __name__ == '__main__':
    main()
