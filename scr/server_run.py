# coding: utf-8

# Think server as remote.

import zmq

from agent import ControllerAgent
from tools.communication import send_array, receive_array
from tools.dashboard import Dashboard


def main():
    print('Init communication.')

    context = zmq.Context()

    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print('Init agent.')
    dashboard = Dashboard()
    agent = ControllerAgent()

    while True:
        print('Wait for result...', end='\t')
        data = receive_array(socket)
        print('Received.')

        ob, reward, done, info = data

        action = agent.act(ob, reward, done)

        print('Send action...', end='\t')
        send_array(socket, action)
        print('Done.')

        dashboard.update(ob, info)


if __name__ == '__main__':
    main()
