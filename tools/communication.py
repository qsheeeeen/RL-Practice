# coding: utf-8

import time
from multiprocessing import Process

import blosc
import numpy as np
import zmq


def send_array(socket, array):
    """Pack the array and send it"""
    packed_array = blosc.pack_array(array)
    return socket.send(packed_array)


def receive_array(socket):
    """Receive the pack and unpack it."""
    packed_array = socket.recv()
    return blosc.unpack_array(packed_array)


def client_run():
    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    array = np.random.randint(0, 256, (1, 320, 640, 3), np.uint8)

    print('Send data for the first time...', end='\t')
    send_array(socket, array)
    print('Com success.')

    for i in range(20):
        start = time.time()

        print('Wait for action...', end='\t')
        b = receive_array(socket)
        print('Received.')

        print('')
        print(i)
        print('FPS:%f' % (1 / (time.time() - start)))
        print("Checking array...")
        print("Okay" if (array == b).all() else "Failed")

        array = np.random.randint(0, 256, (1, 320, 640, 3), np.uint8)

        print('Send result...', end='\t')
        send_array(socket, array)
        print('Done.')


def server_run():
    context = zmq.Context()

    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print('Wait for the first result...', end='\t')
    b = receive_array(socket)
    print('Com success.')

    for i in range(20):
        print('Send action...', end='\t')
        send_array(socket, b)
        print('Done.')

        print('Wait for result...', end='\t')
        b = receive_array(socket)
        print('Received.')


def main():
    # Test. Send data through different process.
    p1 = Process(target=client_run)
    p2 = Process(target=server_run)

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == '__main__':
    main()
