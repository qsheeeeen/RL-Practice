# coding: utf-8


import time

import blosc
import numpy as np
import zmq

# TODO: Compress and send.
def send_data(socket, *inputs):
    """Pack the array and send it"""
    if len(inputs) == 1:
        packed_array = blosc.pack_array(inputs[0])
        return socket.send((packed_array,))

    elif len(inputs) == 4:
        packed_array = blosc.pack_array(inputs[0])

        return socket.send((packed_array,) + (inputs[1:]))


def receive_data(socket):
    """Receive the pack and unpack it."""
    data = socket.recv()
    if len(data) == 1:
        array = blosc.unpack_array(data[0])
        return array

    elif len(data) == 4:
        array = blosc.unpack_array(data[0])

        return (array,) + (data[1:])


def client_run():
    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    array = np.random.rand(1, 240, 320, 3).astype(np.float32)

    print('Send data for the first time...', end='\t')
    send_data(socket, array, np.pi, np.pi, np.pi)
    print('Com success.')

    for i in range(20):
        start = time.time()

        print('Wait for action...', end='\t')
        b = receive_data(socket)
        print('Received.')

        print()
        print(i)
        print('Time used: %f' % (time.time() - start))
        print("Checking array...")
        print("Okay" if (array == b).all() else "Failed")

        array = np.random.rand(1, 240, 320, 3).astype(np.float32)

        data = (array, 1, 1, 0)

        print('Send result...', end='\t')
        send_data(socket, array, np.pi, np.pi, np.pi)
        print('Done.')


def server_run():
    context = zmq.Context()

    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print('Wait for the first result...', end='\t')
    b = receive_data(socket)
    print('Com success.')

    for i in range(20):
        print('Send action...', end='\t')
        send_data(socket, b)
        print('Done.')

        print('Wait for result...', end='\t')
        b = receive_data(socket)
        print('Received.')


def main():
    from multiprocessing import Process

    # Test. Send data through different process.
    p1 = Process(target=client_run)
    p2 = Process(target=server_run)

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == '__main__':
    main()
