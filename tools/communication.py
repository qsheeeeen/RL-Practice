# coding: utf-8

import pickle
import time
import zlib

import numpy as np
import zmq


def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)


def receive_zipped_pickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)


def main():
    context = zmq.Context()

    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:5555")

    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5555")

    array = np.ones((320, 640, 3), np.float32)

    start = time.time()

    send_zipped_pickle(sender, array)

    B = receive_zipped_pickle(receiver)

    print(time.time() - start)
    print("Checking zipped pickle...")
    print("Okay" if (array == B).all() else "Failed")


if __name__ == '__main__':
    main()
