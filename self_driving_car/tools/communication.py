# coding: utf-8

import blosc


def send_array(socket, array):
    packed_array = blosc.pack_array(array)
    return socket.send(packed_array)


def receive_array(socket):
    packed_array = socket.recv()
    return blosc.unpack_array(packed_array)


def send_data(socket, data):
    pass


def receive_data(socket):
    pass
