# coding: utf-8

import blosc
import msgpack

def send_array(socket, array):
    """Pack the array and send it"""
    packed_array = blosc.pack_array(array)
    return socket.send(packed_array)


def receive_array(socket):
    """Receive the pack and unpack it."""
    packed_array = socket.recv()
    return blosc.unpack_array(packed_array)