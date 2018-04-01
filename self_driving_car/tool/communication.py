import blosc
import msgpack
import zmq


class Com(object):
    def __init__(self, mode='server', ip='192.168.1.100', port='6565'):
        assert mode in ['server', 'client'], 'Unknown mode: "{}". Only support "server" or "client".'.format(mode)

        context = zmq.Context()

        if mode == 'server':
            self.socket = context.socket(zmq.REP)
            self.socket.bind('tcp://*:' + port)

        elif mode == 'client':
            self.socket = context.socket(zmq.REQ)
            self.socket.connect('tcp://' + ip + ':' + port)

    def send_data(self, data, clevel=9, shuffle=blosc.SHUFFLE, flags=0, copy=True, track=False):
        data = data.copy()
        data[0] = blosc.pack_array(data[0], clevel, shuffle)
        msg = msgpack.packb(data, use_bin_type=True)
        return self.socket.send(msg, flags, copy, track)

    def receive_data(self, flags=0, copy=True, track=False):
        msg = self.socket.recv(flags=flags, copy=copy, track=track)
        data = msgpack.unpackb(msg)
        data[0] = blosc.unpack_array(data[0])
        return data
