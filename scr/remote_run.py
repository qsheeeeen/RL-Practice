# coding: utf-8

# Think of server as remote.

import numpy as np
import zmq

from tools.communication import SerializingContext


def main():
    ctx = SerializingContext()
    req = ctx.socket(zmq.REQ)
    rep = ctx.socket(zmq.REP)

    rep.bind('inproc://a')
    req.connect('inproc://a')
    A = np.ones((1024, 1024))
    print("Array is %i bytes" % (A.nbytes))

    # send/recv with pickle+zip
    req.send_zipped_pickle(A)
    B = rep.recv_zipped_pickle()
    # now try non-copying version
    rep.send_array(A, copy=False)
    C = req.recv_array(copy=False)
    print("Checking zipped pickle...")
    print("Okay" if (A == B).all() else "Failed")
    print("Checking send_array...")
    print("Okay" if (C == B).all() else "Failed")


if __name__ == '__main__':
    main()
