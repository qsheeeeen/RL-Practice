# coding: utf-8

import zerorpc

from environment import RacingCar


def main():
    s = zerorpc.Server(RacingCar())
    s.bind("tcp://0.0.0.0:4242")
    s.run()


if __name__ == '__main__':
    main()
