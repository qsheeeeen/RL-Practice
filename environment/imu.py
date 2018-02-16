# coding: utf-8

import pigpio


class IMU(object):
    def __init__(self, interrupt_handle):
        # TODO
        pass

    def get_data(self):
        pass

    def _setup_interrupt(self):
        raise NotImplementedError
