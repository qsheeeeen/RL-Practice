class IMU(object):
    def __init__(self, interrupt_handle):
        # NOTE https://www.raspberrypi.org/documentation/configuration/uart.md
        # /dev/serial0
        pass

    def get_data(self):
        pass

    def _setup_interrupt(self):
        raise NotImplementedError
