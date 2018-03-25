import h5py
import numpy as np
import pygame


class JoystickAgent(object):
    def __init__(self, input_shape, output_shape, data_path='./data.hdf5', num_sample=10000):
        self._STEERING_AXIS = 0
        self._GAS_BREAK_AXIS = 2
        self._DONE_BUTTON = 3

        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find controller.'

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._file = h5py.File(data_path, 'w')

        self._state_dataset = self._file.create_dataset('state',
                                                        (num_sample,) + input_shape,
                                                        np.float32,
                                                        chunks=(1,) + input_shape)

        self._action_dataset = self._file.create_dataset('action',
                                                         (num_sample,) + output_shape,
                                                         np.float32,
                                                         chunks=(1,) + output_shape)

        self._reward_dataset = self._file.create_dataset('reward', (num_sample, 1), np.float32, chunks=(1, 1))

        self._count = 0

        while True:
            pygame.event.get()
            gas_break_signal = self._joystick.get_axis(self._GAS_BREAK_AXIS)
            print(gas_break_signal)

            if gas_break_signal == 1:
                break

    def act(self, state, reward=0, done=False):
        pygame.event.get()

        steering = self._joystick.get_axis(self._STEERING_AXIS)

        gas_break_signal = self._joystick.get_axis(self._GAS_BREAK_AXIS)

        gas_break_signal = (gas_break_signal - 1) / -2

        action_array = np.array((steering, gas_break_signal), dtype=np.float32)

        self._state_dataset[self._count] = state
        self._action_dataset[self._count] = action_array

        self._count += 1

        return action_array

    def close(self):
        pygame.quit()
        self._file.close()
