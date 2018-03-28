import h5py
import numpy as np
import pygame


class JoystickAgent(object):
    def __init__(self, input_shape, output_shape, data_path='./data.hdf5', num_sample=10000):
        self._STEERING_AXIS = 0
        self._BREAK_AXIS = 2
        self._GAS_AXIS = 5
        self._DONE_BUTTON = 3

        self._output_shape = output_shape

        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find a controller.'

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._file = h5py.File(data_path, 'w')

        self._state_dataset = self._file.create_dataset('states',
                                                        (num_sample,) + input_shape,
                                                        np.float32,
                                                        chunks=(1,) + input_shape)

        self._action_dataset = self._file.create_dataset('actions',
                                                         (num_sample,) + output_shape,
                                                         np.float32,
                                                         chunks=(1,) + output_shape)

        self._reward_dataset = self._file.create_dataset('rewards',
                                                         (num_sample, 1),
                                                         np.float32,
                                                         chunks=(1, 1))
        self._num_sample = num_sample

        self._count = 0

        print('Begin init.')
        print('Push button...')

        while True:
            pygame.event.get()
            sign = self._joystick.get_axis(self._BREAK_AXIS)
            print(sign, end='\r')
            if sign > 0.9:
                print('OK')
                break

    def act(self, state, reward=0, done=False):
        pygame.event.get()

        steering = self._joystick.get_axis(self._STEERING_AXIS)

        gas_signal = self._joystick.get_axis(self._GAS_AXIS)
        gas_signal = (gas_signal + 1) / 2

        break_signal = self._joystick.get_axis(self._BREAK_AXIS)
        break_signal = (break_signal + 1) / 2

        if self._output_shape[0] == 3:
            action_array = np.array((steering, gas_signal, break_signal), dtype=np.float32)
        elif self._output_shape[0] == 2:
            gas_break_signal = gas_signal + break_signal
            action_array = np.array((steering, gas_break_signal), dtype=np.float32)
        else:
            raise NotImplementedError

        self._state_dataset[self._count] = state
        self._action_dataset[self._count] = action_array
        self._reward_dataset[self._count] = reward

        self._count += 1
        if self._num_sample == self._count:
            self.close()

        return action_array

    def close(self):
        self._file.close()
        pygame.quit()
