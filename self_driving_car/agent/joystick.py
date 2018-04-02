import h5py
import numpy as np
import pygame


class JoystickAgent(object):
    def __init__(self, input_shape, output_shape, data_path='./data.hdf5', num_sample=10000):
        self.STEERING_AXIS = 0
        self.BREAK_AXIS = 2
        self.GAS_AXIS = 5
        self.START_BUTTON = 8

        self.output_shape = output_shape

        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find a controller.'

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.file = h5py.File(data_path, 'w')

        self.state_dataset = self.file.create_dataset('states',
                                                      (num_sample,) + input_shape,
                                                      np.float32,
                                                      chunks=(1,) + input_shape)

        self.action_dataset = self.file.create_dataset('actions',
                                                       (num_sample,) + output_shape,
                                                       np.float32,
                                                       chunks=(1,) + output_shape)

        self.reward_dataset = self.file.create_dataset('rewards',
                                                       (num_sample, 1),
                                                       np.float32,
                                                       chunks=(1, 1))
        self.num_sample = num_sample

        self.count = 0

        print('Push break...')
        while True:
            pygame.event.get()
            sign = self.joystick.get_axis(self.BREAK_AXIS)
            print(sign, end='\r')
            if sign > 0.9:
                print('OK. Now Release.')
                break

        print('Push gas...')
        while True:
            pygame.event.get()
            sign = self.joystick.get_axis(self.GAS_AXIS)
            print(sign, end='\r')
            if sign > 0.9:
                print('OK. Now Release.')
                break
        while True:
            pygame.event.get()
            if self.joystick.get_button(self.START_BUTTON) == 1:
                print('Start.')
                break

    def act(self, state, reward=0, done=False):
        pygame.event.get()

        steering = self.joystick.get_axis(self.STEERING_AXIS)

        gas_signal = self.joystick.get_axis(self.GAS_AXIS)
        gas_signal = (gas_signal + 1) / 2

        break_signal = self.joystick.get_axis(self.BREAK_AXIS)
        break_signal = -(break_signal + 1) / 2

        if self.output_shape[0] == 3:
            action_array = np.array((steering, gas_signal, break_signal), dtype=np.float32)
        elif self.output_shape[0] == 2:
            gas_break_signal = gas_signal + break_signal
            action_array = np.array((steering, gas_break_signal), dtype=np.float32)
        else:
            raise NotImplementedError

        self.state_dataset[self.count] = state
        self.action_dataset[self.count] = action_array
        self.reward_dataset[self.count] = reward

        self.count += 1
        if self.num_sample == self.count:
            self.close()

        return action_array

    def close(self):
        self.file.close()
        pygame.quit()
