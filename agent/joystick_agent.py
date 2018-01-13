# coding: utf-8

import pygame
from h5py import File
from numpy import array, float32

from .core import Agent


class JoystickAgent(Agent):
    def __init__(self, state_shape, action_shape):
        """For testing and fun.

        TODO:
            Add button controlled shut down.
            Add button controlled begin record.

        Args:
            state_shape (tuple):
            action_shape (tuple):

        Raises:
            AssertionError: Can not find any joystick.

        """
        self.steering_axis = 4
        self.gas_break_axis = 2

        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find controller.'
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        self.file = File('./data.h5', 'w')

        self.state_data_set = self.file.require_dataset('state', state_shape, float32, exact=True, chunks=True)
        self.action_data_set = self.file.require_dataset('action', action_shape, float32, exact=True, chunks=True)

        self.last_action = None
        self.last_station = None

    def act(self, observation, reward, done):
        steering = self.controller.get_axis(self.steering_axis)
        gas_break = -self.controller.get_axis(self.gas_break_axis)

        return array((steering, gas_break), dtype=float32)

    def close(self):
        pygame.quit()
        self.file.close()
