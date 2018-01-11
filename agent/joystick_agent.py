# coding: utf-8

import h5py
import numpy as np
import pygame

from .agent import Agent


class JoystickAgent(Agent):
    def __init__(self, state_shape, action_shape):
        """For testing and fun.

        TODO:
            Add button controlled shut down.

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

        self.file = h5py.File('./data.h5', 'w')

        self.state_data_set = self.file.require_dataset('state', state_shape, np.uint8, exact=True, chunks=True)
        self.action_data_set = self.file.require_dataset('action', action_shape, np.float32, exact=True, chunks=True)

        self.last_action = None
        self.last_station = None

    def act(self, observation, reward, done) -> np.ndarray:
        steering = self.controller.get_axis(self.steering_axis)
        gas_break = -self.controller.get_axis(self.gas_break_axis)

        return np.array((steering, gas_break), dtype=np.float32)

    def close(self):
        pygame.quit()
        self.file.close()
