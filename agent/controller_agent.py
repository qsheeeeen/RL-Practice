# coding: utf-8

import numpy as np
import pygame


class ControllerAgent(object):
    """For testing and fun.
    """

    def __init__(self):
        self.steering_axis = 4
        self.gas_break_axis = 2

        pygame.init()
        pygame.joystick.init()

        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

    def act(self, observation, reward, done) -> np.ndarray:
        steering = self.controller.get_axis(self.steering_axis)
        gas_break = -self.controller.get_axis(self.gas_break_axis)

        return np.array((steering, gas_break), dtype=np.float32)
