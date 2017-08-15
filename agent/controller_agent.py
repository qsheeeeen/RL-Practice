# coding: utf-8

import numpy as np
import pygame


class ControllerAgent(object):
    """For testing and fun.
    """
    def __init__(self):
        self.__steering_axis = 4
        self.__gas_break_axis = 2

        pygame.init()
        pygame.joystick.init()

        self.__controller = pygame.joystick.Joystick(0)
        self.__controller.init()

    def act(self, observation, reward, done) -> np.ndarray:
        steering = self.__controller.get_axis(self.__steering_axis)
        gas_break = -self.__controller.get_axis(self.__gas_break_axis)

        return np.array((steering, gas_break), dtype=np.float32)
