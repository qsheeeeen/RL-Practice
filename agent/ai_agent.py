# coding: utf-8

import numpy as np
import pygame


class AIAgent(object):
    def __init__(self):
        self.__steering_axis = 4
        self.__gas_break_axis = 2

        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find controller.'
        self.__controller = pygame.joystick.Joystick(0)
        self.__controller.init()

    def act(self, observation, reward, done):
        steering = None
        gas_break = None

        # TODO: Add AI part

        controller_steering = self.__controller.get_axis(self.__steering_axis)
        controller_gas_break = -self.__controller.get_axis(self.__gas_break_axis)

        if controller_steering != 0.:
            steering = controller_steering

        if controller_gas_break != 0.:
            gas_break = controller_gas_break

        return np.array((steering, gas_break), dtype=np.float32)
