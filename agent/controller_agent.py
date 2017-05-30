#!/usr/local/bin/python3
# coding: utf-8

import numpy as np
import pygame


class ControllerAgnet(object):
    def __init__(self):
        self.__steering_axis = 4
        self.__gas_break_axis = 2

        pygame.init()
        pygame.joystick.init()

        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

    def act(self, ob, reward, done):
        steering = self.controller.get_axis(self.__steering_axis)
        gas_break = -self.controller.get_axis(self.__gas_break_axis)

        return np.array((steering, gas_break), dtype=np.float32)
