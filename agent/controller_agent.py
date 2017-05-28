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
        val_steering = self.controller.get_axis(self.__steering_axis)
        val_gas_break = self.controller.get_axis(self.__gas_break_axis)

        if val_gas_break > 0:
            val_gas = val_gas_break
            val_break = 0

        else:
            val_gas = 0
            val_break = -val_gas_break

        return np.array((val_steering, val_gas, val_break), dtype=np.float32)
