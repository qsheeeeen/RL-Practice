# coding: utf-8

import numpy as np
import pygame
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.merge import Add
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Sequential

from .replay_buffer import ReplayBuffer


class AIAgent(object):
    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.__steering_axis = 4
        self.__gas_break_axis = 2

        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find controller.'
        self.__controller = pygame.joystick.Joystick(0)
        self.__controller.init()

        self.replay_buffer = ReplayBuffer(10 ** 6)

        self.__actor_model = self.__get_actor_model()
        self.__critic_model = self.__get_critic_model()

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

    def __get_actor_model(self):
        model = Sequential()
        model.add(Conv2D(32, 3, padding='same', activation='relu', bias_initializer='glorot_normal'))
        model.add(Conv2D(64, 3, padding='same', activation='relu', bias_initializer='glorot_normal'))
        model.add(Conv2D(64, 3, padding='same', activation='relu', bias_initializer='glorot_normal'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu', bias_initializer='glorot_normal'))
        model.add(Dense(128, activation='relu', bias_initializer='glorot_normal'))
        model.add(Dense(2, activation='tanh', bias_initializer='glorot_normal'))
        # TODO learning rate 10 ** -4
        model.compile(optimizer='adam', loss='mse')

        return model

    def __get_critic_model(self):
        # TODO learning rate 10 ** -3
        state_input = Input(shape=self.state_shape)
        action_input = Input(shape=self.state_shape)

        added = Add()([state_input, action_input])

        input1 = Input(shape=(16,))
        x1 = Dense(8, activation='relu')(input1)
        input2 = Input(shape=(32,))
        x2 = Dense(8, activation='relu')(input2)
        added = Add()([x1, x2])

        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[state_input, action_input], outputs=out)

        model = Sequential()
        model.add(Conv2D(32, 3, padding='same', activation='relu'))
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='tanh'))
        model.compile(optimizer='adam', loss='mse')

        return model
