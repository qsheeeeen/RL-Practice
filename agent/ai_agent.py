# coding: utf-8

import numpy as np
import pygame
from keras import Input

from .replay_buffer import ReplayBuffer


class AIAgent(object):
    def __init__(self, sample_state, sample_action, train=True):
        self.sample_state = sample_state
        self.sample_action = sample_action
        self.train = train

        self.STEERING_AXIS = 4
        self.GAS_BREAK_AXIS = 2

        # Init joystick.
        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find any joystick.'
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.replay_buffer = ReplayBuffer(10 ** 6, self.sample_state.shape, self.sample_action.shape)
        self.actor_model = self.get_actor_model(self.sample_state.shape)
        self.critic_model = self.get_critic_model(self.sample_action.shape)

    def act(self, observation, reward, done):
        steering = None
        gas_break = None

        # TODO: Add AI part

        controller_steering = self.joystick.get_axis(self.STEERING_AXIS)
        controller_gas_break = -self.joystick.get_axis(self.GAS_BREAK_AXIS)

        if controller_steering != 0.:
            steering = controller_steering

        if controller_gas_break != 0.:
            gas_break = controller_gas_break

        return np.array((steering, gas_break), dtype=np.float32)

    def get_actor_model(self, shape):
        print("Now we build the model")
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Acceleration = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Brake = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        V = merge([Steering, Acceleration, Brake], mode='concat')
        model = Model(input=S, output=V)
        print("We finished building the model")
        return model, model.trainable_weights, S

    def get_critic_model(self, shape):

        return model
