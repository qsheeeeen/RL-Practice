# coding: utf-8

import os
import time

import numpy as np
import picamera
import pigpio


class RacingCar(object):
    def __init__(self, motor_limitation=0.6, reward_refresh_rate=20):
        # Camera setup.
        image_width, image_height = (320, 240)

        self.__image = np.empty((image_height, image_width, 3), dtype=np.uint8)

        self.__cam = picamera.Picamera()
        self.__cam.resolution = image_width, image_height
        self.__cam.framerate = 40
        self.__cam.exposure_mode = 'sport'

        # Controller setup.
        self.__STEERING_SERVO_PIN = 4
        self.__MOTOR_PIN = 17
        self.__LEFT_LINE_SENSOR_PIN = 1
        self.__RIGHT_LINE_SENSOR_PIN = 2
        self.__ENCODER_DIR_PIN = 12
        self.__ENCODER_ZERO_PIN = 13

        self.__FORWARD_PIN_LEVEL = 1

        os.system('sudo pigpiod')
        time.sleep(1)
        self.__pi_car = pigpio.pi()

        self.__pi_car.callback(self.__LEFT_LINE_SENSOR_PIN, pigpio.RISING_EDGE, self.__line_interrupt_handle)
        self.__pi_car.callback(self.__RIGHT_LINE_SENSOR_PIN, pigpio.RISING_EDGE, self.__line_interrupt_handle)
        self.__pi_car.callback(self.__ENCODER_ZERO_PIN, pigpio.RISING_EDGE, self.__update_reward)

        self.__pi_car.set_servo_pulsewidth(self.__MOTOR_PIN, 0)
        self.__pi_car.set_servo_pulsewidth(self.__STEERING_SERVO_PIN, 0)

        # Setup interacting environment.
        self.__last_action_time = time.time()
        self.__reward_time_interval = 1. / reward_refresh_rate
        self.__reward = 0.
        self.__done = False

        self.__last_encoder_time = time.time()
        self.__car_speed = 0.
        self.__info = self.__motor_signal, self.__steering_signal, self.__car_speed

        self.__action_list = (
            'Left & Right',
            'Gas & Break')

        self.__motor_signal = 0.
        self.__motor_limitation = motor_limitation
        self.__steering_signal = 0.

        self.action_shape = len(self.__action_list)
        self.state_shape = self.__image.shape

    def reset(self) -> (np.ndarray, float, bool, tuple):
        self.__cam.capture(self.__image, 'rgb', use_video_port=True)
        self.__reward = 0
        self.__done = False
        self.__motor_signal = 0
        self.__steering_signal = 0
        self.__car_speed = 0

        self.__update_info()

        return self.__image, self.__reward, self.__done, self.__info

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, tuple):
        self.__motor_signal *= self.__motor_limitation

        self.__update_pwm(self.__steering_signal, self.__motor_signal)

        self.__cam.capture(self.__image, 'rgb', use_video_port=True)

        current_time = time.time()
        if current_time - self.__last_action_time >= self.__reward_time_interval:
            self.__reward -= 5
        self.__last_action_time = current_time

        if self.__reward < 0:
            self.__done = True

        if self.__done:
            self.__steering_signal, self.__motor_signal = 0., 0.
        else:
            self.__steering_signal, self.__motor_signal = action

        self.__update_info()

        return self.__image, self.__reward, self.__done, self.__info

    def close(self) -> None:
        self.__update_info()
        self.__cam.close()
        self.__update_pwm(0, 0)
        self.__pi_car.stop()

    def __line_interrupt_handle(self) -> None:
        self.__done = True

    def __update_reward(self) -> None:
        tire_diameter = 0.05

        current_time = time.time()
        self.__car_speed = 3.14 * 2 * tire_diameter / (current_time - self.__last_encoder_time) / 1.5
        self.__last_encoder_time = current_time

        self.__reward += 5

    def __update_pwm(self, steering_signal: float, motor_signal: float) -> None:
        real_pwm = self.__scale_range(steering_signal, -1., 1., 1000., 2000.)
        self.__pi_car.set_servo_pulsewidth(self.__STEERING_SERVO_PIN, real_pwm)

        real_pwm = self.__scale_range(motor_signal, -1., 1., 1000., 2000.)
        self.__pi_car.set_servo_pulsewidth(self.__MOTOR_PIN, real_pwm)

    def __update_info(self) -> None:
        __info_value = self.__reward, self.__motor_signal, self.__steering_signal, self.__car_speed
        info_name = ('Reward', 'Motor signal:', 'Steering signal:', 'Car speed:')
        self.__info = dict(zip(info_name, __info_value))

    @staticmethod
    def __scale_range(old_value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
