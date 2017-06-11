# coding: utf-8

import os
import time

import numpy as np
import picamera
import pigpio


class RacingCar(object):
    def __init__(self):
        self.___spoiler_servo_pin = 5
        self.__steering_servo_pin = 4
        self.__motor_pin = 17
        self.__left_line_sensor_pin = 1
        self.__right_line_sensor_pin = 2
        self.__encoder_dir_pin = 12
        self.__encoder_zero_pin = 13

        self.__forward_pin_level = 1

        image_width, image_height = (320, 240)

        self.image = np.empty((image_height, image_width, 3), dtype=np.uint8)

        self.__cam = picamera.Picamera()
        self.__cam.resolution = image_width, image_height
        self.__cam.framerate = 40
        self.__cam.exposure_mode = 'sport'

        os.system('sudo pigpiod')
        time.sleep(1)
        self.__pi_car = pigpio.pi()

        self.__pi_car.callback(self.__left_line_sensor_pin, pigpio.RISING_EDGE, self.__line_interrupt_handle)
        self.__pi_car.callback(self.__right_line_sensor_pin, pigpio.RISING_EDGE, self.__line_interrupt_handle)
        self.__pi_car.callback(self.__encoder_zero_pin, pigpio.RISING_EDGE, self.__update_reward)

        self.__pi_car.set_servo_pulsewidth(self.__motor_pin, 0)
        self.__pi_car.set_servo_pulsewidth(self.__steering_servo_pin, 0)
        self.__pi_car.set_servo_pulsewidth(self.___spoiler_servo_pin, 0)

        self.__motor = 0.
        self.__steering_angle = 0.

        self.__reward = 0.
        self.__done = False
        self.__info = (self.__motor, self.__steering_angle)

        self.action_list = (
            'Left & Right',
            'Gas & Break'
        )

        self.action_space = np.zeros(len(self.action_space), np.float32)

    def reset(self) -> (np.ndarray, float, bool, list):
        self.__get_image()
        self.__reward = 0
        self.__done = False

        return self.image, self.__reward, self.__done, self.__info

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, list):
        self.__steering_angle = action[0]
        self.__motor = action[1]

        real_pwm = self.scale_range(self.__motor, -1., 1., 1000., 2000.)
        self.__pi_car.set_servo_pulsewidth(self.__motor_pin, real_pwm)
        self.__pi_car.set_servo_pulsewidth(self.___spoiler_servo_pin, real_pwm)

        real_pwm = self.scale_range(self.__steering_angle, -1., 1., 1000., 2000.)
        self.__pi_car.set_servo_pulsewidth(self.__steering_servo_pin, real_pwm)

        self.__get_image()

        return self.image, self.__reward, self.__done, self.__info

    def close(self) -> None:
        self.__cam.close()

        self.__pi_car.set_servo_pulsewidth(self.__motor, 0)
        self.__pi_car.set_servo_pulsewidth(self.__steering_servo_pin, 0)
        self.__pi_car.stop()

    def __get_image(self) -> None:
        self.__cam.capture(self.image, 'rgb', use_video_port=True)

    def __line_interrupt_handle(self) -> None:
        self.__done = True

    def __update_reward(self):
        self.__reward += 5

    @staticmethod
    def scale_range(old_value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
