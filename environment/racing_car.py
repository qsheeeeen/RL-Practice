# coding: utf-8

import time

import numpy as np
import picamera
import pigpio


class RacingCar(object):
    def __init__(self, motor_limitation=0.6, reward_refresh_rate=20):
        # Controller setup.
        self.STEERING_SERVO_PIN = 4
        self.MOTOR_PIN = 17
        self.LEFT_LINE_SENSOR_PIN = 1
        self.RIGHT_LINE_SENSOR_PIN = 2
        self.ENCODER_DIR_PIN = 12
        self.ENCODER_ZERO_PIN = 13

        self.FORWARD_PIN_LEVEL = 1

        self.pi_car = pigpio.pi()
        self.pi_car.callback(self.LEFT_LINE_SENSOR_PIN, pigpio.RISING_EDGE, self.line_interrupt_handle)
        self.pi_car.callback(self.RIGHT_LINE_SENSOR_PIN, pigpio.RISING_EDGE, self.line_interrupt_handle)
        self.pi_car.callback(self.ENCODER_ZERO_PIN, pigpio.RISING_EDGE, self.update_reward)

        self.update_pwm(0, 0)

        # Camera setup.
        image_width, image_height = (320, 240)

        self.image = np.empty((image_height, image_width, 3), dtype=np.uint8)

        self.cam = picamera.Picamera()
        self.cam.resolution = image_width, image_height
        self.cam.framerate = 40
        self.cam.exposure_mode = 'sport'

        # Setup interacting environment.
        self.last_action_time = time.time()
        self.reward_time_interval = 1. / reward_refresh_rate
        self.reward = 0.
        self.done = False

        self.last_encoder_time = time.time()
        self.car_speed = 0.

        info_value = (self.reward, self.motor_signal, self.steering_signal, self.car_speed)
        info_name = ('Reward', 'Motor signal:', 'Steering signal:', 'Car speed:')
        self.info = dict(zip(info_name, info_value))

        self.action_list = (
            'Left & Right',
            'Gas & Break')

        self.motor_signal = 0.
        self.motor_limitation = motor_limitation
        self.steering_signal = 0.

        self.action_shape = len(self.action_list)
        self.state_shape = self.image.shape

    def reset(self) -> (np.ndarray, float, bool, tuple):
        self.reward = 0
        self.done = False
        self.motor_signal = 0
        self.steering_signal = 0
        self.car_speed = 0

        self.update_pwm(self.steering_signal, self.motor_signal)
        self.cam.capture(self.image, 'rgb', use_video_port=True)

        self.update_info()

        return self.image, self.reward, self.done, self.info

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, tuple):
        # Calculate reward.
        current_time = time.time()
        if current_time - self.last_action_time >= self.reward_time_interval:
            self.reward -= 1
        self.last_action_time = current_time

        if self.reward < 0:
            self.done = True

        if self.done:
            self.steering_signal, self.motor_signal = 0., 0.
        else:
            self.steering_signal, self.motor_signal = action

        # Perform action.
        self.motor_signal *= self.motor_limitation
        self.update_pwm(self.steering_signal, self.motor_signal)

        # Get state.
        self.cam.capture(self.image, 'rgb', use_video_port=True)
        self.update_info()

        return self.image, self.reward, self.done, self.info

    def close(self) -> None:
        self.update_info()
        self.cam.close()
        self.update_pwm(0, 0)
        self.pi_car.stop()

    def line_interrupt_handle(self) -> None:
        self.done = True

    def update_reward(self) -> None:
        tire_diameter = 0.05

        current_time = time.time()
        self.car_speed = 3.14 * tire_diameter / (current_time - self.last_encoder_time) * 1.5
        self.last_encoder_time = current_time

        self.reward += 5

    def update_pwm(self, steering_signal: float, motor_signal: float) -> None:
        real_pwm = self.scale_range(steering_signal, -1., 1., 1000., 2000.)
        self.pi_car.set_servo_pulsewidth(self.STEERING_SERVO_PIN, real_pwm)

        real_pwm = self.scale_range(motor_signal, -1., 1., 1000., 2000.)
        self.pi_car.set_servo_pulsewidth(self.MOTOR_PIN, real_pwm)

    def update_info(self) -> None:
        info_value = (self.reward, self.motor_signal, self.steering_signal, self.car_speed)
        info_name = ('Reward', 'Motor signal:', 'Steering signal:', 'Car speed:')
        self.info = dict(zip(info_name, info_value))

    @staticmethod
    def scale_range(old_value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
