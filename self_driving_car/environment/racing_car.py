# coding: utf-8

import os
import time

import numpy as np
import picamera
import pigpio


class RacingCar(object):
    def __init__(self, motor_limitation=0.6):
        assert motor_limitation <= 1, '"motor_limitation" should not be greater than 1.'

        # Pin configuration.
        self._STEERING_SERVO_PIN = 23
        self._MOTOR_PIN = 24
        self._LEFT_LINE_SENSOR_PIN = 17
        self._RIGHT_LINE_SENSOR_PIN = 18
        self._ENCODER_DIR_PIN = 21
        self._ENCODER_PUL_PIN = 20
        self._ENCODER_ZERO_PIN = 19
        self._FORWARD_PIN_LEVEL = 0

        # Hardware configuration.
        self._TIRE_DIAMETER = 0.05
        self._GEAR_RATIO = 145. / 35.
        self._ENCODER_LINE = 512
        self._SERVO_RANGE = 200
        self._MOTOR_RANGE = 500 * motor_limitation

        # Parameters.
        self._SPEED_UPDATE_INTERVAL = 500
        self._REWARD_UPDATE_INTERVAL = 500

        # Hardware controlling.
        os.system('sudo pigpiod')
        time.sleep(1)

        self._pi = pigpio.pi()
        self._pi.set_watchdog(self._ENCODER_PUL_PIN, self._SPEED_UPDATE_INTERVAL)
        self._pi.set_watchdog(self._ENCODER_ZERO_PIN, self._REWARD_UPDATE_INTERVAL)
        self._pi.callback(self._ENCODER_PUL_PIN, pigpio.RISING_EDGE, self._interrupt_handle)
        self._pi.callback(self._ENCODER_ZERO_PIN, pigpio.RISING_EDGE, self._interrupt_handle)
        self._pi.callback(self._LEFT_LINE_SENSOR_PIN, pigpio.FALLING_EDGE, self._interrupt_handle)
        self._pi.callback(self._RIGHT_LINE_SENSOR_PIN, pigpio.FALLING_EDGE, self._interrupt_handle)

        self._update_pwm(0, 0)

        # Camera.
        self._image_width, self._image_height = 240, 240
        self._image = np.empty((self._image_height, self._image_width, 3), dtype=np.uint8)

        self._cam = picamera.Picamera()
        self._cam.resolution = 1640, 1232
        self._cam.framerate = 30
        self._cam.exposure_mode = 'sport'
        self._cam.image_effect = 'denoise'
        self._cam.meter_mode = 'backlit'

        self._car_info = {
            'Steering signal': 0,
            'Motor signal': 0,
            'Reward': 0,
            'Speed': 0,
            'Done': False}

        self._encoder_pulse_count = 0

        self.observation_space = np.random.rand(self._image_height, self._image_width, 3).astype(np.float32)
        self.action_space = (np.random.rand(2) * 2 - 1).astype(np.float32)

    def reset(self):
        self._update_pwm(0, 0)
        self._update_image()

        self._car_info = {
            'Steering signal': 0,
            'Motor signal': 0,
            'Reward': 0,
            'Speed': 0,
            'Done': False}

        return self._image

    def step(self, action):
        """Perform action.

        Args:
            action: Anything with shape (2,).

                First number:
                left    <-->    right
                -1      <-->    1

                Second number:
                break   <-->    gas
                -1      <-->    1

        Raises:
            AssertionError: When input array is not in correct shape or value (eg. > 1).

        """
        assert len(action) == 2, 'Incorrect input shape.'
        action = np.minimum(np.maximum(action, -1), 1)

        if self._car_info['Done']:
            self._car_info['Steering signal'], self._car_info['Motor signal'] = 0, 0
        else:
            self._car_info['Steering signal'], self._car_info['Motor signal'] = action

        self._update_pwm(self._car_info['Steering signal'], self._car_info['Motor signal'])
        self._update_image()

        return self._image, self._car_info['Reward'], self._car_info['Done'], self._car_info

    def close(self):
        self._cam.close()
        self._update_pwm(0, 0)
        self._pi.stop()

    def _update_image(self):
        # TODO: Check image value.
        self._cam.capture(self._image, 'rgb', use_video_port=True, resize=(self._image_width, self._image_height))

    def _interrupt_handle(self, gpio, level, tick):
        if (gpio == self._LEFT_LINE_SENSOR_PIN) or (gpio == self._RIGHT_LINE_SENSOR_PIN):
            self._car_info['Done'] = True

        elif gpio == self._ENCODER_PUL_PIN:
            if level == pigpio.RISING_EDGE:
                self._encoder_pulse_count += 1

            if level == pigpio.TIMEOUT:
                s = self._encoder_pulse_count / self._ENCODER_LINE * np.pi * self._TIRE_DIAMETER
                t = self._SPEED_UPDATE_INTERVAL / 1000
                self._car_info['Speed'] = s / t * self._GEAR_RATIO
                self._encoder_pulse_count = 0

        elif gpio == self._ENCODER_ZERO_PIN:
            if level == pigpio.RISING_EDGE:
                self._car_info['Reward'] += 5

            if level == pigpio.TIMEOUT:
                self._car_info['Reward'] -= 1
                if self._car_info['Reward'] < 0:
                    self._car_info['Done'] = True

    def _update_pwm(self, steering_signal, motor_signal):

        def scale_range(old_value, old_min, old_max, new_min, new_max):
            return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

        pwm_wave = scale_range(steering_signal, -1, 1, (1500 - self._SERVO_RANGE), (1500 + self._SERVO_RANGE))
        self._pi.set_servo_pulsewidth(self._STEERING_SERVO_PIN, pwm_wave)

        pwm_wave = scale_range(motor_signal, -1, 1, (1500 - self._MOTOR_RANGE), (1500 + self._MOTOR_RANGE))
        self._pi.set_servo_pulsewidth(self._MOTOR_PIN, pwm_wave)
