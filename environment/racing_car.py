# coding: utf-8

import numpy as np
import picamera
import pigpio


class RacingCar(object):
    def __init__(self, motor_limitation=0.6):
        self.STEERING_SERVO_PIN = 4
        self.MOTOR_PIN = 17
        self.LEFT_LINE_SENSOR_PIN = 1
        self.RIGHT_LINE_SENSOR_PIN = 2
        self.ENCODER_DIR_PIN = 12
        self.ENCODER_PUL_PIN = 12
        self.ENCODER_ZERO_PIN = 13
        self.FORWARD_PIN_LEVEL = 1

        self.SPEED_UPDATE_INTERVAL = 500
        self.REWARD_UPDATE_INTERVAL = 500

        self.encoder_pulse_count = 0

        self.pi_car = pigpio.pi()
        self.pi_car.set_watchdog(self.ENCODER_PUL_PIN, self.SPEED_UPDATE_INTERVAL)
        self.pi_car.set_watchdog(self.ENCODER_ZERO_PIN, self.REWARD_UPDATE_INTERVAL)
        # TODO: RISING_EDGE or FALLING_EDGE
        self.pi_car.callback(self.LEFT_LINE_SENSOR_PIN, pigpio.RISING_EDGE, self.interrupt_handle)
        self.pi_car.callback(self.RIGHT_LINE_SENSOR_PIN, pigpio.RISING_EDGE, self.interrupt_handle)
        self.pi_car.callback(self.ENCODER_ZERO_PIN, pigpio.RISING_EDGE, self.interrupt_handle)

        self.update_pwm(0, 0)

        image_width, image_height = (320, 240)

        self.image = np.empty((image_height, image_width, 3), dtype=np.uint8)

        self.cam = picamera.Picamera()
        self.cam.resolution = image_width, image_height
        self.cam.framerate = 40
        self.cam.exposure_mode = 'sport'

        self.car_info = {
            'Steering signal': 0,
            'Motor signal': 0,
            'Reward': 0,
            'Car speed': 0,
            'Done': False}

        self.motor_limitation = motor_limitation

        self.action_list = (
            'Left & Right',
            'Gas & Break')

        self.action_shape = len(self.action_list)
        self.state_shape = self.image.shape

    def reset(self):
        self.update_pwm(0, 0)
        self.cam.capture(self.image, 'rgb', use_video_port=True)

        self.car_info = {
            'Steering signal': 0,
            'Motor signal': 0,
            'Reward': 0,
            'Car speed': 0,
            'Done': False}

        return self.image, self.car_info['Reward'], self.car_info['Done'], self.car_info

    def step(self, action):
        if self.car_info['Done']:
            steering_signal, motor_signal = 0, 0
        else:
            steering_signal, motor_signal = action

        self.car_info['Steering signal'] = steering_signal
        self.car_info['Motor signal'] = motor_signal

        motor_signal *= self.motor_limitation
        self.update_pwm(steering_signal, motor_signal)

        self.cam.capture(self.image, 'rgb', use_video_port=True)

        return self.image, self.car_info['Reward'], self.car_info['Done'], self.car_info

    def close(self):
        self.cam.close()
        self.update_pwm(0, 0)
        self.pi_car.stop()

    def interrupt_handle(self, gpio, level, tick):
        tire_diameter = 0.05
        gear_ratio = 1.5
        encoder_line = 512

        if (gpio == self.LEFT_LINE_SENSOR_PIN) or (gpio == self.RIGHT_LINE_SENSOR_PIN):
            self.car_info['Done'] = True

        elif gpio == self.ENCODER_PUL_PIN:
            if level == pigpio.RISING_EDGE:
                self.encoder_pulse_count += 1

            if level == pigpio.TIMEOUT:
                s = self.encoder_pulse_count / encoder_line * 3.14 * tire_diameter
                t = self.SPEED_UPDATE_INTERVAL / 1000
                self.car_info['Car speed'] = s / t * gear_ratio
                self.encoder_pulse_count = 0

        elif gpio == self.ENCODER_ZERO_PIN:
            if level == pigpio.RISING_EDGE:
                self.car_info['Reward'] += 5

            if level == pigpio.TIMEOUT:
                self.car_info['Reward'] -= 1
                if self.car_info['Reward'] < 0:
                    self.car_info['Done'] = True

    def update_pwm(self, steering_signal, motor_signal):
        pwm_wave = self.scale_range(steering_signal, -1, 1, 1000, 2000)
        self.pi_car.set_servo_pulsewidth(self.STEERING_SERVO_PIN, pwm_wave)

        pwm_wave = self.scale_range(motor_signal, -1, 1, 1000, 2000)
        self.pi_car.set_servo_pulsewidth(self.MOTOR_PIN, pwm_wave)

    @staticmethod
    def scale_range(old_value, old_min, old_max, new_min, new_max):
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
