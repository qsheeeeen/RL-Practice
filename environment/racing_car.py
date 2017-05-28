#!/usr/local/bin/python3
# coding: utf-8

import os
import time

import numpy as np
import picamera
import pigpio


class RacingCar(object):
    def __init__(self, use_controller=False, use_rc_motor=False):
        self.__use_controller = use_controller
        self.__use_rc_motor = use_rc_motor

        self.__servo_pin = 6
        self.__motor_forward_pin = 12
        self.__motor_backward_pin = 13
        self.__left_line_sensor_pin = 1
        self.__right_line_sensor_pin = 2
        self.__encoder_dir_pin = 5

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

        if self.__use_rc_motor:
            self.__pi_car.set_servo_pulsewidth(self.__motor_forward_pin, 0)
        else:
            self.__pi_car.set_PWM_frequency(self.__motor_forward_pin, 10000)
            self.__pi_car.set_PWM_frequency(self.__motor_backward_pin, 10000)

            self.__pi_car.set_PWM_dutycycle(self.__motor_forward_pin, 0)
            self.__pi_car.set_PWM_dutycycle(self.__motor_backward_pin, 0)

        self.__pi_car.set_servo_pulsewidth(self.__servo_pin, 0)

        self.__reward = 0.
        self.__done = False
        self.__info = [''] * 10

        self.__gas_duty_cycle = 0.
        self.__break_duty_cycle = 0.
        self.__steering = 0.

        if self.__use_controller:
            self.action_list = (
                'Gas ',
                'Break',
                'Left & Right'
            )
            self.action_space = (2,)
        else:
            self.action_list = (
                'None',
                'Gas',
                'Break',
                'Turn left',
                'Turn right',
                'Gas & Turn left',
                'Gas & Turn right',
                'Break & Turn left',
                'Break & Turn right'
            )

            self.action_space = len(self.action_list)

    def reset(self) -> (np.ndarray, int, bool, list):
        """Reset environment

        Returns: state, reward, done, info.

        """
        self.__get_image()
        self.__reward.value = 0
        self.__done.value = False
        self.__info.value = (self.__gas_duty_cycle, self.__break_duty_cycle, self.__steering)

        return self.image, self.__reward, self.__done, self.__info

    def step(self, action: np.ndarray) -> (np.ndarray, int, bool, list):
        """

        Args:
            action: Actions to perform.

        Returns: state, reward, done, info.

        """
        if self.__use_controller:
            assert action.shape == (3,), 'Action must contains 3 number.'
            self.__steering = -action[0]
            self.__gas_duty_cycle = action[1]
            self.__break_duty_cycle = action[2]

        else:
            index = np.argmax(action)
            print('Perform action:' + self.action_list[index])
            if index == 0:
                pass

            elif index == 1:
                self.__gas_duty_cycle += 0.1
                self.__break_duty_cycle = 0.

            elif index == 2:
                self.__gas_duty_cycle = 0.
                self.__break_duty_cycle += 0.1

            elif index == 3:
                self.__steering -= 0.1

            elif index == 4:
                self.__steering += 0.1

            if self.__gas_duty_cycle > 1.:
                self.__gas_duty_cycle = 1.

            elif self.__gas_duty_cycle < 0.:
                self.__gas_duty_cycle = 0.

            if self.__break_duty_cycle > 1.:
                self.__break_duty_cycle = 1.

            elif self.__break_duty_cycle < 0.:
                self.__break_duty_cycle = 0.

            if self.__steering > 1.:
                self.__steering = 1.

            elif self.__steering < -1.:
                self.__steering = -1.

        if self.__use_rc_motor:
            pass

        else:
            real_pwm = self.scale_range(self.__gas_duty_cycle, 0., 1., 0., 255.)
            self.__pi_car.set_PWM_dutycycle(self.__motor_forward_pin, real_pwm)

            if self.__pi_car.read(self.__encoder_dir_pin) == self.__forward_pin_level:
                real_pwm = self.scale_range(self.__break_duty_cycle, 0., 1., 0., 255.)
                self.__pi_car.set_PWM_dutycycle(self.__motor_backward_pin, real_pwm)
            else:
                self.__pi_car.set_PWM_dutycycle(self.__motor_backward_pin, 0)

        real_pwm = self.scale_range(self.__steering, -1., 1., 1000., 2000.)
        self.__pi_car.set_servo_pulsewidth(self.__servo_pin, real_pwm)

        self.__get_image()

        return self.image, self.__reward, self.__done, self.__info

    def close(self) -> None:
        """Close system.

        Close camera and clear GPIO.

        Returns: None.

        """
        self.__cam.close()

        self.__pi_car.set_PWM_dutycycle(self.__motor_backward_pin, 0)
        self.__pi_car.set_PWM_dutycycle(self.__motor_forward_pin, 0)
        self.__pi_car.set_servo_pulsewidth(self.__servo_pin, 0)
        self.__pi_car.stop()

    def __get_image(self) -> None:
        """Get image from camera.

        Returns: None.

        """
        self.__cam.capture(self.image, 'rgb', use_video_port=True)

    def __line_interrupt_handle(self) -> None:
        """Handles ecent from line sensor.

        Triggers done when car hit lines.

        Returns: None.

        """
        self.__done = True

    @staticmethod
    def scale_range(old_value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
