import numpy as np
import picamera
import pigpio


class RacingCar(object):
    def __init__(self, time_limit=1000, motor_limitation=0.2):
        assert motor_limitation <= 1, '"motor_limitation" should not > 1.'

        # Pin configuration.
        self.STEERING_SERVO_PIN = 23
        self.MOTOR_PIN = 24
        self.LEFT_LINE_SENSOR_PIN = 17
        self.RIGHT_LINE_SENSOR_PIN = 18
        self.ENCODER_DIR_PIN = 21
        self.ENCODER_PUL_PIN = 20
        self.ENCODER_ZERO_PIN = 19
        self.FORWARD_PIN_LEVEL = 0

        # Hardware configuration.
        self.TIRE_DIAMETER = 0.05
        self.GEAR_RATIO = 145. / 35.
        self.ENCODER_LINE = 512
        self.SERVO_RANGE = 200
        self.MOTOR_RANGE = 500 * motor_limitation

        # Parameters.
        self.SPEED_UPDATE_INTERVAL = 500
        self.REWARD_UPDATE_INTERVAL = 500

        self.time_limit = time_limit
        self.time_count = 0

        self.last_reward = 0
        self.total_reward = 0.
        self.encoder_pulse_count = 0
        self.speed = 0.
        self.done = False

        self.image_width, self.image_height = 96, 96
        self.image = np.empty((self.image_height, self.image_width, 3), dtype=np.uint8)

        self.observation_space = np.random.randint(0, 256, (self.image_height, self.image_width, 3), np.uint8)
        self.action_space = np.array([1, -1], np.float32)

        # Hardware controlling.
        self.pi = pigpio.pi()
        self.pi.set_watchdog(self.ENCODER_PUL_PIN, self.SPEED_UPDATE_INTERVAL)
        self.pi.set_watchdog(self.ENCODER_ZERO_PIN, self.REWARD_UPDATE_INTERVAL)
        self.pi.callback(self.ENCODER_PUL_PIN, pigpio.RISING_EDGE, self._interrupt_handle)
        self.pi.callback(self.ENCODER_ZERO_PIN, pigpio.RISING_EDGE, self._interrupt_handle)
        self.pi.callback(self.LEFT_LINE_SENSOR_PIN, pigpio.FALLING_EDGE, self._interrupt_handle)
        self.pi.callback(self.RIGHT_LINE_SENSOR_PIN, pigpio.FALLING_EDGE, self._interrupt_handle)

        self._update_pwm(0, 0)

        # Camera.
        self.cam = picamera.PiCamera()
        self.cam.resolution = 96, 96
        self.cam.framerate = 30
        self.cam.exposure_mode = 'sports'
        self.cam.image_effect = 'denoise'
        self.cam.meter_mode = 'backlit'

    def reset(self):
        self._update_pwm(0, 0)
        self._update_image()

        self.time_count = 0

        self.last_reward = 0
        self.total_reward = 0.
        self.encoder_pulse_count = 0
        self.speed = 0.
        self.done = False

        return self.image

    def step(self, action):
        assert len(action) == 2, 'Incorrect input shape.'

        action = np.minimum(np.maximum(action, -1), 1)

        if self.done:
            steering_signal, motor_signal = 0, 0
        else:
            steering_signal, motor_signal = action

        self._update_pwm(steering_signal, motor_signal)

        self._update_image()

        reward = self.total_reward - self.last_reward
        self.last_reward = self.total_reward

        self.time_count += 1
        if self.time_count > self.time_limit:
            self.done = True

        car_info = self._generate_info(steering_signal, motor_signal)

        return self.image, reward, self.done, car_info

    def close(self):
        self.cam.close()
        self._update_pwm(0, 0)
        self.pi.stop()

    def _update_image(self):
        self.cam.capture(self.image, 'rgb', use_video_port=True)

    def _interrupt_handle(self, gpio, level, tick):
        print('Interrupt trigered.')
        print('gpio: {}\t level: {}'.format(gpio, level))

        if (gpio == self.LEFT_LINE_SENSOR_PIN) or (gpio == self.RIGHT_LINE_SENSOR_PIN):
            print('Done trigered.')
            self.done = True

        elif gpio == self.ENCODER_PUL_PIN:
            if level == pigpio.RISING_EDGE:
                print('pulse count +1.')
                self.encoder_pulse_count += 1

            elif level == pigpio.TIMEOUT:
                print('Calculate speed.')
                s = self.encoder_pulse_count / self.ENCODER_LINE * np.pi * self.TIRE_DIAMETER
                t = self.SPEED_UPDATE_INTERVAL / 1000
                self.speed = s / t * self.GEAR_RATIO
                self.encoder_pulse_count = 0

            else:
                raise NotImplementedError('Unknown level for encoder pulse pin: {}.'.format(level))

        elif gpio == self.ENCODER_ZERO_PIN:
            if level == pigpio.RISING_EDGE:
                print('total_reward += 5.')
                self.total_reward += 500

            elif level == pigpio.TIMEOUT:
                print('total_reward -= 1.')
                self.total_reward -= 1

            else:
                raise NotImplementedError('Unknown level for encoder zero pin: {}.'.format(level))
        else:
            raise NotImplementedError('Unknown GPIO pin: {}.'.format(gpio))

    def _update_pwm(self, steering_signal, motor_signal):

        def scale_range(old_value, old_min, old_max, new_min, new_max):
            return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

        pwm_wave = scale_range(steering_signal, -1, 1, (1500 - self.SERVO_RANGE), (1500 + self.SERVO_RANGE))
        self.pi.set_servo_pulsewidth(self.STEERING_SERVO_PIN, pwm_wave)

        pwm_wave = scale_range(motor_signal, -1, 1, (1500 - self.MOTOR_RANGE), (1500 + self.MOTOR_RANGE))
        self.pi.set_servo_pulsewidth(self.MOTOR_PIN, pwm_wave)

    def _generate_info(self, steering_signal, motor_signal):
        return {'steering_signal': float(steering_signal),
                'motor_signal': float(motor_signal),
                'total_reward': float(self.total_reward),
                'speed': float(self.speed),
                'done': float(self.done)}
