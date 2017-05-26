#!/usr/local/bin/python3
# coding: utf-8

import numpy as np
import picamera
import pigpio

class RacingCar(object):
    def __init__(self):
        self.__image_size = (320, 240)

        self.__cam = picamera.Picamera()

        self.__cam.resolution = self.__image_size
        self.__cam.framerate = 40
        self.__cam.exposure_mode = 'sport'

        image_width, image_height = self.__image_size

        self.__image = np.empty((image_height, image_width, 3), dtype=np.uint8)

        pi = pigpio.pi()

        self.__reward = 0.
        self.__done = False
        self.__info = [''] * 10

        self.action_space = len(self.__order_list)

    def reset(self) -> (np.ndarray, int, bool, list):
        self.__get_image()
        self.__reward.value = 0
        self.__done.value = False
        self.__info.value = range(10)

        return self.__image, self.__reward, self.__done, self.__info

    def step(self, action: np.ndarray) -> (np.ndarray, int, bool, list):
        self.__perform_action(action)
        self.__get_image()

        return self.__image, self.__reward, self.__done, self.__info

    def close(self) -> None:
        self.__com.close()
        self.__cam.close()

    def __perform_action(self, action) -> None:
        index = np.argmax(action)
        self.__send_order(self.__order_list[index])

    def __get_image(self) -> None:
        """

        Returns: None

        """
        self.__cam.capture(self.__image, 'rgb', use_video_port=True)
