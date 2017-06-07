# coding: utf-8

import time

import h5py
import numpy as np

from agent.controller_agent import ControllerAgnet
from environment.racing_car import RacingCar


def main():
    agent = ControllerAgnet()
    env = RacingCar()

    h5_file = h5py.File('./data_set.hdf5', 'a')
    group = h5_file.create_group(time.ctime().replace(' ', '-'))

    camera_data_set = group.create_dataset('camera_data',
                                           ((100,) + env.image.shape),
                                           env.image.dtype,
                                           max_shape=((None,) + env.image.shape),
                                           chunks=(100,) + env.image.shape)

    car_data_set = group.create_dataset('car_data',
                                        ((100,) + env.action_space),
                                        np.float,
                                        max_shape=((None,) + env.action_space),
                                        chunks=(100,) + env.action_space)
    data_num = 0
    ob, reward, done, info = env.reset()
    while True:
        action = agent.act(ob, reward, done)

        camera_data_set[data_num, :, :, :] = ob
        car_data_set[data_num, :] = action

        data_num += 1

        ob, reward, done, info = env.step(action)


if __name__ == '__main__':
    main()
