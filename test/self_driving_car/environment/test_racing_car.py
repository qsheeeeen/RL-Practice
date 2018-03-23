import time

import numpy as np

from self_driving_car.environment import RacingCar


def limit(x):
    return np.maximum(np.minimum(x, 1), -1)


def main():
    env = RacingCar()

    ob = env.reset()
    for x in range(10000):
        ob, r, d, _ = env.step([0.1, 0.1])
        print(time.ctime())
        print(_)
        if d:
            print('Done.')
            break


if __name__ == '__main__':
    main()
