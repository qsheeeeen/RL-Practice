import time

import numpy as np

from self_driving_car.environment import RacingCar


def main():
    env = RacingCar()
    env.reset()

    for x in range(10000):
        time.sleep(0.1)
        a = np.random.rand(2)
        ob, r, d, _ = env.step(a)
        if d:
            print('Done.')
            break


if __name__ == '__main__':
    main()
