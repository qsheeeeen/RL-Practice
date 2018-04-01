import random

import numpy as np

from self_driving_car.tool.dashboard import Dashboard


def main():
    dashboard = Dashboard()

    done = False
    while not done:
        img = np.random.randint(0, 255, (240, 240, 3), np.uint8)

        name = ('Motor signal', 'Steering signal', 'Car speed')
        num = (random.random(), random.random(), random.random())
        info = dict(zip(name, num))

        done = dashboard.update(img, info)

    print('Closed.')


if __name__ == '__main__':
    main()
