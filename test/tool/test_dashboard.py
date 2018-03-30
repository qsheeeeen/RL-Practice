import numpy as np
import random

from self_driving_car.tool import Dashboard


def main():
    dashboard = Dashboard()

    done = False
    while not done:
        img = np.random.rand(320, 240, 3).astype(np.float32)

        name = ('Motor signal', 'Steering signal', 'Car speed')
        num = (random.random(), random.random(), random.random())
        info = dict(zip(name, num))

        done = dashboard.update(img, info)

    print('Closed.')


if __name__ == '__main__':
    main()
