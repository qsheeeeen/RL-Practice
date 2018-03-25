import random
from multiprocessing import Process
import time

import numpy as np

from self_driving_car.tool import Com


def server_run():
    server = Com('server')

    for i in range(10):
        data = server.receive_data()
        server.send_data(data)


def client_run():
    client = Com('client', '127.0.0.1')
    for i in range(10):
        array = np.random.randint(0, 255, (240, 240, 3), np.uint8)
        reward = random.random()
        done = False
        info = {'asdfasdf': random.random(), 'qwerqwer': random.random()}

        data = (array, reward, done, info)

        start = time.time()

        client.send_data(data)
        receive = client.receive_data()

        print('Time used: {}'.format(time.time()-start))

        if (data[0] == receive[0]).all():
            print('Check OK.')
        if data[1] == data[1]:
            print('Check OK.')
        if data[2] == data[2]:
            print('Check OK.')
        if data[3] == data[3]:
            print('Check OK.')


def main():
    client_process = Process(target=client_run)
    server_process = Process(target=server_run)
    server_process.start()
    client_process.start()

    server_process.join()
    client_process.join()


if __name__ == '__main__':
    main()
