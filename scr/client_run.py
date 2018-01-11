import zmq

from environment import RacingCar
from tools import send_array, receive_array


def client_run():
    print('Init environment.')
    env = RacingCar()
    data = env.reset()

    print('Init communication.')
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.2:5555")

    print('Send data for the first time...')
    send_array(socket, data)

    print('Wait for action...', end='\t')
    action = receive_array(socket)
    print('Com success.')

    try:
        while True:
            data = env.step(action)

            print('Send result...', end='\t')
            send_array(socket, data)
            print('Done.')

            print('Wait for action...', end='\t')
            action = receive_array(socket)
            print('Received.')
    finally:
        env.close()


if __name__ == '__main__':
    client_run()
