import h5py
import pygame


class JoystickAgent(object):
    def __init__(self):
        self.steering_axis = 0
        self.break_axis = 2
        self.gas_axis = 5

        pygame.init()
        pygame.joystick.init()

        assert pygame.joystick.get_count() > 0, 'Can not find controller.'
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.file = h5py.File('./data.h5', 'w')

        self.state_data_set = self.file.create_dataset('state', (10000, 96, 96, 3), np.float32, chunks=(1, 96, 96, 3))
        self.action_data_set = self.file.create_dataset('action', (10000, 2), np.float32, chunks=(1, 2))

        self.count = 0

        while True:
            pygame.event.get()
            gas_signal = -self.joystick.get_axis(self.gas_axis)
            break_signal = self.joystick.get_axis(self.break_axis)

            # print(gas_signal)
            print(break_signal)
            # print()

            if (gas_signal == 1) and (break_signal == -1):
                break

    def act(self, state, reward=0, done=False):
        pygame.event.get()

        steering = self.joystick.get_axis(self.steering_axis)

        gas_signal = -self.joystick.get_axis(self.gas_axis)
        break_signal = self.joystick.get_axis(self.break_axis)

        gas_signal = (gas_signal - 1) / -2
        break_signal = (break_signal + 1) / 2

        gas_break = gas_signal + break_signal

        action_array = np.array((steering, gas_break), dtype=np.float32)

        self.state_data_set[self.count] = state
        self.action_data_set[self.count] = action_array

        self.count += 1

        return action_array

    def close(self):
        pygame.quit()
        self.file.close()

    def load(self):
        raise NotImplementedError


if __name__ == '__main__':
    def convert_action(action):
        x = np.array([action[0], 0, 0])
        if action[1] > 0:
            x[1] = action[1]
        else:
            x[2] = -action[1]
        return x


    import gym
    import time
    import numpy as np

    agent = JoystickAgent()
    env = gym.make('CarRacing-v0')
    print('Begin.')
    ob = env.reset()
    env.render()
    action = agent.act(ob)

    action = convert_action(action)

    for x in range(100000):
        ob, r, d, _ = env.step(action)
        env.render()
        action = agent.act(ob, r, d)

        action = convert_action(action)

        if d:
            print('Done x:{} '.format(x))
            print(time.ctime())
            print()
            d = False
            agent.close()
            break
