# coding: utf-8
import zerorpc

from agent import JoystickAgent
from tools import Dashboard


def main():
    env = zerorpc.Client()
    env.connect("tcp://127.0.0.1:4242")

    print('Init Dashboard.')
    dashboard = Dashboard()

    agent = JoystickAgent()

    ob = env.reset()
    action = agent.act(ob)

    close = False
    while not close:
        ob, reward, done, info = env.step(action)
        action = agent.act(ob, reward, done)
        close = dashboard.update(ob, info)

    agent.close()


if __name__ == '__main__':
    main()
