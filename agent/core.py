# coding: utf-8


class Agent(object):
    def act(self, observation, reward, done):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
