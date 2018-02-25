# coding: utf-8


class Agent(object):
    def act(self, *inputs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

