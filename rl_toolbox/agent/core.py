class Agent(object):
    def act(self, state, reward=0., done=False):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
