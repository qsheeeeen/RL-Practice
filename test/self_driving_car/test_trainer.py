import gym

from self_driving_car import Trainer
from self_driving_car.agent.policy.shared import CNNPolicy


def main():
    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    del env

    trainer = Trainer(CNNPolicy, inputs, outputs, './agent/data.h5')

    trainer.fit(batch_size=32, epochs=1)

    trainer.save()


if __name__ == '__main__':
    main()
