import gym

from self_driving_car.policy import CNNPolicy
from self_driving_car.trainer import Trainer


def main():
    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    env.close()
    del env

    trainer = Trainer(CNNPolicy, inputs, outputs, './agent/data.hdf5')

    trainer.fit(batch_size=32, epochs=10, test_split=0.2)

    trainer.save()


if __name__ == '__main__':
    main()
