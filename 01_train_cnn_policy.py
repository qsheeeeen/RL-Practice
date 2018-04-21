from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNPolicy


def main():
    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        CNNPolicy,
        record_data=False,
        data_path=None,
        save=True,
        load=False,
        weight_path='./weights/')

    runner.run(num_episode=1000, num_worker=4)


if __name__ == '__main__':
    main()
