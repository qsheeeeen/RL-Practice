from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNPolicy


def main():
    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        CNNPolicy,
        record_data=True,
        data_path='./data/',
        save=False,
        load=True,
        weight_path='./weights/')

    runner.run(num_episode=500, num_worker=1, train=False)


if __name__ == '__main__':
    main()
