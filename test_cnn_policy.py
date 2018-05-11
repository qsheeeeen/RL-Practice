from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNLSTMPolicy


def main():
    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        CNNLSTMPolicy,
        record_data=False,
        data_path=None,
        save=False,
        load=False,
        weight_path='./weights/')

    runner.run(extra_message='simple-train')


if __name__ == '__main__':
    main()
