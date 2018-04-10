import argparse

from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNPolicy


def get_args():
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--env-name', type=str, default='CarRacing-v0', metavar='N',
                        help='name of gym environments.')

    parser.add_argument('--load', action='store_true', default=True,
                        help='load trained weights.')
    parser.add_argument('--weights-path', type=str, default='./weights/vae_weights.pth', metavar='N',
                        help='where the weights file is.')

    parser.add_argument('--data-path', type=str, default='./data/', metavar='N',
                        help='where the data file is.')

    parser.add_argument('--train', action='store_true', default=True,
                        help='train or not.')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    return parser.parse_args()


def main():
    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        CNNPolicy,
        record_data=True,
        data_path='./data/',
        save=False,
        load=False,
        weight_path='./weights/lstm_policy_weights.pth')

    runner.run(num_episode=1000, num_worker=2, train=False)


if __name__ == '__main__':
    main()
