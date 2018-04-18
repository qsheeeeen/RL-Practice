from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import MLPPolicy


def main():
    runner = Runner(
        'LunarLanderContinuous-v2',
        PPOAgent,
        MLPPolicy,
        record_data=False,
        data_path=None,
        save=False,
        load=False,
        weight_path='./weights/mlp_lstm_policy_weights.pth')

    runner.run(num_episode=600, num_worker=1, train=True)


if __name__ == '__main__':
    main()
