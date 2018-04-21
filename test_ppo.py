from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import MLPLSTMPolicy


def main():
    runner = Runner(
        'LunarLanderContinuous-v2',
        PPOAgent,
        MLPLSTMPolicy,
        record_data=False,
        data_path=None,
        save=True,
        load=False,
        weight_path='./weights/')

    runner.run(num_episode=1000, num_worker=1)


if __name__ == '__main__':
    main()
