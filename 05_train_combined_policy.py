from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import VAEPolicy


def main():
    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        VAEPolicy,
        record_data=False,
        data_path=None,
        save=False,
        load=True,
        weight_path='./weights/')

    runner.run(abs_output_limit=[1, 1, 1])


if __name__ == '__main__':
    main()
