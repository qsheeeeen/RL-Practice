from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import VAEPolicy, VAELSTMPolicy


def main():
    for i in range(2, 12, 2):
        runner = Runner(
            'CarRacing-v0',
            PPOAgent,
            VAEPolicy,
            record_data=False,
            data_path=None,
            save=True,
            load=(i > 2),
            weight_path='./weights/')

        runner.run({'abs_output_limit': i / 10}, num_episode=500, continue_plot=(i < 12))

    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        VAELSTMPolicy,
        save=True,
        load=True,
        weight_path='./weights/')

    runner.run()


if __name__ == '__main__':
    main()
