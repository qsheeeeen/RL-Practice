from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNLSTMPolicy, CNNPolicy


def main():
    for policy_fn in (CNNPolicy, CNNLSTMPolicy):
        for i in range(2, 12, 2):
            runner = Runner(
                'CarRacing-v0',
                PPOAgent,
                policy_fn,
                record_data=False,
                data_path=None,
                save=True,
                load=(i > 2),
                weight_path='./weights/')

            runner.run({'abs_output_limit': i / 10}, num_episode=500, num_worker=1, continue_plot=(i < 12))


if __name__ == '__main__':
    main()
