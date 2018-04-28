from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import MLPLSTMPolicy, MixtureMLPPolicy, MLPPolicy


def main():
    for num in (1, 4):
        for policy_fn in (MLPPolicy, MixtureMLPPolicy, MLPLSTMPolicy):
            runner = Runner(
                'LunarLanderContinuous-v2',
                PPOAgent,
                policy_fn,
                record_data=False,
                data_path=None,
                save=True,
                load=False,
                weight_path='./weights/')

            runner.run(num_episode=1000, num_worker=num)


if __name__ == '__main__':
    main()
