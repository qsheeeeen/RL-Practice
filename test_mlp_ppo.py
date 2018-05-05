from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import MLPLSTMPolicy, MLPPolicy


def main():
    for policy_fn in (MLPPolicy, MLPLSTMPolicy):
        runner = Runner(
            'LunarLanderContinuous-v2',
            PPOAgent,
            policy_fn,
            record_data=False,
            data_path=None,
            save=True,
            load=False,
            weight_path='./weights/',
            seed=123)

        runner.run(num_episode=1000)
        # runner.run({'use_gpu': False}, num_episode=1000)  # For debug.


if __name__ == '__main__':
    main()
