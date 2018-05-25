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
        save=True,
        load=False,
        weight_path='./weights/')

    kwargs = {
        'horizon': 512,
        'buffer_size': 4,
        'lr': 3e-4,
        'num_epoch': 10,
        'batch_size': 64,
        'clip_range': 0.2}

    runner.run(num_episode=1000, extra_message='1-layer-lstm', **kwargs)
    # runner.run({'use_gpu': False}, num_episode=1000)  # For debug.


if __name__ == '__main__':
    main()
