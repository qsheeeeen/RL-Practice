from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import MLPLSTMPolicy, MLPPolicy


def main(recurrent=True, multi_agnet=True):
    if recurrent:
        policy_fn = MLPLSTMPolicy
    else:
        policy_fn = MLPPolicy

    if multi_agnet:
        num = 4
    else:
        num = 1

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
    for r in (False, True):
        for n in (False, True):
            main(r, n)
