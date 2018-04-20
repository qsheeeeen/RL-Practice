from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNPolicy


def policy_fn1(policy):
    return PPOAgent(policy, train=True, abs_output_limit=0.5)


def policy_fn2(policy):
    return PPOAgent(policy, train=True)


def main():
    runner = Runner(
        'CarRacing-v0',
        policy_fn1,
        CNNPolicy,
        record_data=False,
        data_path=None,
        save=True,
        load=False,
        weight_path='./weights/')

    runner.run(num_episode=1000, num_worker=4)

    runner = Runner(
        'CarRacing-v0',
        policy_fn2,
        CNNPolicy,
        record_data=False,
        data_path=None,
        save=True,
        load=True,
        weight_path='./weights/')

    runner.run(num_episode=1000, num_worker=4)


if __name__ == '__main__':
    main()
