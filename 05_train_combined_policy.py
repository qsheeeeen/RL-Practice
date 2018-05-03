from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import VAEPolicy, VAELSTMPolicy


def main():
    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        VAEPolicy,
        save=True,
        load=True,
        weight_path='./weights/')

    runner.run()

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
