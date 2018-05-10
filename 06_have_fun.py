from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import VAEPolicy, CNNPolicy


def main():
    for fun in (VAEPolicy, CNNPolicy):
        runner = Runner(
            'CarRacing-v0',
            PPOAgent,
            fun,
            record_data=False,
            data_path=None,
            save=False,
            load=True,
            weight_path='./weights/')

        runner.run({'train': False})


if __name__ == '__main__':
    main()
