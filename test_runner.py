from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNPolicy


def main():
    # runner = Runner('LunarLanderContinuous-v2', PPOAgent, MLPPolicy, data_path='./data/')
    runner = Runner('CarRacing-v0', PPOAgent, CNNPolicy, data_path='./data/', load=True)
    runner.run(num_episode=1000, num_worker=1, train=True)


if __name__ == '__main__':
    main()
