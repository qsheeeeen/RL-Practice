from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent, RandomAgent
from rl_toolbox.policy import MLPPolicy, CNNPolicy


def main():
    # runner = Runner('LunarLanderContinuous-v2', PPOAgent, MLPPolicy, data_path='./data/')
    runner = Runner('CarRacing-v0', PPOAgent, CNNPolicy, data_path='./data/')
    runner.run(num_episode=10,num_worker=1)


if __name__ == '__main__':
    main()
