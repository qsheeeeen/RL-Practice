from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import MLPPolicy, CNNPolicy


def main():
    runner = Runner('LunarLanderContinuous-v2', PPOAgent, MLPPolicy)
    # runner = Runner('CarRacing-v0', PPOAgent, CNNPolicy, num_worker=8)
    runner.run(num_worker=1)


if __name__ == '__main__':
    main()
