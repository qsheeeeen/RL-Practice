from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import MLPPolicy


def main():
    runner = Runner('LunarLanderContinuous-v2', PPOAgent, MLPPolicy, num_worker=8)
    runner.run()


if __name__ == '__main__':
    main()
