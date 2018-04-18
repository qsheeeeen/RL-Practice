from rl_toolbox import Runner
from rl_toolbox.agent import PPOAgent
from rl_toolbox.policy import CNNPolicy, VisualMemoryPolicy


def main():
    runner = Runner(
        'CarRacing-v0',
        PPOAgent,
        CNNPolicy,
        record_data=False,
        data_path=None,
        save=False,
        load=False,
        weight_path='./weights/cnn_policy_weights.pth')

    # runner = Runner(
    #     'CarRacing-v0',
    #     PPOAgent,
    #     VisualMemoryPolicy,
    #     record_data=False,
    #     data_path=None,
    #     save=False,
    #     load=False,
    #     weight_path='./weights/vm_policy_weights.pth')

    runner.run(num_episode=100, num_worker=1, train=True)


if __name__ == '__main__':
    main()
