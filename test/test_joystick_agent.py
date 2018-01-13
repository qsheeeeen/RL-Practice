def main():
    # TODO: Test DDPG
    def main():
        # env = gym.make('Enduro-v0')
        env = gym.make('MountainCarContinuous-v0')
        observation = env.reset()

        # agent = AIAgent()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            action = np.random.rand(1) * 2 - 1
            print(action)
            observation, reward, done, info = env.step(action)

    print("Episode finished")

    if __name__ == '__main__':
        main()


if __name__ == '__main__':
    main()
