import gym
import torch

from self_driving_car.agent import PPOAgent
from self_driving_car.policy.shared import CNNPolicy


# nvprof --profile-from-start off -o trace_name.prof -- python3 gpu_profile.py

def main():
    env = gym.make('CarRacing-v0')

    inputs = env.observation_space.shape
    outputs = env.action_space.shape

    agent = PPOAgent(CNNPolicy, inputs, outputs, horizon=128, lr=2.5e-4, num_epoch=4, batch_size=4, clip_range=0.1,
                     output_limit=0.5)

    ob = env.reset()
    env.render()
    action = agent.act(ob)
    for i in range(255):
        ob, r, d, _ = env.step(action)
        env.render()
        action = agent.act(ob, r, d)
        if d:
            break

    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():
            agent.act(ob)

    env.close()

    # prof = torch.autograd.profiler.load_nvprof('trace_name.prof')
    # print(prof)


if __name__ == '__main__':
    main()
