# Reinforcement learning toolbox (Unfinished)
A self-driving model car running reinforcement learning algorithms implemented in PyTorch.
## 1. Result
### 1.1 gym environment:
<img src="image/result1.png" align="middle" width="3000"/>

### 1.2 Real track:
## 2. Setup
### 2.1. Hardware:

### 2.2. Software:
Install dependencies according to [rpi_requirements.txt](rpi_requirements.txt) and [requirements.txt](requirements.txt).
## 3. Run
On Pi:

    python3 client_run.py    
### 3.1 For fun or gathering data.
On PC:
    
    python3 server_run.py -A joystick
### 3.2 Train in real world.
On PC:

    python3 server_run.py -A ppo
## 4. TODO
[Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

[Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)


## Reference
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[openai/baselines](https://github.com/openai/baselines)
## License
This project is released under MIT License. Please review [License](LICENSE) file for more details.

