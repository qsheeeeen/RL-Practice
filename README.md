# Self Driving Car (Unfinished)
A self-driving model car running reinforcement learning algorithms implemented in PyTorch.

## 1. Result

### 1.1 gym environment:

### 1.2 Real track:

## 2. Setup

### 2.1. Hardware:

### 2.2. Software:

#### 2.2.1 Raspberry Pi:
    pip3 install -r rpi_requirements.txt
#### 2.2.2 PC:
    pip3 install -r requirements.txt
Then install [PyTorch](https://http://pytorch.org/).

## 3. Run

### 3.1 Train in gym.

    python3 have_fun.py
    
### 3.2 Gather training data.
On PC:
    
    python3 server_run.py --joystick
On Pi:

    python3 client_run.py

### 3.3 Train in real world.
On PC:

    python3 server_run.py
On Pi:
    
    python3 client_run.py
## 4. TODO
[Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

[Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)


## Reference
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[openai/baselines](https://github.com/openai/baselines)

## License
This project is released under MIT License.
Please review [License](LICENSE) file for more details.

