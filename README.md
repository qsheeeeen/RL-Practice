# Self Driving Car (Unfinished)
A self-driving model car. About 1:18 compare to real car.


## 1. Result

### 1.1 Training track:

### 1.2 Test track:

## 2. Setup

### 2.1. Hardware:


##### 2.1.1 Raspberry Pi 3:
It will be running `client_run.py` for gathering data and performing action.
Do not need much computing power.
This is the reason why I choose a Raspberry Pi.

##### 2.1.2 PC:
It will be running `server_run.py`.
Deep Reinforcement learning algorithm needs a lot of computing power.
I run it on a gaming laptop with GTX 980m. It performs quite well.

### 2.2. Software:
Depending on what you intend to do, different package are needed.
See [requirements.txt](requirements.txt) for more detail.
To make the whole system work.
`numpy`, `pyzmq`, `blosc`, `pigpio` and `picamera` should be installed on Raspberry Pi.
`numpy`, `pyzmq`, `blosc`, `pygame` and `torch` should be installed on PC.
## 3. Run

### 3.1 Gather training data

### 3.2 Train in virtual environment

### 3.3 Run in real world

## 4. Next stage

1:10 RC Model.

Jetson TX2.

Maybe drift?

Stay tuned!

## Reference
[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

[Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)

## License
This project is released under MIT License.
Please review [License](LICENSE) file for more details.

