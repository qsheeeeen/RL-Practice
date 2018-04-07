import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(4096, 512)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))

        h3 = h3.view(h3.size(0), -1)

        return F.relu(self.fc(h3))


class SmallCNNTranspose(nn.Module):
    def __init__(self):
        super(SmallCNNTranspose, self).__init__()

        self.fc = nn.Linear(512, 4096)

        self.convt1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.convt3 = nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4)

    def forward(self, x):
        h1 = F.relu(self.fc(x))

        h1 = h1.view(h1.size(0), 64, 8, 8)

        h2 = F.relu(self.convt1(h1))
        h3 = F.relu(self.convt2(h2))

        return F.sigmoid(self.convt3(h3))
