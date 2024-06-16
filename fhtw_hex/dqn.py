import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from hex_engine import hexPosition
import matplotlib.pyplot as plt
import torch.nn.functional as F



class ChannelSplitter(nn.Module):
    def __init__(self):
        super(ChannelSplitter, self).__init__()

    def forward(self, x):
        white_stones = (x == 1).float()
        black_stones = (x == -1).float()
        return torch.cat([white_stones, black_stones], dim=1)
class DQN(nn.Module):
    def __init__(self, board_size, output_dim):
        super(DQN, self).__init__()
        self.splitter = ChannelSplitter()  # Add the custom layer
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.splitter(x)  # Use the custom layer
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x