import torch.nn as nn
import torch

class BaseNet(nn.Module):
    def __init__(self, kernel_sizes: list, strides: list, paddings: list, channels = 16):
        super(BaseNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0]),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # B x 1 x F x T
        x = self.features(x)
        return x
