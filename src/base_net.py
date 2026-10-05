import torch.nn as nn
import torch

class BaseNet(nn.Module):
    def __init__(self, kernel_sizes: list, strides: list, paddings: list):
        super(BaseNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )

    def forward(self, x):
        # B x 1 x F x T
        # print(f"BaseNet input shape: {x.shape}")
        x = self.features(x)
        # print(f"BaseNet output shape: {x.shape}")
        return x