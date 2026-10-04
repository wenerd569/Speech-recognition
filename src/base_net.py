import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, kernel_size):
        super(BaseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=kernel_size, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        return x