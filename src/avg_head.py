
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, kernel_size):
        super(BaseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=kernel_size),
            ## TODO
        )

    def forward(self, x):
        x = self.features(x)
        return x