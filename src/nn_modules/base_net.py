from torch import nn


class BaseNet(nn.Module):
    def __init__(self, kernel_sizes: list, channels: list):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=kernel_sizes[0], padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[0]),
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_sizes[1], padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[1])
        )

    def forward(self, x):
        # B x 1 x F x T
        x = self.features(x)
        x = x.squeeze(1)
        return x
