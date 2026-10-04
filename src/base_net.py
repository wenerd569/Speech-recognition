import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self, kernel_sizes: list, strides: list, paddings: list):
        super(BaseNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=kernel_sizes, stride=strides[0], padding=paddings[0])),
            nn.BatchNorm2d(10),
            nn.Conv2d(2, 4, kernel_size=kernel_sizes, stride=strides[1], padding=paddings[1]),
            nn.BatchNorm2d(10),
            nn.Conv2d(4, 8, kernel_size=kernel_sizes, stride=strides[2], padding=paddings[2]),
            nn.BatchNorm2d(10),
            nn.Conv2d(8, 16, kernel_size=kernel_sizes, stride=strides[3], padding=paddings[3]),
            nn.BatchNorm2d(10),
        )

    def forward(self, x):
        #B x 1 x F x T
        x = self.features(x)
        return x