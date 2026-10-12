from torch import nn

class ConvNet(nn.Module):
    def __init__(self, conv_cnt: int, kernel_sizes: list, channels: list):
        super().__init__()

        self.features = []
        for i in range(conv_cnt):
            self.features.append(nn.Conv2d(channels[i], channels[i+1],kernel_size=kernel_sizes[i], padding="same")),
            self.features.append(nn.ReLU()),
            self.features.append(nn.BatchNorm2d(channels[i+1]))
        
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        # B x 1 x F x T
        x = self.features(x)
        x = x.squeeze(1)
        return x
