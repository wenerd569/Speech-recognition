import torch
from torch import nn


class MeanNet(nn.Module):
    def __init__(self, output_features: int, input_t: int):
        super().__init__()
        self.output_features = output_features
        self.linear = nn.Linear(input_t, output_features)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x.shape = (B, F',T')
        x = x[:, -1, :]
        x = self.linear(x)
        x = self.logsoftmax(x)
        return x