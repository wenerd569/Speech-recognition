import torch.nn as nn
import torch
import numpy as np


class MeanNet(nn.Module):
    def __init__(self, output_features: int, input_features: int):
        super(MeanNet, self).__init__()
        self.output_features = output_features
        self.linear = nn.Linear(input_features, output_features)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x.shape = (B, F',T')
        x = torch.mean(x, dim=2, keepdim=False)
        x = self.linear(x)
        x = self.logsoftmax(x)
        return x