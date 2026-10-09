import torch.nn as nn


class UnionNet(nn.Module):
    def __init__(self, modules):
        super(UnionNet, self).__init__()
        self.layers = nn.Sequential(*modules) 

    def forward(self, x):
        return self.layers(x)
