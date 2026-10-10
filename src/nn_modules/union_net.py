from torch import nn


class UnionNet(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.layers = nn.Sequential(*modules) 

    def forward(self, x):
        return self.layers(x)
