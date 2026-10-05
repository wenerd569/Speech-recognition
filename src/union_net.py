import torch.nn as nn


class UnionNet(nn.Module):
    def __init__(self, base_net: nn.Module, head: nn.Module):
        super(UnionNet, self).__init__()
        self.base_net = base_net
        self.head = head

    def forward(self, x):
        x = self.base_net(x)
        x = self.head(x)
        return x

