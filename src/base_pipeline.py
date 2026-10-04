import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_preparing import MelSpecDataset
from torch.testing._internal.data.network1 import Net
from base_net import BaseNet


def train(net, epochs: int, train_loader, validate_loader, optimizer, loss):
    
    for epoch in range(10):
        print(f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss_value = loss(output, target)
            loss_value.backward()
            optimizer.step()
        
        for batch_idx, (data, target) in enumerate(validate_loader):
            output = net(data)
            loss_value = loss(output, target)
            print(f"Validation loss: {loss_value.item()}")

???


net = BaseNet(kernel_sizes=[3, 3, 3, 3], strides=[1, 1, 1, 1], paddings=[1, 1, 1, 1])

optimizer = Adam(net.parameters(), lr=0.001)
loss = nn.NLLLoss()


train(net, epochs=40, train_loader=train_loader, validate_loader=validate_loader, optimizer=optimizer, loss=loss)
