import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_preparing import MelSpecDataset
from torch.testing._internal.data.network1 import Net
from base_net import BaseNet

DIR_TO_SAVE_TO = os.path.join(os.path.dirname(DIR_CURRENT), "data/data_proccessed")
BATCH_SIZE = 32


def train(net, epochs: int, train_loader, validate_loader, optimizer, loss):
    net.train()
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



net = BaseNet(kernel_sizes=[3, 3, 3, 3], strides=[1, 1, 1, 1], paddings=[1, 1, 1, 1])

optimizer = Adam(net.parameters(), lr=0.001)
loss = nn.NLLLoss()

df_train = pd.read_csv(os.path.join(DIR_TO_SAVE_TO, "df_train.csv"))
train_dataset = MelSpecDataset(df_train, DIR_TO_SAVE_TO)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)



train(net, epochs=40, train_loader=train_loader, validate_loader=validate_loader, optimizer=optimizer, loss=loss)
