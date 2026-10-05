import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, DataLoader
from torch.testing._internal.data.network1 import Net
from torchinfo import summary


from data_preparing import MelSpecDataset
from base_net import BaseNet
from mean_net import MeanNet
from union_net import UnionNet

from wav_preproccess import DIR_CURRENT, DIR_TO_SAVE_TO, DF_PATH

BATCH_SIZE = 32
LABELS = sorted(pd.read_csv(DF_PATH)["label"].unique())
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
from wav_preproccess import N_mels, Target_T
BASE_NET_INPUT_SHAPE = (32, 1, N_mels, Target_T)

def train(net, epochs: int, train_loader, validate_loader, optimizer, loss):
    previous_validation_loss = None
    for epoch in range(epochs):
        net.train()
        print(f"Epoch {epoch}")
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = net(data)
            loss_value = loss(output, target)

            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item()
        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss}")

        
        avg_accuracy = 0.0
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            for data, target in validate_loader:
                output = net(data)
                validation_loss += loss(output, target).item()
                _, preds = torch.max(output, dim=1)
                avg_accuracy += (preds == target).float().mean().item()
        
        validation_loss = validation_loss / len(validate_loader)
        print(f"Validation loss: {validation_loss}")
        
        if previous_validation_loss is not None:
            print(f"Loss change: {previous_validation_loss - validation_loss}")
        previous_validation_loss = validation_loss

        avg_accuracy /= len(train_loader)
        print(f"Average accuracy: {avg_accuracy}")


def get_loader(df_csv_name: str):
    df_train = pd.read_csv(os.path.join(DIR_TO_SAVE_TO, df_csv_name))
    dataset = MelSpecDataset(df_train, DIR_TO_SAVE_TO, LABEL_TO_INDEX)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

if __name__ == "__main__":

    base_cnn = BaseNet(kernel_sizes=[3, 3, 3, 3], strides=[1, 1, 1, 1], paddings=[1, 1, 1, 1])

    BASE_NET_OUTPUT_SHAPE = summary(base_cnn.features, input_size = BASE_NET_INPUT_SHAPE, verbose=0).summary_list[-1].output_size
   
    mean_net = MeanNet(output_features=len(LABEL_TO_INDEX), input_features=BASE_NET_OUTPUT_SHAPE[1]*BASE_NET_OUTPUT_SHAPE[2])

    net = UnionNet(
        base_cnn,
        mean_net,
    )

    optimizer = Adam(net.parameters(), lr=0.001)
    loss = nn.NLLLoss()

    train_loader = get_loader("df_train.csv")
    validation_loader = get_loader("df_validation.csv")

    train(net, epochs=40, train_loader=train_loader, validate_loader=validation_loader, optimizer=optimizer, loss=loss)
