import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from train_evaluate import train, step_decay

from data_preparing import MelSpecDataset
from base_net import BaseNet
from heads.lstm_net import LSTMNet
from heads.attention_net import AttentionCompleteNet
from union_net import UnionNet

from wav_preproccess import DIR_TO_SAVE_TO, DF_PATH
from wav_preproccess import N_mels, Target_T





BATCH_SIZE = 64
LABELS = sorted(pd.read_csv(DF_PATH)["label"].unique())
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
DIR_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_TO = os.path.join(DIR_PROJECT, "models/test.pt")
SAVE_TO_2 = os.path.join(DIR_PROJECT, "models/test2.pt")
BASE_NET_INPUT_SHAPE = (32, 1, N_mels, Target_T)


def get_loader(df_csv_name, device, place):
    df_train = pd.read_csv(os.path.join(DIR_TO_SAVE_TO, df_csv_name))
    dataset = MelSpecDataset(df_train, DIR_TO_SAVE_TO, device, LABEL_TO_INDEX, place)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device: ", device)

    base_cnn = BaseNet(kernel_sizes=[(5,1),(5,1)], channels=[10,1])

    lstm_net = LSTMNet(input_features=N_mels)
    attention_net = AttentionCompleteNet(output_features=NUM_CLASSES)

    net = UnionNet([base_cnn, lstm_net, attention_net])

    optimizer = Adam(net.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch)/0.001)

    train_loader = get_loader("df_train.csv", device, "vram")
    validation_loader = get_loader("df_validation.csv", device, "vram")

    train(net, epochs=40, train_loader=train_loader, validate_loader=validation_loader, optimizer=optimizer, loss=loss, scheduler=scheduler, num_classes=NUM_CLASSES, device=device, eps=0.001, epochs_to_wait=2, save_path=SAVE_TO_2, load_path=None)
