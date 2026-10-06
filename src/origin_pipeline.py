import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

from train_evaluate import train

from data_preparing import MelSpecDataset
from base_net import BaseNet
from heads.lstm_net import LSTMNet
from heads.attention_net import AttentionCompleteNet
from union_net import UnionNet

from wav_preproccess import DIR_TO_SAVE_TO, DF_PATH
from wav_preproccess import N_mels, Target_T

BATCH_SIZE = 32
LABELS = sorted(pd.read_csv(DF_PATH)["label"].unique())
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
DIR_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_TO = os.path.join(DIR_PROJECT, "models/test.pt")
BASE_NET_INPUT_SHAPE = (32, 1, N_mels, Target_T)


def get_loader(df_csv_name: str):
    df_train = pd.read_csv(os.path.join(DIR_TO_SAVE_TO, df_csv_name))
    dataset = MelSpecDataset(df_train, DIR_TO_SAVE_TO, LABEL_TO_INDEX)
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

    BASE_NET_OUTPUT_SHAPE = summary(base_cnn.features, input_size = BASE_NET_INPUT_SHAPE, verbose=0).summary_list[-1].output_size
   
    lstm_net = LSTMNet(input_features=BASE_NET_OUTPUT_SHAPE[1]*BASE_NET_OUTPUT_SHAPE[2])
    attention_net = AttentionCompleteNet(output_features=NUM_CLASSES)


    net = UnionNet([base_cnn, lstm_net, attention_net])

    optimizer = Adam(net.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    train_loader = get_loader("df_train.csv")
    validation_loader = get_loader("df_validation.csv")

    train(net, epochs=40, train_loader=train_loader, validate_loader=validation_loader, optimizer=optimizer, loss=loss, num_classes=NUM_CLASSES, device=device, eps=0.001, epochs_to_wait=2, save_path=SAVE_TO, load_path=SAVE_TO)
