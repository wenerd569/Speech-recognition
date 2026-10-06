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
from heads.mean_net import MeanNet
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cnn = BaseNet(kernel_sizes=[(5,1),(5,1)], channels=[10,1])

    BASE_NET_OUTPUT_SHAPE = summary(base_cnn.features, input_size = BASE_NET_INPUT_SHAPE, verbose=0).summary_list[-1].output_size
   
    mean_net = MeanNet(output_features=len(LABEL_TO_INDEX), input_features=BASE_NET_OUTPUT_SHAPE[1]*BASE_NET_OUTPUT_SHAPE[2])

    net = UnionNet([base_cnn, mean_net])

    optimizer = Adam(net.parameters(), lr=0.001)
    loss = nn.NLLLoss()

    train_loader = get_loader("df_train.csv")
    validation_loader = get_loader("df_validation.csv")

    train(net, epochs=40, train_loader=train_loader, validate_loader=validation_loader, optimizer=optimizer, loss=loss, num_classes=NUM_CLASSES, device=device, eps=0.001, epochs_to_wait=2, save_path=SAVE_TO)
