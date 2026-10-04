import torch
from torch import nn
from torch.utils.data import Dataset
import os
import pandas as pd

class MelSpecDataset(Dataset):

    def __init__(self, df, MELSPEC_DATA_DIR):
        self.MELSPEC_DATA_DIR = MELSPEC_DATA_DIR
        self.dataframe = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        melspec = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.df.iloc[index, "path"]))
        label = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.df.iloc[index, "label"]))
        return melspec, label

def init_datasets(DATASET_DIR, MELSPEC_DATA_DIR):

    testing_list = []
    validation_list = []

    with open(os.path.join(DATASET_DIR, "testing_list.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            testing_list.append(line[:-4] + ".pt")
    f.close()

    with open(os.path.join(DATASET_DIR, "validation_list.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            validation_list.append(line[:-4] + ".pt")
    f.close()

    df = pd.read_csv(os.path.join(MELSPEC_DATA_DIR), "DF.csv")
    df_train = df[df["path"] not in testing_list and df["path"] not in validation_list]
    df_test = df[df["path" in testing_list]]
    df_validation = df[df["path" in validation_list]]

    return df_train, df_test, df_validation

