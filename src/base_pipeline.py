import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

class MelSpecDataset(Dataset):

    def __init__(self, MELSPEC_DATA_DIR, melspec_indexes):
        self.MELSPEC_DATA_DIR = MELSPEC_DATA_DIR
        self.melspec_paths = [os.path.join(self.MELSPEC_DATA_DIR,f) for f in os.listdir(self.MELSPEC_DATA_DIR) if f.endswith('.pt')]
        self.df_slice = pd.read_csv(os.path.join(MELSPEC_DATA_DIR, "DF.csv")).loc[melspec_indexes, :]

    def __len__(self):
        return len(self.df_slice)

    def __getitem__(self, index):
        melspec = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.df_slice.iloc[index, "path"]))
        label = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.df_slice.iloc[index, "label"]))
        return melspec, label



    