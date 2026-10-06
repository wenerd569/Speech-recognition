import torch
from torch import nn
from torch.utils.data import Dataset
import os
import pandas as pd

AVG = 9.8931
STD = 210.8841

class MelSpecDataset(Dataset):

    def __init__(self, df, MELSPEC_DATA_DIR, label_to_index=None, fast = True):
        self.MELSPEC_DATA_DIR = MELSPEC_DATA_DIR
        self.df = df
        self.label_to_index = label_to_index or {
            label: index for index, label in enumerate(sorted(df["label"].drop_duplicates()))
        }
        self.labels = torch.tensor([label_to_index[l] for l in df["label"]])
        self.paths = self.df["path"].tolist()
        self.fast = fast
        if fast:
            temp = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.paths[0]))
            shape = temp.shape
            dtype = temp.dtype
            N = len(self.paths)
            self.data = torch.empty((N,*shape), dtype = dtype)
            for i,p in enumerate(self.paths):
                self.data[i] = torch.load(os.path.join(self.MELSPEC_DATA_DIR,p))
            self.data = (self.data - AVG)/STD
        else:
            self.data = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.fast:
            return self.data[index], self.labels[index]
        else:
            melspec = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.paths[index]))
            return (melspec - AVG)/STD, self.labels[index]

#dataloadfer посмотреть, как сохранить в памяти весь датасет

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

    print(len(testing_list), len(validation_list))

    df = pd.read_csv(os.path.join(MELSPEC_DATA_DIR,"DF.csv"))
    df_train = df[~df["path"].isin(testing_list) & ~df["path"].isin(validation_list)]
    df_test  = df[df["path"].isin(testing_list)]
    df_validation = df[df["path"].isin(validation_list)]

    df_train.to_csv(os.path.join(MELSPEC_DATA_DIR, "df_train.csv"), index=False)
    df_test.to_csv(os.path.join(MELSPEC_DATA_DIR,"df_test.csv"), index=False)
    df_validation.to_csv(os.path.join(MELSPEC_DATA_DIR,"df_validation.csv"), index=False)

    print('created csvs')
    print(df_train.shape)
    print(df_test.shape)
    print(df_validation.shape)

    return df_train, df_test, df_validation


DIR_CURRENT = os.path.dirname(os.path.abspath(__file__))
DIR_TO_UPLOAD_FROM = os.path.join(os.path.dirname(DIR_CURRENT), "data/speech_commands_v0.01")
DIR_TO_SAVE_TO = os.path.join(os.path.dirname(DIR_CURRENT), "data/data_proccessed")

if __name__ == "__main__":
    init_datasets(DIR_TO_UPLOAD_FROM, DIR_TO_SAVE_TO)
