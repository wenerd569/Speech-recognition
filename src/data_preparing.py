import torch
from torch import nn
from torch.utils.data import Dataset
import os
import pandas as pd

class MelSpecDataset(Dataset):

    def __init__(self, df, MELSPEC_DATA_DIR):
        self.MELSPEC_DATA_DIR = MELSPEC_DATA_DIR
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        melspec = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.df.iloc[index, "path"]))
        label = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.df.iloc[index, "label"]))
        return melspec, label
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
