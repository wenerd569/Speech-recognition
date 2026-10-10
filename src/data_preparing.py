import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from common.settings import TrainSettings

AVG = 9.8932
STD = 210.8802

class MelSpecDataset(Dataset): 
    #датасет можно хранить на:
    # 1) SSD/HD - с постоянной выгрузкой батчей на videoram - в таком случае стоит подобрать размер батча максимально возможным, чтоб он влезал в память вашего девайса
    # 2) RAM - скорее всего любой из датасетов влезет, причём размер батча также можно выбрать большим, если вы считаете на cuda. на эплах видео память вроде объединена с ram, так что разгуляться в размером батча не выйдет
    # 3) VIDEORAM - на колаб получилось подгрузить только первый датасет, второй не влез. очевидно лучший варинат, не гоняющий данные туда-сюда. размер батча можно выбирать так, чтоб он с датасетом уместился в videoram. 
    # ! если вы юзаете cpu или mps, то ram и vram - суть одно и то же. однако мне показалось, что при использовани mps+ram лосс на валидации скачет сильнее, чем на mps+vram. в любом случае юзайте второй вариант если ноут позволяет.
    # ! если такая же проблема будет в колабе с cuda+ram - надо скорее всего фиксить. ну надеюсь мы арендуем карточку с достаточным vram и забьём на это
    # места для хранение - "storage", "ram", "vram"
    def __init__(self, df, MELSPEC_DATA_DIR, device, label_to_index=None, place_to_store_dataset = 'storage'):
        self.MELSPEC_DATA_DIR = MELSPEC_DATA_DIR
        self.df = df
        self.label_to_index = label_to_index or {
            label: index for index, label in enumerate(sorted(df["label"].drop_duplicates()))
        }


        # pyrefly: ignore [unsupported-operation]
        self.labels = torch.tensor([label_to_index[l] for l in df["label"]])
        self.paths = self.df["path"].tolist()
        self.place = place_to_store_dataset
        map_location = device if self.place == 'vram' else 'cpu'
        
        if self.place == 'ram':
            temp = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.paths[0]), map_location=map_location)
            shape = temp.shape
            dtype = temp.dtype
            N = len(self.paths)
            self.data = torch.empty((N,*shape), dtype = dtype)
            for i,p in enumerate(self.paths):
                self.data[i] = torch.load(os.path.join(self.MELSPEC_DATA_DIR,p), map_location=map_location)

        elif self.place == 'vram':
            temp = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.paths[0]), map_location=map_location)
            shape = temp.shape
            dtype = temp.dtype
            N = len(self.paths)
            self.data = torch.empty((N,*shape), dtype = dtype).to(device)
            for i,p in enumerate(self.paths):
                self.data[i] = torch.load(os.path.join(self.MELSPEC_DATA_DIR,p), map_location=map_location).to(device)
        elif self.place == 'storage':
            self.data = None
        else:
            print("uknown device")
            raise ValueError

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.place != "storage":
            # pyrefly: ignore [unsupported-operation]
            return (self.data[index] - AVG)/STD, self.labels[index]
        else:
            melspec = torch.load(os.path.join(self.MELSPEC_DATA_DIR, self.paths[index]),map_location="cpu")
            return (melspec - AVG)/STD, self.labels[index]


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

    return df_train, df_test, df_validation

if __name__ == "__main__":

    if len(sys.argv) > 1:
        path_settings = TrainSettings(sys.argv[1])
    else:
        path_settings = TrainSettings()

    init_datasets(path_settings.get_dir_to_upload_from(), path_settings.get_dir_to_save_to)
