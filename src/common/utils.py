import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_evaluate import train

from data_preparing import MelSpecDataset


def get_loader(df_csv_name, device, train_settings):
    dir_to_save_to = train_settings.get_dir_to_save_to()
    label_to_index = train_settings.get_labels_to_index()
    batch_size = train_settings.get_batch_size()
    place = train_settings.get_place_to_store_dataset()

    df_train = pd.read_csv(os.path.join(dir_to_save_to, df_csv_name))
    dataset = MelSpecDataset(df_train, dir_to_save_to, device, label_to_index, place)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_XY(train_settings):
    preprocessing_settings_file_name = os.path.join(train_settings.get_dir_to_save_to(), "preprocess_settings.json")
    preprocessing_settings = None

    with open(preprocessing_settings_file_name, "r") as file:
        preprocessing_settings = json.load(file)

    target_T = preprocessing_settings["X"]
    n_mels = preprocessing_settings["Y"]
    
    return target_T, n_mels