import json
import os
import sys

import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
import torchaudio

from common.settings import TrainSettings

# пик трансформера датасета
def make_melspec(settings):
    return T.MelSpectrogram(
        sample_rate=settings["sample_rate"],
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        n_mels=settings["n_mels"],
    )

def make_mfcc(settings):
    return T.MFCC(
        log_mels = True,
        sample_rate=settings["sample_rate"],
        n_mfcc=settings["n_mfcc"],
        melkwargs={
            "n_fft": settings["n_fft"],
            "hop_length": settings["hop_length"],
            "n_mels": settings["n_mels"],
        },
    )

transformer_funcs = {
    "melspec":  make_melspec,
    "mfcc": make_mfcc,
}

# настройка сеттингов
def set_melspec(settings):
    target_T = settings["sample_rate"] // settings["hop_length"] + 1
    return target_T, settings["n_mels"]

def set_mfcc(settings):
    target_T = settings["sample_rate"] // settings["hop_length"] + 1
    return target_T, settings["n_mfcc"]

transformer_settings = {
    "melspec":  set_melspec,
    "mfcc": set_mfcc,
}

def get_transform_func(settings: dict):
    settings = settings.copy()
    tp = settings["type"]

    target_T, Y = transformer_settings[tp](settings)
    settings["target_T"] = target_T
    settings["X"] = target_T
    settings["Y"] = Y

    return transformer_funcs[tp](settings), settings

#TODO: спросить что ставить в параметры
def transform_wavs_to_tensors(path_settings: TrainSettings, transform_func, settings): #см документацию к параметрам
    DIR_TO_UPLOAD_FROM = path_settings.get_dir_to_upload_from()
    DIR_TO_SAVE_TO = path_settings.get_dir_to_save_to()
    DF_PATH = path_settings.get_df_path()
    Sample_rate = settings["sample_rate"]
    logFlag = settings["log_flag"]
    Duration = settings["duration"]

    print(f"created {type(transform_func)} model")
    os.makedirs(DIR_TO_SAVE_TO, exist_ok=True)
    rows = []
    total, total_sq, n = 0, 0, 0

    for root, _, files in os.walk(DIR_TO_UPLOAD_FROM):
        print("checking root ", root)
        for f in files:
            if not f.endswith('.wav'):
                continue

            wav_path = os.path.join(root,f)
            wav_file = torch.from_numpy(sf.read(wav_path, dtype="float32")[0])
            wav_file = torch.nn.functional.pad(
                wav_file,(0, Sample_rate * Duration - wav_file.numel()), value=0.0
            )   # Padding crops the tensor, if difference is negative.

            if wav_file.ndim == 1:
                wav_file = wav_file.unsqueeze(0)
            else:
                wav_file = wav_file.transpose(0, 1)

            label = os.path.basename(root)
            out_name = f"{label}/{f.replace(".wav", ".pt")}"

            if wav_file.shape[0] > 1: #одноканальность делаем
                wav_file = torch.mean(wav_file, dim=0, keepdim=True)

            wav_melspec = transform_func(wav_file)

            if logFlag:
                wav_melspec = T.AmplitudeToDB(stype='power', top_db=80)(wav_melspec)

            total += torch.sum(wav_melspec.flatten())
            total_sq += torch.sum(wav_melspec.flatten() ** 2)
            n += wav_melspec.numel()

            os.makedirs(os.path.join(DIR_TO_SAVE_TO, label), exist_ok=True)
            torch.save(wav_melspec, os.path.join(DIR_TO_SAVE_TO, out_name))
            rows.append({"path": out_name, "label": label})

    print("creating df")
    df = pd.DataFrame(rows, columns=["path","label"])
    df.to_csv(DF_PATH, index=False)


    settings_file_name = os.path.join(DIR_TO_SAVE_TO, "preprocess_settings.json")
    avg = total / n
    var = total_sq/n - (total/n)**2
    settings["avg"] = float(avg)
    settings["var"] = float(var)

    with open(settings_file_name, "w") as file:
        json.dump(settings, file, indent=4)
    
    # pyrefly: ignore [bad-argument-type]
    return avg, torch.sqrt(var)

def load_settings(path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"wav_preproccess_settings.json")):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path_settings = TrainSettings(sys.argv[1])
    else:
        path_settings = TrainSettings()

    url = sys.argv[2] if len(sys.argv) > 2 else path_settings.get_dataset_url()
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=os.path.dirname(os.path.dirname(path_settings.get_dir_to_upload_from())), url = url, download=True)
    print('downloaded dataset')
    settings = load_settings()
    print(settings)
    # melspec_setings or mfcc_setings
    key = sys.argv[3] if len(sys.argv) > 3 else "melspec_settings"

    
    transform_func, settings = get_transform_func(settings[key])
    avg, std = transform_wavs_to_tensors(path_settings, transform_func, settings)

    # transform_func, settings = get_melspec_transform_func(settings_mfcc)
    # avg, std = transform_wavs_to_tensors(path_settings, transform_func, settings)

    print(avg, std)
