from sympy import true
from common.settings import TrainSettings
import os
import soundfile as sf
import pandas as pd
import torch
import json
import torchaudio.transforms as T

from common.settings import TrainSettings



def get_melspec_transform_func(settings):

    settings = settings.copy()

    target_T = settings["sample_rate"] // settings["hop_length"] + 1
    settings["target_T"] = target_T
    settings["X"] = target_T
    settings["Y"] = settings["n_mels"]

    return T.MelSpectrogram(
        sample_rate=settings["sample_rate"],
        n_fft = settings["n_fft"], 
        hop_length= settings["hop_length"],
        n_mels = settings["n_mels"],
    ), settings


def get_mfcc_transform_func(settings):

    settings = settings.copy()
    target_T = settings["sample_rate"] // settings["hop_length"] + 1

    settings["target_T"] = target_T
    settings["X"] = target_T
    settings["Y"] = settings["n_mfcc"]

    
    return T.MFCC(
        sample_rate=settings["sample_rate"],
        n_mfcc=settings["n_mfcc"],
        melkwargs={
            "n_fft": settings["n_fft"],
            "hop_length": settings["hop_length"],
            "n_mels": settings["n_mels"]
        }
    ), settings




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
    settings["avg"] = avg
    settings["var"] = var

    with open(settings_file_name, "w") as file:
        json.dump(settings, file, indent=4)
    
    return avg, torch.sqrt(var)


if __name__ == "__main__":
    settings_melspec = {}
    
    settings_melspec["type"] = "melspec"
    settings_melspec["duration"] = 1
    settings_melspec["log_flag"] = True
    settings_melspec["sample_rate"] = 16000
    settings_melspec["n_mels"] = 128
    settings_melspec["n_fft"] = 1024
    settings_melspec["hop_length"] = 128

    settings_mfcc = {}

    settings_mfcc["type"] = "mfcc"
    settings_mfcc["log_flag"] = False
    settings_mfcc["duration"] = 1
    settings_mfcc["sample_rate"] = 16000
    settings_mfcc["n_mels"] = 80
    settings_mfcc["hop_length"] = 160
    settings_mfcc["n_fft"] = 400
    settings_mfcc["n_mfcc"] = 20

    path_settings = TrainSettings()
    
    transform_func, settings = get_melspec_transform_func(settings_melspec)
    avg, std = transform_wavs_to_tensors(path_settings, transform_func, settings)

    # transform_func, settings = get_melspec_transform_func(settings_mfcc)
    # avg, std = transform_wavs_to_tensors(path_settings, transform_func, settings)
    

    print(avg, std)