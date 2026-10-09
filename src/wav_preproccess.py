from settings import TrainSettings
import os
import soundfile as sf
import pandas as pd
import torch
import json
import torchaudio.transforms as T

from settings import TrainSettings



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

    print(f"created {type(transform_func)} model")
    os.makedirs(DIR_TO_SAVE_TO, exist_ok=True)

    rows = []

    total = 0
    total_sq = 0
    n = 0


    settings_file_name = os.path.join(DIR_TO_SAVE_TO, "preprocess_settings.json")
    with open(settings_file_name, "w") as file:
        json.dump(settings, file, indent=4)
    

    for root, _, files in os.walk(DIR_TO_UPLOAD_FROM):
        print("checking root ", root)
        for f in files:
            if not f.endswith('.wav'):
                continue

            wav_path = os.path.join(root,f)

            wav_file, sample_rate = sf.read(wav_path, dtype="float32")

            wav_file = torch.from_numpy(wav_file)
            wav_file = torch.nn.functional.pad(wav_file, (0, Sample_rate - wav_file.numel()), value=0.0)

            if wav_file.ndim == 1:
                wav_file = wav_file.unsqueeze(0)
            else:
                wav_file = wav_file.transpose(0, 1)
            
            out_name = f.replace(".wav", ".pt")
            label = os.path.basename(root)

            out_name = f"{label}/{out_name}"

            if wav_file.shape[0] > 1: #одноканальность делаем
                wav_file = torch.mean(wav_file, dim=0, keepdim=True)

            wav_melspec = transform_func(wav_file)
            total += torch.sum(wav_melspec.flatten())
            total_sq += torch.sum(wav_melspec.flatten() ** 2)
            n += wav_melspec.numel()

            os.makedirs(os.path.join(DIR_TO_SAVE_TO, label), exist_ok=True)
            torch.save(wav_melspec, os.path.join(DIR_TO_SAVE_TO, out_name))
            rows.append({"path": out_name, "label": label})

    print("creating df")
    df = pd.DataFrame(rows, columns=["path","label"])
    df.to_csv(DF_PATH, index=False)




    avg = total/n
    var = total_sq/n - (total/n)**2
    return avg, torch.sqrt(var)


if __name__ == "__main__":
    settings_melspec = {}
    
    settings_melspec["type"] = "melspec"
    settings_melspec["sample_rate"] = 16000
    settings_melspec["n_mels"] = 128
    settings_melspec["n_fft"] = 1024
    settings_melspec["hop_length"] = 128

    settings_mfcc = {}

    settings_mfcc["type"] = "mfcc"
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
