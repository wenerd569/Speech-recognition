import json
import os
import argparse

import pandas as pd
import soundfile as sf

import torch
import torchaudio.transforms as T
import torchaudio

from common.settings import TrainSettings

# пик трансформера датасета
def make_melspec(settings):
    raw_mel = T.MelSpectrogram(
        sample_rate=settings["sample_rate"],
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        n_mels=settings["n_mels"],
    )
    def log_mel(x):
        return T.AmplitudeToDB(stype="power")(raw_mel(x))
    return log_mel if settings["log_flag"] else raw_mel



def make_mfcc(settings):
    return T.MFCC(
        log_mels = True,
        sample_rate=settings["sample_rate"],
        n_mfcc=settings["n_mfcc"],
        log_mels=settings["log_flag"],
        melkwargs={
            "n_fft": settings["n_fft"],
            "hop_length": settings["hop_length"],
            "n_mels": settings["n_mels"],
        },
    )

def make_lfcc(settings):
    return T.LFCC(
        sample_rate=settings["sample_rate"],
        n_lfcc=settings["n_lfcc"],
        log_lf=settings["log_flag"],
        n_filter=settings["n_filter"],
        speckwargs={
            "n_fft": settings["n_fft"],
            "hop_length": settings["hop_length"],
        },
    )

transformer_funcs = {
    "melspec": make_melspec,
    "mfcc": make_mfcc,
    "lfcc": make_lfcc
}

# настройка сеттингов
def set_melspec(settings):
    target_T = settings["sample_rate"] // settings["hop_length"] + 1
    return target_T, settings["n_mels"]

def set_mfcc(settings):
    target_T = settings["sample_rate"] // settings["hop_length"] + 1
    return target_T, settings["n_mfcc"]

def set_lfcc(settings):
    target_T = settings["sample_rate"] // settings["hop_length"] + 1
    return target_T, settings["n_lfcc"]

transformer_settings = {
    "melspec": set_melspec,
    "mfcc": set_mfcc,
    "lfcc": set_lfcc
}

def get_transform_func(settings: dict):
    settings = settings.copy()
    tp = settings["type"]

    target_T, Y = transformer_settings[tp](settings)
    settings["target_T"] = target_T
    settings["X"] = target_T
    settings["Y"] = Y

    return transformer_funcs[tp](settings), settings

def transform_wavs_to_tensors(path_settings: TrainSettings, transform_func, settings): #см документацию к параметрам
    DIR_TO_UPLOAD_FROM = path_settings.get_dir_to_upload_from()
    DIR_TO_SAVE_TO = path_settings.get_dir_to_save_to()
    DF_PATH = path_settings.get_df_path()
    Sample_rate = settings["sample_rate"]
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

def load_settings(
    path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "wav_preproccess_settings.json"
    )
):
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SPEECHCOMMANDS and pick feature settings")
    parser.add_argument(
        "--settings", "-s",
        default=None,
        help="Path to train settings (optional)",
    )
    parser.add_argument(
        "--url", "-u",
        default=None,
        help="Dataset URL (optional)",
    )
    parser.add_argument(
        "--feature", "-f",
        default="melspec",
        choices=["melspec", "mfcc", "lfcc"],
        help="Which feature settings to use (default: melspec)",
    )

    args = parser.parse_args()
    path_settings = TrainSettings(args.settings) if args.settings else TrainSettings()
    url = args.url or path_settings.get_dataset_url()
    root = os.path.dirname(os.path.dirname(path_settings.get_dir_to_upload_from()))
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, url=url, download=True)
    print("downloaded dataset")

    settings = load_settings()
    print(settings)

    key = f"{args.feature}_settings"
    print(f"using feature key: {key}")

    # Достаём нужные настройки
    feature_settings = settings[key]
    print(feature_settings)

    transform_func, settings = get_transform_func(settings[key])
    avg, std = transform_wavs_to_tensors(path_settings, transform_func, settings)

    print(avg, std)
