import os
import soundfile as sf
import pandas as pd
import torch
import torchaudio.transforms as T

DIR_CURRENT = os.path.dirname(os.path.abspath(__file__))
DIR_TO_UPLOAD_FROM = os.path.join(os.path.dirname(DIR_CURRENT), "data/speech_commands_v0.01")
DIR_TO_SAVE_TO = os.path.join(os.path.dirname(DIR_CURRENT), "data/data_proccessed")
DF_PATH = os.path.join(DIR_TO_SAVE_TO, "DF.csv")

Sample_rate = 16000
Duration = 1
N_fft = 1024
N_mels = 128
Hop_length = 128
Target_T = (Sample_rate//Hop_length) + 1

def transform_wavs_to_tensors(
        DIR_TO_UPLOAD_FROM, DIR_TO_SAVE_TO, sample_rate = Sample_rate,
        n_fft = N_fft, n_mels = N_mels, hop_length = Hop_length,
        logFlag: bool = True
):
    melspec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft = n_fft,
        hop_length=hop_length,
        n_mels = n_mels,
    )

    print("created melspectrogram model")
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

            wav_melspec = melspec_transform(wav_file)

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

    avg = total/n
    var = total_sq/n - (total/n)**2
    return avg, torch.sqrt(var)

if __name__ == "__main__":
    avg, std = transform_wavs_to_tensors(
        DIR_TO_UPLOAD_FROM,
        DIR_TO_SAVE_TO,
        Sample_rate,
        N_fft,
        N_mels,
        Hop_length
    )
    print(avg, std)
