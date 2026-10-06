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
N_fft = 1024
N_mels = 128
Hop_length = 128
Target_T = Sample_rate // Hop_length + 1
#TODO: спросить что ставить в параметры
def transform_wavs_to_tensors(DIR_TO_UPLOAD_FROM, DIR_TO_SAVE_TO, sample_rate = Sample_rate, n_fft = N_fft, n_mels = N_mels, hop_length = Hop_length): #см документацию к параметрам

    melspec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft = n_fft, 
        hop_length=hop_length,
        n_mels = n_mels,
    )

    print("created melspectrogram model")
    os.makedirs(DIR_TO_SAVE_TO, exist_ok=True)

    rows = []

    for root, _, files in os.walk(DIR_TO_UPLOAD_FROM):
        print("checking root ", root)
        for f in files:
            if not f.endswith('.wav'):
                continue

            wav_path = os.path.join(root,f)

            wav_file, sample_rate = sf.read(wav_path, dtype="float32")

            wav_file = torch.from_numpy(wav_file)

            if wav_file.ndim == 1:
                wav_file = wav_file.unsqueeze(0)
            else:
                wav_file = wav_file.transpose(0, 1)
            
            out_name = f.replace(".wav", ".pt")
            label = os.path.basename(root)

            out_name = f"{label}/{out_name}"

            if wav_file.shape[0] > 1: #одноканальность делаем
                wav_file = torch.mean(wav_file, dim=0, keepdim=True)

            wav_melspec = melspec_transform(wav_file)

            if Target_T is not None: #выравнивание по длительности
                T_current = wav_melspec.shape[-1]
                if T_current < Target_T:
                    wav_melspec = torch.nn.functional.pad(wav_melspec, (0, Target_T - T_current))
                else:
                    wav_melspec = wav_melspec[..., :Target_T]

            os.makedirs(os.path.join(DIR_TO_SAVE_TO, label), exist_ok=True)
            torch.save(wav_melspec, os.path.join(DIR_TO_SAVE_TO, out_name))

            rows.append({"path": out_name, "label": label})
    print("creating df")
    df = pd.DataFrame(rows, columns=["path","label"])
    df.to_csv(DF_PATH, index=False)

if __name__ == "__main__":
    transform_wavs_to_tensors(DIR_TO_UPLOAD_FROM, DIR_TO_SAVE_TO, Sample_rate, N_fft, N_mels, Hop_length)