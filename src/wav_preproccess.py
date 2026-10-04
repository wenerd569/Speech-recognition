import torch
import torchaudio.transforms as T
import torchaudio
import os
import pandas as pd

DIR_CURRENT = os.path.dirname(os.path.abspath(__file__))
DIR_TO_UPLOAD_FROM = os.path.join(os.path.dirname(DIR_CURRENT), "data/speech_commands_v0.01")
DIR_TO_SAVE_TO = os.path.join(os.path.dirname(DIR_CURRENT), "data/data_proccessed")
DF_PATH = os.path.join(DIR_TO_SAVE_TO, "DF.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO: спросить что ставить в параметры
def transform_wavs_to_tensors(DIR_TO_UPLOAD_FROM, DIR_TO_SAVE_TO, sample_rate = 16000, n_fft = 1024, n_mels = 128): #см документацию к параметрам

    target_T = (1 * sample_rate) // (n_fft // 2) + 1
    
    melspec_transform = T.MelSpectrogram(
        n_fft = n_fft, 
        n_mels = n_mels
    )

    print("created melspectrogram model")
    os.makedirs(DIR_TO_SAVE_TO, exist_ok=True)

    rows = []
    seen = set()

    for root, _, files in os.walk(DIR_TO_UPLOAD_FROM):
        print("checking root ", root)
        for f in files:
            if not f.endswith('.wav'):
                continue

            out_name = f.replace(".wav", ".pt")
            if out_name in seen:
                continue
            else:
                seen.add(out_name) #убираем дупликаты файлов в разных папках лол

            wav_path = os.path.join(root,f)
            label = os.path.basename(root)

            wav_file, _ = torchaudio.load(wav_path)

            if wav_file.shape[0] > 1: #одноканальность делаем
                wav_file = torch.mean(wav_file, dim=0, keepdim=True)

            wav_melspec = melspec_transform(wav_file)

            if target_T is not None: #выравнивание по длительности
                T_current = wav_melspec.shape[-1]
                if T_current < target_T:
                    wav_melspec = torch.nn.functional.pad(wav_melspec, (0, target_T - T_current))
                else:
                    wav_melspec = wav_melspec[..., :target_T]


            torch.save(wav_melspec, os.path.join(DIR_TO_SAVE_TO, out_name))

            rows.append({"path": out_name, "label": label})
    print("creating df")
    df = pd.DataFrame(rows, columns=["path","label"])
    df.to_csv(DF_PATH, index=False)

if __name__ == "__main__":
    transform_wavs_to_tensors(DIR_TO_UPLOAD_FROM, DIR_TO_SAVE_TO)
