import sys, os
import os.path as Path
import runpy
import json

import shutil
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common.settings import PROJECT_ROOT
from common.settings import TrainSettings
from common.utils import get_device


from run_test import test_model as run_test_model

SRC_DIR = Path.join(PROJECT_ROOT, "src")
BASIC_MODEL_SETTING = Path.join(PROJECT_ROOT, "train_settings.json")
LOGS_DIR = Path.join(PROJECT_ROOT, "logs")

def _run_main(script, *args):
    sys.argv = [script, *args]
    runpy.run_path(Path.join(SRC_DIR, script), run_name="__main__")

def load_dataset(setting_path=BASIC_MODEL_SETTING, feature="melspec"):
    _run_main("wav_preproccess.py", 
              "--settings", setting_path, 
              "--feature", feature
              )
    _run_main("data_preparing.py", setting_path)

def create_model(llm_heads, epochs=None, name=None, setting_path=BASIC_MODEL_SETTING):
    with open(setting_path, encoding="utf-8") as file:
        config = json.load(file)

        model_path = config["base_settings"]["model_saving_path"]
        model_dir = Path.join(
            PROJECT_ROOT,
            model_path,
            f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{name or '-'.join(llm_heads)}"
        ) # folder to store data
    os.makedirs(os.path.join(model_dir, "tensorboard"))

    # creating your own train_settings.json
    config["parts"] = llm_heads
    config["base_settings"]["model_saving_path"] = model_dir
    if epochs is not None:
        config["base_settings"]["epoch_count"] = epochs

    with open(os.path.join(model_dir, "train_settings.json"), "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)
    print(f"run folder: {model_dir}")

    return model_dir

# this part represents our American colleges)
def train_model(run_dir):
    os.makedirs(LOGS_DIR, exist_ok=True)
    logs_before = set(os.listdir(LOGS_DIR))
    try:
        _run_main("origin_pipeline.py", os.path.join(run_dir, "train_settings.json"))
    except KeyboardInterrupt:
        print("interrupted")

    for file_name in set(os.listdir(LOGS_DIR)) - logs_before:
        shutil.copy(os.path.join(LOGS_DIR, file_name), os.path.join(run_dir, "tensorboard"))

    events = EventAccumulator(os.path.join(run_dir, "tensorboard"))
    events.Reload()
    history = pd.DataFrame({tag.removesuffix("_epoch"): {e.step: e.value for e in events.Scalars(tag)}
                        for tag in events.Tags()["scalars"] if tag.endswith("_epoch")})
    history.to_csv(os.path.join(run_dir, "history.csv"), index_label="epoch")
    history.plot(subplots=[("Loss/train", "Loss/val")], figsize=(10, 6), grid=True, xlabel="epoch")
    
    plt.savefig(os.path.join(run_dir, "curves.png"))
    
    plt.show()
    return history


def test_model(model_dir, settings):
    # choose the best epoch
    history = pd.read_csv(Path.join(model_dir, "history.csv"), index_col="epoch")
    epoch = history["Accuracy/val"].idxmax()

    settings = TrainSettings(settings)
    metrics = run_test_model(Path.join(model_dir, f"model_{epoch}.pt"),
                         settings, 
                         get_device()
                         )

    with open(os.path.join(model_dir, f"test_model_{epoch}.json"), "w") as file:
        results = {}
        for key, value in metrics.items():
            if hasattr(value, "tolist"):
                value = value.tolist()
            results[key] = value
        json.dump(results, file, indent=4)

    #some pictures
    plt.figure(figsize=(14, 12))
    sb.heatmap(metrics["confusion matrix"], annot=True, fmt="d")
    plt.savefig(os.path.join(model_dir, f"confusion_test_model_{epoch}.png"))
    plt.show()
    
    return metrics


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("setting", help="path to model_setting.json")
        parser.add_argument("model", help="path to model parametrs (model.pt)")
        argv = parser.parse_args()
    
    
        model_setting = TrainSettings(argv.setting)
        model_path = argv.model
    
        test_model(model_path, model_setting)