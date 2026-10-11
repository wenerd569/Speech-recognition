# Speech-recognition

## Start Jupyter:

```python
!git clone https://github.com/wenerd569/Speech-recognition/
%cd Speech-recognition
%pip install -r requirements.txt

import sys
sys.path.append("src")
import launcher as L

L.load_dataset()
run_dir = L.create_model(["base15", "lstm64", "att128"], epochs=40)
L.train_model(run_dir)
L.test_model(run_dir)
```

### Functions

| Function | What it does |
|---|---|
| `load_dataset(setting_path, feature="melspec")` | Downloads the dataset, computes features (`melspec`, `mfcc` or `lfcc`) and splits it into train / validation / test. Recomputes everything on every call. |
| `create_model(llm_heads, epochs=None, name=None, setting_path)` | Creates the model folder with its own `train_settings.json`: a copy of the settings with the heads, epochs and save path replaced. Folder name is `<date>_<name>`, `name` defaults to the heads joined by `-`. Returns the folder. |
| `train_model(run_dir)` | Trains through `origin_pipeline.py`, saves `history.csv` and `curves.png`. Stopping the cell keeps the finished epochs. |
| `test_model(model_dir)` | Runs `run_test.test_model` on the epoch with the best validation accuracy. |

`setting_path` defaults to `train_settings.json` in the repository root.

### Heads for `parts`

`B` — batch size, `F` — feature size, `T` — timeline.

| Head | Input | Output |
|---|---|---|
| `base15` | `B x 1 x F x T` | `B x F x T` |
| `lstm64` | `B x F x T` | `B x 128 x T` |
| `gru` | `B x F x T` | `B x 128 x T` |
| `transformer` | `B x F x T` | `B x F x T` |
| `att128` — *end* | `B x 128 x T` | `B x class` |
| `meannet` — *end* | `B x F x T` | `B x class` |
| `ast` — *complete model* | `B x 1 x F x T` | `B x class` |

### Model folder

```
Speech-recognition/
├╼ …
├╼ models/                                 # model_saving_path from train_settings.json
│  ╰╼ <date>_<name>/                       # one folder per create_model
│     ├╼ train_settings.json               # settings of this model
│     ├╼ model_<epoch>.pt                  # checkpoint after every epoch
│     ├╼ tensorboard/                      # training logs
│     ├╼ history.csv                       # loss and accuracy per epoch
│     ├╼ curves.png                        # plots of history.csv
│     ├╼ test_model_<epoch>.json           # test metrics
│     ╰╼ confusion_test_model_<epoch>.png  # test confusion matrix
╰╼ …
```

Test a checkpoint from the terminal: `python src/run_test.py <path>/train_settings.json <path>/model_<epoch>.pt`
