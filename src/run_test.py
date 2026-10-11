import sys
import os

import argparse

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from common.settings import TrainSettings
from common.utils import get_device, get_loader, get_XY
from nn_modules.attention_net import AttentionCompleteNet
from nn_modules.base_net import BaseNet
from nn_modules.lstm_net import LSTMNet
from nn_modules.mean_net import MeanNet
from nn_modules.gru_net import GruNet
from nn_modules.transformer_net import TransformerNet
from nn_modules.ast import AudioTransformer

from nn_modules.union_net import UnionNet
from train_evaluate import evaluate, load_model

def eval_test(model, data_loader, device):
    class_true, class_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)

            class_true.append(y.cpu())
            class_pred.append(model(X).argmax(dim=1).cpu())
    return torch.cat(class_true), torch.cat(class_pred)

def bootstrap_acc(data, B = 10_000, alpha = 0.1):
    n = len(data)
    smpl_acc = torch.zeros(B)
    for i in range(B):
        sample = torch.randint(0, n, (n, ))
        smpl_acc[i] = data[sample].mean()

    low = smpl_acc.quantile(alpha / 2).item()
    upper = smpl_acc.quantile(1 - alpha / 2).item()
    acc = data.mean().item()

    return {
        "accuracy": acc,
        "lower_bound": low, 
        "upper_bound": upper
    }


def test_model(model_path, model_setting, device):
    num_classes = model_setting.get_num_classes()
    target_T, n_mels = get_XY(model_setting)

    nets = {
        "base15": lambda: BaseNet(kernel_sizes=[(1, 5), (1, 5)], channels=[10, 1]),
        "lstm64": lambda: LSTMNet(input_features=n_mels),
        "gru": lambda: GruNet(input_features=n_mels),
        "att128": lambda: AttentionCompleteNet(output_features=num_classes),
        "meannet": lambda: MeanNet(input_features=n_mels, output_features=num_classes),
        "transformer": lambda: TransformerNet(input_features=n_mels),
        "ast": lambda: AudioTransformer(num_classes=num_classes, input_f=n_mels, input_t=target_T),
        "dropout": lambda: nn.Dropout(0.3),
        "base_new_1": lambda: BaseNet(kernel_sizes=[(1,5),(1,5)], channels=[10, 20]),
        "base_new_2": lambda: BaseNet(kernel_sizes=[(1,5),(1,5)], channels=[10, 1]),
    }
    model = UnionNet([nets[x]() for x in model_setting.get_parts()])

    load_model(model, None, None, model_path, device)

    test_loader = get_loader("df_test.csv", device, model_setting)
    metrics = evaluate(model, test_loader, num_classes, nn.CrossEntropyLoss(), device)

    # bootstrap
    c_true, c_pred = eval_test(model, test_loader, device)
    bootstrap = bootstrap_acc((c_true == c_pred).float(), B = 10_000, alpha = 0.1)

    metrics.update(bootstrap)

    print(f'test accuracy: {metrics["accuracy"]} in [{metrics["lower_bound"]}, {metrics["upper_bound"]}] \n')
    print(f'test loss: {metrics["validation loss"]}')
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setting", help="path to model_setting.json")
    parser.add_argument("model", help="path to model parametrs (model.pt)")
    argv = parser.parse_args()

    device = get_device()
    print("device: ", device)

    model_setting = TrainSettings(argv.setting)
    model_path = argv.model

    test_model(model_path, model_setting, device)
