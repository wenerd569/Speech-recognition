import sys
import os

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

from nn_modules.union_net import UnionNet
from train_evaluate import step_decay_mult, train

if __name__ == "__main__":

    #if len(sys.argv) != 2:
    #    raise Exception("1 аргумент - путь до файла настроек")

    device = get_device()
    print("device: ", device)

    if len(sys.argv) > 1:
        train_settings = TrainSettings(sys.argv[1])
    else:
        train_settings = TrainSettings()
        
    num_classes = train_settings.get_num_classes()
    
    target_T, n_mels = get_XY(train_settings)

    nets = {
        "base15": lambda: BaseNet(kernel_sizes=[(1, 5), (1, 5)], channels=[10, 1]),
        "lstm64": lambda: LSTMNet(input_features=n_mels),
        "gru": lambda: GruNet(input_features=n_mels),
        "att128": lambda: AttentionCompleteNet(output_features=num_classes),
        "meannet": lambda: MeanNet(input_features=n_mels, output_features=num_classes),
        "transformer": lambda: TransformerNet(input_features=n_mels),
    }

    parts_names = train_settings.get_parts()
    parts = [nets[x]() for x in parts_names]

    net = UnionNet(parts)

    optimizer = Adam(net.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=step_decay_mult)


    train_loader = get_loader("df_train.csv", device, train_settings)
    validation_loader = get_loader("df_validation.csv", device, train_settings)

    train(net, epochs=train_settings.get_epoch_count(), train_loader=train_loader, validate_loader=validation_loader, optimizer=optimizer, loss=loss, scheduler=scheduler, num_classes=num_classes, device=device, eps=0.001, epochs_to_wait=2, save_path=train_settings.get_model_saving_path(), load_path=None)