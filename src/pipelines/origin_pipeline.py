from sympy.printing.tree import tree
import sys
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from train_evaluate import train, step_decay

from base_net import BaseNet
from heads.lstm_net import LSTMNet
from heads.attention_net import AttentionCompleteNet
from union_net import UnionNet

from settings import TrainSettings
from base_pipeline import get_device, get_XY, get_loader

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise Exception("1 аргумент - путь до файла настроек")
    
    settings_file_name = sys.argv[1]

    device = get_device()
    print("device: ", device)
    
    train_settings = TrainSettings(settings_file_name)
    num_classes = train_settings.get_num_classes()
    
    target_T, n_mels = get_XY(train_settings)


    base_cnn = BaseNet(kernel_sizes=[(5,1),(5,1)], channels=[10,1])

    lstm_net = LSTMNet(input_features=n_mels)
    attention_net = AttentionCompleteNet(output_features=num_classes)

    net = UnionNet([base_cnn, lstm_net, attention_net])

    optimizer = Adam(net.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch)/0.001)


    train_loader = get_loader("df_train.csv", device, train_settings)
    validation_loader = get_loader("df_validation.csv", device, train_settings)

    train(net, epochs=train_settings.get_epoch_count(), train_loader=train_loader, validate_loader=validation_loader, optimizer=optimizer, loss=loss, scheduler=scheduler, num_classes=num_classes, device=device, eps=0.001, epochs_to_wait=2, save_path=train_settings.get_model_saving_path(), load_path=None)
