import os
import torch.nn as nn
from torch.utils.data import DataLoader

from base_pipeline import MelSpecDataset 
from torch.testing._internal.data.network1 import Net
from base_net import BaseNet



BATCH_SIZE = 32

DIR_CURRENT = os.path.dirname(os.path.abspath(__file__))
PREPARED_DDATA = os.path.join(os.path.dirname(DIR_CURRENT), "data/data_proccessed")

dataset = MelSpecDataset(PREPARED_DDATA, melspec_indexes=range(1000))


train_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

net = BaseNet(kernel_size=3)

for batch_idx, (data, target) in enumerate(train_loader):
    output = net(data)
    # TODO: добавить loss и backward
    print(f"Batch {batch_idx}")

