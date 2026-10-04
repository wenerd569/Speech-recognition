import os
from src.dataset import MelSpecDataset 
from torch.utils.data import DataLoader


BATCH_SIZE = 32

DIR_CURRENT = os.path.dirname(os.path.abspath(__file__))
PREPARED_DDATA = os.path.join(os.path.dirname(DIR_CURRENT), "data/data_proccessed")

dataset = MelSpecDataset(PREPARED_DDATA, melspec_indexes=range(1000))


train_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)





