from src.base_pipeline import MyDataset as Dataset
form torch.utils.data import DataLoader, random_split
import torch.nn as nn


BATCH_SIZE = 32





train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42) 
)

train_losder = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

train_losder = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)




class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x