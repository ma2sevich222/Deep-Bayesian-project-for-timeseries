
import torch
from torch.utils.data import Dataset
class DBNataset(Dataset):
    def __init__(self, train_features, train_labels):

        self.x_train = torch.tensor(train_features, dtype=torch.float32)
        self.y_train = torch.tensor(train_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]