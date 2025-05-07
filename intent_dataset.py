import torch
from torch.utils.data import Dataset

class IntentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)  # input word indices
        self.y = torch.tensor(y, dtype=torch.long)  # intent class indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
