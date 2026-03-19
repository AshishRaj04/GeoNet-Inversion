import numpy as np
import torch
from torch.utils.data import Dataset

class SeismicDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)

        self.y_max = self.Y.max()
        
        self.X = (self.X - self.X.mean()) / self.X.std()
        self.Y = self.Y / self.y_max

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx].squeeze(0), dtype=torch.float32)
        return x, y