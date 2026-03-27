import numpy as np
import torch
from torch.utils.data import Dataset
from config import Y_MAX

class MultiFileSeismicDataset(Dataset):
    def __init__(self, file_pairs):
        """
        file_pairs = [
            ("seis_1.npy", "vel_1.npy"),
            ("seis_2.npy", "vel_2.npy"),
            ...
        ]
        """
        self.file_pairs = file_pairs
        self.index_map = []

        for file_idx, (x_path, y_path) in enumerate(file_pairs):
            x = np.load(x_path, mmap_mode='r')  
            n_samples = x.shape[0]

            for i in range(n_samples):
                self.index_map.append((file_idx, i))

        
        self.X_files = [np.load(p[0], mmap_mode='r') for p in file_pairs]
        self.Y_files = [np.load(p[1], mmap_mode='r') for p in file_pairs]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]

        x = self.X_files[file_idx][sample_idx]   # (5,1000,70)
        y = self.Y_files[file_idx][sample_idx]   # (1,70,70)

        # normalize
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = y / Y_MAX

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y.squeeze(0), dtype=torch.float32)

        return x, y