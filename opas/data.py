import h5py
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class DummyDataset:
    def __init__(self, arr):
        self.c = arr


class PairDatasetTrain(Dataset):
    def __init__(self, filepath, num_q=300, negative_exploration=800, seed=15):
        super().__init__()
        self.data = h5py.File(filepath, "r")

        np.random.seed(seed)
        if num_q == -1: 
            num_q = len(self.data[f"labels"])

        self.qlocs = np.random.choice(range(len(self.data['Q'])), size=num_q, replace=False)
        self.q = self.data['Q'][:][self.qlocs]
        self.l = self.data['labels'][:][self.qlocs]
        self.c = self.data['C'][:]

        self.exploration = []
        for i in range(len(self.q)):
            q, l = self.q[i], self.l[i]
            remaining_idx = list(set(range(len(self.c))) - set(l))
            exploration = list(l) + np.random.choice(remaining_idx, size=negative_exploration, replace=False).tolist()
            exploration = np.random.choice(exploration, size=len(exploration), replace=False).tolist()
            self.exploration.append(exploration)
        self.exploration = np.stack(self.exploration)

        self.clen = self.exploration.shape[1]

    def get_pair_ids(self, idx):
        id1 = idx // self.clen
        id2 = idx - self.clen * (id1)
        return id1, id2

    def get_dataloader(self, batch_size, shuffle=True, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=8, **kwargs)

    def __getitem__(self, idx):
        qloc, cloc = self.get_pair_ids(idx)
        cloc = self.exploration[qloc][cloc]
        q, c = self.q[qloc], self.c[cloc]
        l = self.l[qloc]

        label = 1 if cloc in l else 0
        
        q, c = torch.from_numpy(q).float(), torch.from_numpy(c).float()
        return q, c, label
    
    def get_positive_samples(self, num_samples=1):
        positive_samples = np.random.randint(len(self.q), size=num_samples)
        q, l = self.q[positive_samples], self.l[positive_samples]
        # l has 16 indices, need to pick one
        l = [np.random.choice(l_, p=[0.1] + [0.9/(len(l_)-1)]*(len(l_)-1)) for l_ in l]
        c = np.stack([self.c[l_] for l_ in l])
        label = torch.ones(num_samples)

        q, c = torch.from_numpy(q).float(), torch.from_numpy(c).float()
        q, c = torch.atleast_3d(q), torch.atleast_3d(c)
        return q, c, label

    def __len__(self):
        return len(self.q) * self.clen
    

class PairDatasetTest(Dataset):
    def __init__(self, filepath):
        super().__init__()
        self.data = h5py.File(filepath, "r")

        self.q = self.data['Q'][:]
        self.l = self.data['labels'][:]
        self.c = self.data['C'][:]

        self.clen = len(self.c)

    def get_dataloader(self, batch_size, shuffle=True, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=8, **kwargs)

    def __getitem__(self, idx):
        q, l = self.q[idx], self.l[idx]
        c = np.stack([self.c[l_] for l_ in l])
        
        q, c = torch.from_numpy(q).float(), torch.from_numpy(c).float()
        return q, F.one_hot(torch.tensor(l), num_classes=self.clen).sum(dim=0)

    def __len__(self):
        return len(self.q)