import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

Q = torch.load("data/queries.pt")
C = torch.load("data/corpuses.pt")
dist = torch.load("data/sims.pt")


def seed_everything(seed):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
seed_everything(69)


import sys
sys.path.append("../neuts/")

import tools.sampling_methods as sm
from geo_rnns.wrloss import WeightedRankingLoss

class DTWDataset(Dataset):
    def __init__(self, Q, C, dist, per_q_expl=200):
        self.Q = Q
        self.C = C
        self.dist = dist
        self.per_q_expl = per_q_expl
        self.batches = []
        self.create_batches()

    def create_batches(self):
        for i in range(len(self.Q)):
            distances = self.dist[i]
            sortidx = torch.argsort(distances)
            top_distances = distances[sortidx[:self.per_q_expl]]
            bot_distances = distances[sortidx[-self.per_q_expl:]]

            batch = [i, sortidx[:self.per_q_expl], sortidx[-self.per_q_expl:], top_distances, bot_distances]
            self.batches.append(batch)

    def __len__(self):
        return len(self.Q)

    def __getitem__(self, idx):
        query_idx, pos_corpus_idxs, neg_corpus_idxs, pos_distances, neg_distances = self.batches[idx]
        query = self.Q[query_idx]
        pos_corpus = self.C[pos_corpus_idxs]
        neg_corpus = self.C[neg_corpus_idxs]
        return query, pos_corpus, neg_corpus, pos_distances, neg_distances
    
nepochs = 100
device = "cuda:0"
step = 1


d_model = Q.shape[-1] // step
xnhead = 2
xff = 1024
xnumlayers = 2
xoutdim = 128
xwarmupepochs = 10
loss_delta = 0.7

from transformers import get_linear_schedule_with_warmup, AdamW
model = nn.Sequential(
    PositionalEncoding(d_model=d_model, max_seq_length=20),
    nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=xnhead, batch_first=True, dim_feedforward=xff), xnumlayers),
    nn.Linear(d_model, xoutdim)
).to(device)

# model = nn.GRU(input_size=1000, hidden_size=128, num_layers=2, batch_first=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_size = 600
train_dataset = DTWDataset(Q[:train_size,:,::step], C[:train_size,:,::step], dist[:train_size,:train_size])
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = DTWDataset(Q[train_size:,:,::step], C[train_size:,:,::step], dist[train_size:,train_size:])
valloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

scheduler = get_linear_schedule_with_warmup(optimizer, xwarmupepochs * len(trainloader), nepochs * len(trainloader))

def embed(x):
    return model(x).mean(dim=1)
    # return model(x)[0][:,-1,:]

def loss_batch(q, pc, nc, pd, nd):
    q, pc, nc, pd, nd = q.to(device), pc.to(device), nc.to(device), pd.to(device), nd.to(device)

    qemb = embed(q).repeat_interleave(len(pc[0]), dim=0)
    pcemb = embed(pc[0])
    ncemb = embed(nc[0])

    pos_preds = F.pairwise_distance(qemb, pcemb)
    neg_preds = F.pairwise_distance(qemb, ncemb)

    # want pos_preds <= neg_preds - delta
    # want pos - neg + delta <= 0
    pos_minus_neg = (pos_preds.unsqueeze(0) - neg_preds.unsqueeze(1)).flatten()
    loss = F.relu(pos_minus_neg + loss_delta).mean()
    return loss

if __name__ == "__main__":
    es = 0
    best_loss = 1e9

    pbar = tqdm(range(nepochs))
    for epoch in pbar:
        model.train()
        losses = []
        for q, pc, nc, pd, nd in tqdm(trainloader, leave=False):
            loss = loss_batch(q, pc, nc, pd, nd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            if epoch == 0:
                pbar.set_postfix_str(f"Loss: {np.mean(losses):.6f}")
            else:
                pbar.set_postfix_str(f"ES: {es}, Loss: {np.mean(losses):.6f}, Val Loss: {val_loss:.6f}")

        val_losses = []
        with torch.no_grad():
            model.eval()
            for q, pc, nc, pd, nd in tqdm(valloader, desc="Validation", leave=False):
                loss = loss_batch(q, pc, nc, pd, nd)
                val_losses.append(loss.item())
                pbar.set_postfix_str(f"ES: {es}, Loss: {np.mean(losses):.6f}, Val Loss: {np.mean(val_losses):.6f}")
            val_loss = np.mean(val_losses)

            if val_loss <= best_loss-1e-5:
                best_loss = val_loss
                es = 0
                torch.save(model.cpu(), "model.pt")
            else:
                es += 1
                if es >= 15:
                    break

    import ipdb; ipdb.set_trace()