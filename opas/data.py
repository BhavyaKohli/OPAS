import h5py
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .utils import embed_full_corpus, get_opas_constants, gumbel_sinkhorn, stagger_and_concat, AttributeDict, tqdm


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
        self.lonehot = F.one_hot(torch.from_numpy(self.l), num_classes=self.clen).sum(dim=1)

    def get_dataloader(self, batch_size, shuffle=True, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=8, **kwargs)

    def __getitem__(self, idx):
        q, l = self.q[idx], self.l[idx]
        c = np.stack([self.c[l_] for l_ in l])
        
        q, c = torch.from_numpy(q).float(), torch.from_numpy(c).float()
        return q, F.one_hot(torch.tensor(l), num_classes=self.clen).sum(dim=0)

    def __len__(self):
        return len(self.q)
    

@torch.no_grad()
def get_all_pair_scores(q_emb, c_emb, models, args):
    model, scoremodel = models
    device = next(model.parameters()).device

    A_mat, a_vec, Rm_mat = get_opas_constants(M=q_emb[0].shape[0], N=c_emb[0].shape[0], device=device)
    CFG = AttributeDict({
        'tau': 1,
        'n_sink_iter': getattr(args, "n_sink_iter", 20),
        'n_samples': 1,
    })
    lamwt = getattr(args, "lamwt", 1e-2)
    b = getattr(args, "b", 1)

    qcscores = []
    _batch_size = 100
    _inner_loader = range(0, len(q_emb), _batch_size)
    for i, idx in enumerate(tqdm(_inner_loader, desc="Computing scores", leave=False)):
        q = q_emb[idx:idx+_batch_size]
        qct = torch.einsum("bmd,Nnd->bNmn", q, c_emb)     # verified
        model_inputs = stagger_and_concat(qct, num_stagger=0)   # !! hardcoded 0 stagger !!
        model_inputs = model_inputs.squeeze(1)                  # !! hardcoded 0 stagger !!
        lambdas = torch.stack([model(x) for x in model_inputs])
        
        P = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))
        P = gumbel_sinkhorn(P, CFG.tau, CFG.n_sink_iter, noise=False)

        RmPC = Rm_mat @ P @ c_emb.squeeze(-1)
        lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
        normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

        allscores = torch.stack([lamscore, normscore], dim=2)
        netscore = 2*scoremodel(-allscores).squeeze()      # b
        qcscores.append(netscore.to('cpu'))
        
        del qct, P, RmPC, q, model_inputs, lambdas, lamscore, normscore, netscore, allscores
    torch.cuda.empty_cache()
    return torch.vstack(qcscores).cpu()
    

class PairDatasetTrainHPlane(PairDatasetTrain):
    def __init__(self, filepath, models, args, num_q=300, negative_exploration=800, seed=15):
        # models: (model, scoremodel, embed_model, preembed_model, hasher)
        super().__init__(filepath, num_q, negative_exploration, seed)

        model, scoremodel, embed_model, preembed_model, hasher = models
        with torch.no_grad():
            q_emb = embed_full_corpus(DummyDataset(self.q), embed_model, preembed_model, inner_batch_size=200, verbose=True)
            c_emb = embed_full_corpus(DummyDataset(self.c), embed_model, preembed_model, inner_batch_size=200, verbose=True)
            # q_emb, c_emb on device
            self.qcscores = get_all_pair_scores(q_emb, c_emb, models=[model, scoremodel], args=args)
            self.q = hasher[0](q_emb).cpu()
            self.c = hasher[1](c_emb).cpu()

            del q_emb, c_emb
            torch.cuda.empty_cache()

    def __getitem__(self, idx):
        qloc, cloc = self.get_pair_ids(idx)
        cloc = self.exploration[qloc][cloc]
        q, c = self.q[qloc], self.c[cloc]
        l = self.l[qloc]

        label = 1 if cloc in l else 0
        return q, c, self.qcscores[qloc][cloc]

    def get_positive_samples(self, num_samples=2):
        if num_samples == 1:
            raise ValueError("num_samples must be greater than 1")
        positive_samples = np.random.randint(len(self.q), size=num_samples)
        q, l = self.q[positive_samples], self.l[positive_samples]
        # l has 16 indices, need to pick one
        l = [np.random.choice(l_, p=[0.1] + [0.9/(len(l_)-1)]*(len(l_)-1)) for l_ in l]
        c = torch.stack([self.c[l_] for l_ in l])
        label = torch.ones(num_samples)

        scores = torch.tensor([self.qcscores[qloc][cloc] for qloc, cloc in zip(positive_samples, l)])
        return q, c, scores


class PairDatasetTestHPlaneSampled(PairDatasetTrainHPlane):
    def __init__(self, filepath, models, args, negative_exploration, seed=15, num_q=-1):
        super().__init__(filepath, models, args, num_q=num_q, negative_exploration=negative_exploration, seed=seed)


class PairDatasetTestHPlane(PairDatasetTest):
    def __init__(self, filepath, models, args):
        super().__init__(filepath)
        model, scoremodel, embed_model, preembed_model, hasher = models

        model, scoremodel, embed_model, preembed_model, hasher = models
        with torch.no_grad():
            q_emb = embed_full_corpus(DummyDataset(self.q), embed_model, preembed_model, inner_batch_size=200, verbose=True)
            c_emb = embed_full_corpus(DummyDataset(self.c), embed_model, preembed_model, inner_batch_size=200, verbose=True)
            # q_emb, c_emb on device
            self.qcscores = get_all_pair_scores(q_emb, c_emb, models=[model, scoremodel], args=args)
            self.q = hasher[0](q_emb).cpu()
            self.c = hasher[1](c_emb).cpu()

            del q_emb, c_emb
            torch.cuda.empty_cache()
        
    def __getitem__(self, idx):
        q, l = self.q[idx], self.l[idx]
        # c = np.stack([self.c[l_] for l_ in l])
        
        return q, F.one_hot(torch.tensor(l), num_classes=self.clen).sum(dim=0), self.qcscores[idx]


class PairDatasetTrainHPlaneQCSc(PairDatasetTest):
    def __init__(self, filepath, models, args):
        super().__init__(filepath)
        model, scoremodel, embed_model, preembed_model, hasher = models

        model, scoremodel, embed_model, preembed_model, hasher = models
        with torch.no_grad():
            q_emb = embed_full_corpus(DummyDataset(self.q), embed_model, preembed_model, inner_batch_size=200, verbose=True)
            c_emb = embed_full_corpus(DummyDataset(self.c), embed_model, preembed_model, inner_batch_size=200, verbose=True)
            # q_emb, c_emb on device
            self.qcscores = get_all_pair_scores(q_emb, c_emb, models=[model, scoremodel], args=args)
            self.q = hasher[0](q_emb).cpu()
            self.c = hasher[1](c_emb).cpu()

            del q_emb, c_emb
            torch.cuda.empty_cache()
        
    def __getitem__(self, idx):
        q, l = self.q[idx], self.l[idx]
        # c = np.stack([self.c[l_] for l_ in l])
        
        return q, F.one_hot(torch.tensor(l), num_classes=self.clen).sum(dim=0), self.qcscores[idx]