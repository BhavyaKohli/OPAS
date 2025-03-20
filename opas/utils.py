import torch
import numpy as np

from tqdm import tqdm
from functools import partial
from time import perf_counter
from numpy.linalg import norm

tqdm = partial(tqdm, ncols=150)


def normalize(*args):
    ret = []
    for arg in args:
        ret.append(arg / arg.norm(dim=-1, keepdim=True))
    if len(ret) == 1: ret = ret[0]
    return ret


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value


def make_A(m, n):
    """
    Creates the cyclic A matrix with params `m` and `n`
    """
    Rm = torch.hstack([torch.eye(m), torch.zeros((m,n-m))]).float()
    A = torch.eye(m).float()
    A = (A - torch.vstack([torch.zeros(1,m),A[:-1]])) @ Rm
    return A


def get_opas_constants(M, N, device):
    A_mat = make_A(M, N).float().to(device)[:, :M]
    A_mat[0,0] = 0
    a_vec = torch.tensor([[1.2**i for i in range(N)]]).T.float().to(device)
    Rm_mat = torch.hstack([torch.eye(M), torch.zeros((M,N-M))]).float().to(device)

    return A_mat, a_vec, Rm_mat


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


class catchtime:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'
        print(self.readout)


def stagger_and_concat(model_inputs, num_stagger=1):
    # input of shape bmn OR bNmn
    if len(model_inputs.shape) == 3:
        model_inputs = model_inputs.unsqueeze(1)

    model_inputs_staggered = [model_inputs]
    for i in range(1,num_stagger+1,1):
        model_inputs_staggered.append(torch.dstack((model_inputs[:,:,i:,:], torch.zeros_like(model_inputs)[:,:,:i,:])))
    if num_stagger==0 : return torch.stack(model_inputs_staggered, dim=2)
    return torch.stack(model_inputs_staggered, dim=2).squeeze() 


@torch.no_grad()
def embed_image(model, x):
    x = x.to(next(model.parameters()).device)
    out = model.encoder(x) 
    out = out.flatten(start_dim=-3)
    return out


@torch.no_grad()
def embed_if_image_and_normalize(c, image_embed_model=None):
    if len(c.shape) != 3:   
        # b x n x c x h x w instead of b x n x d
        if image_embed_model is None:
            raise ValueError("Image embed model not provided")
        c = torch.stack([embed_image(image_embed_model, c_) for c_ in c])
    return normalize(c)


@torch.no_grad()
def embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model=None, inner_batch_size=800, aggregator=None, verbose=False):
    C = dataset.c
    if not isinstance(C, torch.Tensor):
        C = torch.from_numpy(C).float()    
    Cembed = []
    for batch in tqdm(range(0, len(C), inner_batch_size), disable=not verbose, leave=False, desc="Embedding..."):
        c = C[batch:batch+inner_batch_size].to(next(embed_model.parameters()).device)
        c = embed_if_image_and_normalize(c, image_embed_model)
        c = embed_model(preembed_model(c))
        c = normalize(c)
        if aggregator is not None:
            c = aggregator[1](c)
        Cembed.append(c)
    C = torch.vstack(Cembed)
    return C


# https://github.com/perrying/gumbel-sinkhorn/blob/master/utils/gumbel_sinkhorn_ops.py

def sinkhorn_norm(alpha: torch.Tensor, n_iter: int = 20):
    for _ in range(n_iter):
        alpha = alpha / alpha.sum(-1, keepdim=True)
        alpha = alpha / alpha.sum(-2, keepdim=True)
    return alpha


def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int =20):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 20, noise: bool = True):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)/tau
    else : log_alpha /= tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix

def gen_assignment(cost_matrix):
    row, col = linear_sum_assignment(cost_matrix)
    np_assignment_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return np_assignment_matrix

def gumbel_matching(log_alpha : torch.Tensor, noise: bool = True):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)
    np_log_alpha = log_alpha.detach().to("cpu").numpy()
    np_assignment_matrices = [gen_assignment(-x) for x in np_log_alpha]
    np_assignment_matrices = np.stack(np_assignment_matrices, 0)
    assignment_matrices = torch.from_numpy(np_assignment_matrices).float().to(log_alpha.device)
    return assignment_matrices