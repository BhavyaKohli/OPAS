import numpy as np
from numpy.linalg import norm
import torch


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


# https://github.com/perrying/gumbel-sinkhorn/blob/master/utils/gumbel_sinkhorn_ops.py

def sinkhorn_norm(alpha: torch.Tensor, n_iter: int = 20) -> (torch.Tensor,):
    for _ in range(n_iter):
        alpha = alpha / alpha.sum(-1, keepdim=True)
        alpha = alpha / alpha.sum(-2, keepdim=True)
    return alpha


def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int =20) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 20, noise: bool = True) -> (torch.Tensor,):
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

def gumbel_matching(log_alpha : torch.Tensor, noise: bool = True) -> (torch.Tensor,):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)
    np_log_alpha = log_alpha.detach().to("cpu").numpy()
    np_assignment_matrices = [gen_assignment(-x) for x in np_log_alpha]
    np_assignment_matrices = np.stack(np_assignment_matrices, 0)
    assignment_matrices = torch.from_numpy(np_assignment_matrices).float().to(log_alpha.device)
    return assignment_matrices