import torch
import numpy as np

from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

Q = torch.load("data/queries.pt")
Q = Q / Q.norm(dim=-1, keepdim=True)
C = torch.load("data/corpuses.pt")
C = C / C.norm(dim=-1, keepdim=True)

cs = lambda x, y: 1 - np.dot(x, y)

global_sims = torch.zeros((Q.shape[0], C.shape[0]))

def fastdtw_wrapper(i, j, q, c, dist):
    sim = fastdtw(q, c, dist=dist)[0]
    global_sims[i][j] = sim  

for i in tqdm(range(Q.shape[0])):
    for j in range(C.shape[0]):
        fastdtw_wrapper(i, j, Q[i], C[j], cs)

torch.save(global_sims, "data/sims.pt")