import torch
import numpy as np

from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from time import perf_counter
from multiprocessing import Pool

st = perf_counter()

Q = torch.load("data/queries.pt")
Q = Q / Q.norm(dim=-1, keepdim=True)
C = torch.load("data/corpuses.pt")
C = C / C.norm(dim=-1, keepdim=True)

cs = lambda x, y: 1 - np.dot(x, y)

global_sims = torch.zeros((Q.shape[0], C.shape[0]))

def fastdtw_wrapper(i, j, q, c):
    dist = lambda x, y: 1 - np.dot(x, y)
    sim = fastdtw(q, c, dist=dist)[0]
    return (i, j, sim)

# results = []
# for i in tqdm(range(Q.shape[0])):
#     for j in range(C.shape[0]):
#         results.append(fastdtw_wrapper(i, j, Q[i], C[j]))

batch_size = 100
pbar = tqdm(range(0, Q.shape[0], batch_size))
for batch in pbar:
    pbar.set_description_str(f"Getting args...")
    args = [(i+batch, j, Q[i+batch], C[j]) for i in range(batch_size) for j in range(C.shape[0])]

    pbar.set_description_str(f"Multiprocessing now...")
    with Pool(processes=16) as pool:
        results = pool.starmap(fastdtw_wrapper, args)

    pbar.set_description_str(f"Updating global_sims...")
    for i, j, sim in results:
        global_sims[i][j] = sim

torch.save(global_sims, "data/sims.pt")
print(perf_counter() - st)
import ipdb; ipdb.set_trace()