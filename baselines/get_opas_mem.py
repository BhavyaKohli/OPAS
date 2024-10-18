import numpy as np
import math
import h5py
import logging
import os, sys
import argparse
import tracemalloc

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.pardir))
from main import *

from time import time
from argparse import Namespace as AttributeDict
from omegaconf import OmegaConf


class DatasetTensors:
    def __init__(self, dataset):
        tensors = torch.load(f"tensors/tensors_{dataset}.pt")
        self.q = tensors["q"]
        self.l = tensors["l"]
        self.c = tensors["c"].numpy()


@torch.no_grad()
def embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model=None, inner_batch_size=800, aggregator=None):
    C = torch.from_numpy(dataset.c).float()
    Cembed = []
    for batch in tqdm(range(0, len(C), inner_batch_size), disable=False, leave=False):
        c = C[batch:batch+inner_batch_size].to(next(embed_model.parameters()).device)
        c = embed_if_image_and_normalize(c, image_embed_model)
        c = embed_model(preembed_model(c))
        c = normalize(c)
        if aggregator is not None:
            c = aggregator(c)
        Cembed.append(c)
    C = torch.vstack(Cembed)
    return C


def load_models(root, name="best", device="cpu"):
    if name not in ["best", "last", "latest"]:
        raise NotImplementedError("Only `best`, `last`, and `latest` models are supported")
    
    if name=="best":
        suffix = ""
    else:
        suffix = f"_{name}"
    
    print(f"Loading `{name}` model")

    model = torch.load(f"{root}/model{suffix}.pt", map_location=device)
    scoremodel = torch.load(f"{root}/scmodel{suffix}.pt", map_location=device)
    embed_model = torch.load(f"{root}/embed_model{suffix}.pt", map_location=device)
    if os.path.exists(f"{root}/preembed_model{suffix}.pt"):
        preembed_model = torch.load(f"{root}/preembed_model{suffix}.pt", map_location=device)
    else:
        tokenize_transform = lambda x: tokenize(x, args)[0]
        preembed_model = TransformInput(tokenize_transform).to(device)
    
    if os.path.exists(f"{root}/aggregator{suffix}.pt"):
        aggregator = torch.load(f"{root}/aggregator{suffix}.pt", map_location=device)
    else:
        aggregator = None

    return model, scoremodel, embed_model, preembed_model, aggregator


@torch.no_grad()
def prepare_batch(q, embed_model, preembed_model, image_embed_model=None, noise_override=False):
    device = next(embed_model.parameters()).device
    q = q.to(device) 
    if not noise_override:
        q = q + batch_get_white_noise(q, args.SNR)     # torch.randn_like(q) * noise
    # q is bmd, c is Nnd, l is bN
    q = embed_if_image_and_normalize(q, image_embed_model).to(device)
    q = embed_model(preembed_model(q))
    q = normalize(q)
    return q


@torch.no_grad()
def get_batch_scores(q, l, C, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, noise_override=False, aggregator=None):
    """
    q: (b, m, d)
    l: (b, num_corpuses)
    C: (num_corpuses, n, d)

    Returns-
    netscore: (b, num_corpuses)
    l       : (b, num_corpuses)
    """
    q = prepare_batch(q, embed_model, preembed_model, image_embed_model=image_embed_model, noise_override=noise_override)

    if aggregator is None:
        qct = torch.einsum("bmd,Nnd->bNmn", q, C)
        model_inputs = stagger_and_concat(qct, num_stagger=stagger) # bNsmn  s = num_stagger+1

        lambdas = torch.stack([model(x) for x in model_inputs])
        # bNm1

        F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))

        P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)

        RmPC = Rm_mat @ P @ C.squeeze(-1)

        lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
        normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

        allscores = torch.stack([lamscore, normscore], dim=2)
        netscore = 2*scoremodel(-allscores).squeeze()      # b

    else:
        q = aggregator(q)
        netscore = 2 * F.sigmoid(-F.relu(q.unsqueeze(1) - C.unsqueeze(0)).sum(dim=-1))

    return netscore.to('cpu'), l.to('cpu')


@torch.no_grad()
def compute_map_mrr(netscores, true_labels):
    """
    netscores: (..., num_corpuses)
    true_labels: (..., num_corpuses)
    """
    netscores = netscores.to('cpu')
    true_labels = true_labels.to('cpu')
    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    mRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    mAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    mAP /= (torch.arange(ranked_output.shape[1]) + 1)
    mAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    mAP = mAP.sum(dim=1).mean().item()

    return mAP, mRR


def get_time_entry(ts, num_runs, num_c=None, num_q=None, map=None, mrr=None):
    return {"t": ts, "nc": num_c, "nq": num_q, "nr": num_runs, "map": map, "mrr": mrr}


def get_mean_std(arr):
    return np.mean(arr), np.std(arr)


def get_mean_std_formatted(arr):
    mu, sigma = get_mean_std(arr)
    return f"{mu:.4f}±{sigma:.4f}"


def print_stats(ts, num_runs, num_c, num_q, map, mrr):
    mu, sig = get_mean_std(ts)
    logging.info(f"Time taken for running inference on {num_q} queries (computing {num_q}*{num_c} scores, averaged over {num_runs} runs): {mu:.4f}±{sig:.4f} s")
    logging.info(f"Average time for {num_c} comparisons: {(mu/num_q):.4f}±{(sig/num_q):.4f} s/query")
    logging.info(f"Average time for single comparison: {1000*(mu/num_q/num_c):.4f}±{1000*(sig/num_q/num_c):.4f} ms/query/corpus item")

    
    logging.info(f"\n\nPerformance ({num_q} queries, averaged over {num_runs} runs): MAP: {get_mean_std_formatted(map)}, MRR: {get_mean_std_formatted(mrr)}")
    perf = np.stack((map, mrr)).T
    maxperf = perf[np.argmax(perf.sum(axis=1))]
    logging.info(f"Best performance: MAP: {maxperf[0]:.4f}, MRR: {maxperf[1]:.4f}")


if __name__ == "__main__":
    logging.info = lambda x: None
    config = OmegaConf.load("configs/opas_mem.config")

    parser = argparse.ArgumentParser('Inference Time Comparisons')
    parser.add_argument("--expt_id", type=str, help="Experiment ID")
    args = parser.parse_args()
    
    tracemalloc.start()

    experiment_id = args.expt_id 
    logging.info(experiment_id)

    image_embed_model = None
    if experiment_id.startswith("C"):
        image_embed_model_ckpt = "../data/image_sequence/embedding_models/cifar_ae.pkl"
        image_embed_model = Autoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False    

    elif experiment_id.startswith("L"):
        image_embed_model_ckpt = "../data/image_sequence/embedding_models/lsun_ae.pkl"
        image_embed_model = LSUNAutoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

    root = f"../models/{experiment_id}/"
    DEVICE = "cpu"

    import pickle
    with open(root+"args.pkl", "rb") as f:
        args = pickle.load(f)

    model, scoremodel, embed_model, preembed_model, aggregator = load_models(root, name="best", device=DEVICE)
    model.eval(); scoremodel.eval(); embed_model.eval(); preembed_model.eval()
    if aggregator is not None:
        aggregator.eval()

    nwt, lamwt = next(scoremodel.parameters()).exp().detach().cpu()
    nwt, lamwt = nwt.item(), lamwt.item()

    logging.info(f"normscore weight: {nwt:.4f}, lamscore weight: {lamwt:.4f}")

    ##### opas HPARAMS ######
    b = args.b          # hinge margin for negative gap penalty (b-Apa)
    b1 = args.b1         # hinge margin for positive gap penalty (Apa-b)
    delta = args.delta     # hinge margin for contrastive loss
    lr = args.lr
    nepochs = args.nepochs
    stagger = args.stagger
    lamwt = args.lamwt    # loss coefficient for negative gap penalty
    gapwt = args.gapwt       # loss coefficient for positive gap penalty
    noise = args.noise
    M = 6
    N = 20
    ####################

    A_mat, a_vec, Rm_mat = get_opas_constants(M, N, DEVICE)

    CFG = AttributeDict(tau=1, n_sink_iter=20, n_samples=1)

    # inference times (and hparams) for all experiments stored for comparisons
    GLOBAL_TIMES = AttributeDict()

    if args.dataset == "lsun384":
        args.dataset = "lsun"
    elif args.dataset == "human":
        args.dataset = "speech"

    test_dataset = DatasetTensors(args.dataset)
    q, l = test_dataset.q, test_dataset.l

    ######################################################
    ######### CORPUS EMBEDDING FOR opas ##################
    ######################################################
    num_runs = int(config['opas_cembed']['num_runs'])

    ts = []
    for _ in tqdm(range(num_runs), desc="Corpus Embedding"):
        start = time()
        C = embed_full_corpus(test_dataset, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator, inner_batch_size=100)
        stop = time()
        ts.append(stop-start)

    mu, sig = get_mean_std(ts)
    GLOBAL_TIMES.corpus_embedding = get_time_entry(ts, num_runs)
    logging.info(f"Corpus embedding cost for {len(test_dataset.c)} corpus items (one-time cost): {mu:.4f}±{sig:.4f} s (averaged over {num_runs} runs)")

    A_mat, a_vec, Rm_mat = get_opas_constants(M, N, DEVICE)

    num_c = len(test_dataset.c)
    num_q = int(config['opas_gpu']['num_q'])
    num_runs = int(config['opas_gpu']['num_runs'])

    ts = []
    map, mrr = [], []
    for _ in tqdm(range(num_runs), desc="OPAS (CPU)"):
        start = time()
        netscore, true_labels = get_batch_scores(q, l, C, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator, stagger=stagger)
        map_, mrr_ = compute_map_mrr(netscore, true_labels)
        stop = time()

        ts.append(stop-start)
        map.append(map_)
        mrr.append(mrr_)

    current, max = tracemalloc.get_traced_memory()
    current, max = current / 1024**2, max / 1024**2
    print(f"{max:.4f}\t{np.mean(map):.4f}")
    tracemalloc.stop()