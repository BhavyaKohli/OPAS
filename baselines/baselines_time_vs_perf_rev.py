import numpy as np
import math
import h5py
import logging
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.pardir))

from tqdm import tqdm
from time import time
from argparse import Namespace as AttributeDict
from omegaconf import OmegaConf


class DatasetTensors:
    def __init__(self, dataset):
        tensors = torch.load(f"tensors/tensors_{dataset}.pt")
        self.q = tensors["q"]
        self.l = tensors["l"]
        self.c = tensors["c"].numpy()
        print(self.q.shape, self.l.shape, self.c.shape)


def normalize(*args):
    ret = []
    for arg in args:
        ret.append(arg / arg.norm(dim=-1, keepdim=True))
    if len(ret) == 1: ret = ret[0]
    return ret


def get_white_noise(signal, SNR):
    RMS_s = torch.sqrt(torch.mean(signal**2))
    RMS_n = torch.sqrt(RMS_s**2 / (pow(10, SNR/10)))
    STD_n = RMS_n
    noise = torch.distributions.Normal(0, STD_n).sample(signal.shape)
    return noise

def batch_get_white_noise(x, SNR):
    # x is B x m/n x signal
    return get_white_noise(x, SNR)


@torch.no_grad()
def compute_hitsk(netscores, true_labels, k):
    pos_scores = [sc[lab==1] for sc,lab in zip(netscores, true_labels)]
    neg_scores = [sc[lab==0] for sc,lab in zip(netscores, true_labels)]

    if isinstance(k, int):
        k = [k]
    
    hits = []
    for k_ in k:
        topkneg = [torch.topk(ns, k_)[0][-1] for ns in neg_scores]
        hits.append(np.mean([(torch.sum(ps > tkn) / len(ps)).item() for ps,tkn in zip(pos_scores, topkneg)]))
    return hits


@torch.no_grad()
def compute_map_mrr(netscores, true_labels, k=None):
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

    if k is not None:
        hits = compute_hitsk(netscores, true_labels, k)
    else:
        hits = None

    return mAP, mRR, hits


# bookkeeping functions
def get_time_entry(ts, num_runs, num_c=None, num_q=None, map=None, mrr=None):
    return {"t": ts, "nc": num_c, "nq": num_q, "nr": num_runs, "map": map, "mrr": mrr}

def get_mean_std(arr):
    return np.mean(arr), np.std(arr)

def get_mean_std_formatted(arr):
    mu, sigma = get_mean_std(arr)
    return f"{mu:.4f}±{sigma:.4f}"

def get_run_stats(ts, num_runs, num_c, num_q, map, mrr, hitsk):
    mu, sig = get_mean_std(ts)
    time_per_comp = f"{1000*(mu/num_q/num_c):.4f}±{1000*(sig/num_q/num_c):.4f}"
    map_ = f"{get_mean_std_formatted(map)}"
    mrr_ = f"{get_mean_std_formatted(mrr)}"
    
    if hitsk[0] is not None:
        hitsk = np.array(hitsk)
        mu = hitsk.mean(axis=0)
        sig = hitsk.std(axis=0)
        hitsk_ = [f"{mu[i]:.4f}±{sig[i]:.4f}" for i in range(len(mu))]
    else:
        hitsk_ = None

    return time_per_comp, map_, mrr_, hitsk_

if __name__ == "__main__":
    logging.info = lambda x: None
    config = OmegaConf.load("configs/main_rev.config")

    parser = argparse.ArgumentParser('Inference Time Comparisons')
    parser.add_argument("--dataset", type=str, help="dataset")#, choices=["audio", "speech", "cifar", "lsun"])
    parser.add_argument("--debug", action='store_true', help="debug mode")
    args = parser.parse_args()

    test_dataset = DatasetTensors(args.dataset)
    q_main, l = test_dataset.q[:10], test_dataset.l[:10]
    test_dataset_c_main = test_dataset.c

    if len(q_main.shape) > 3:
        q_main = q_main.flatten(start_dim=2)
        cshape = test_dataset_c_main.shape
        test_dataset_c_main = test_dataset_c_main.reshape(cshape[0], cshape[1], -1)
                             
    print(q_main.shape, l.shape, test_dataset_c_main.shape)

    M = 6
    N = 20

    # inference times (and hparams) for all experiments stored for comparisons
    GLOBAL_TIMES = AttributeDict()

    DEBUG = args.debug
    TOPK = False        # not used in current version

    if TOPK: k = [1, 2, 3]
    else: k = None

    def run_baseline(name, distance_function, q, test_dataset_c, num_c, num_runs, num_q, SKIP):
        logging.info(f"\nRunning {name}")

        inclusions = []
        for i in range(num_q):
            inclusions += list(l[i].nonzero().squeeze().numpy())    
        inclusions = np.unique(inclusions)
        other_idxs = list(set(range(num_c))-set(inclusions))

        ts = []
        map, mrr, hitsk = [], [], []
        for _ in tqdm(range(num_runs), desc=f"{SKIP}: {name.upper()}"):
            method_q = q.clone()[:num_q]
            method_q = method_q + batch_get_white_noise(method_q, 1)
            method_q = normalize(method_q)

            c = test_dataset_c[inclusions]
            
            random_idxs = np.random.choice(other_idxs, size=num_c-len(c), replace=False)
            c_ = test_dataset_c[random_idxs]
            method_c = normalize(torch.from_numpy(np.vstack((c, c_))))

            method_l = torch.hstack((l[:num_q,inclusions],l[:num_q,random_idxs]))
            sim = torch.zeros(method_q.shape[0], method_c.shape[0])

            start = time()
            for i, query in enumerate(method_q):
                for j, corpus in enumerate(tqdm(method_c, desc=f"Query {i}", leave=False)):
                    sim[i,j] = -distance_function(corpus.cpu().numpy(), query.cpu().numpy())
            map_, mrr_, hitsk_ = compute_map_mrr(sim, method_l, k=k)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_); hitsk.append(hitsk_)

        time_per_comp, map_, mrr_, hitsk_ = get_run_stats(ts, num_runs, num_c, num_q, map, mrr, hitsk)
        setattr(GLOBAL_TIMES, name, [SKIP, time_per_comp, map_, mrr_])

    os.makedirs("../plots_and_figures/data/", exist_ok=True)

    with open(f"../plots_and_figures/data/times_{args.dataset}.log", "a+") as main_logfile:
        print(f"Dataset: {args.dataset}", file=main_logfile)

        SKIP_LIST = [1, 10, 50, 100, 200, 500, 1000][:1]#[::-1]       # inputs: 4000 (audio), 3072 (cifar)
        if args.dataset == "lsun":
            # original = 196_608
            SKIP_LIST = [500, 1000, 15000, 25000, 30000, 40000, 50000, 90000][::-1]
        
        list_of_times = []
        for SKIP in SKIP_LIST:
            q, test_dataset_c = q_main[:,:,::SKIP], test_dataset_c_main[:,:,::SKIP]

            print(f"Query-corpus sizes: {q.shape, test_dataset_c.shape}")

            #######################################################
            ################## SHARP ##############################
            #######################################################

            if config['sharp'].get('skip', 0)==0:
                from sharp import sharp_sdtw_div

                sharp_num_c = len(test_dataset.c)
                sharp_num_runs = int(config['sharp']['num_runs']) if not DEBUG else 2
                sharp_num_q = int(config['sharp']['num_q'])

                run_baseline("SHARP", sharp_sdtw_div, q, test_dataset_c, sharp_num_c, sharp_num_runs, sharp_num_q, SKIP)
                print(GLOBAL_TIMES, file=main_logfile)

            if DEBUG:
                print(GLOBAL_TIMES); raise

            #######################################################
            ################## FASTDTW ############################
            #######################################################

            if config['fdtw'].get('skip', 0)==0:
                from fastdtw import fastdtw as dtw

                dtw_num_c = len(test_dataset.c)
                dtw_num_runs = int(config['fdtw']['num_runs'])
                dtw_num_q = int(config['fdtw']['num_q'])

                cs = lambda x, y: 1 - np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)
                dtw_func = lambda c, q: dtw(c, q, dist=cs)[0]

                run_baseline("FASTDTW", dtw_func, q, test_dataset_c, dtw_num_c, dtw_num_runs, dtw_num_q, SKIP)
                print(GLOBAL_TIMES, file=main_logfile)

            #######################################################
            ##################### SDTW ############################
            #######################################################

            if config['sdtw'].get('skip', 0)==0:
                logging.info("\n\nRunning SDTW")

                from sdtw import sdtw_div

                sdtw_num_c = len(test_dataset.c)
                sdtw_num_runs = int(config['sdtw']['num_runs'])
                sdtw_num_q = int(config['sdtw']['num_q'])

                run_baseline("SDTW", sdtw_div, q, test_dataset_c, sdtw_num_c, sdtw_num_runs, sdtw_num_q, SKIP)
                print(GLOBAL_TIMES, file=main_logfile)

            #######################################################
            ################## MASS ###############################
            #######################################################

            if config['mass'].get('skip', 0)==0:
                try:
                    import mass_ts as mts

                    def mass_dist(c, q) :
                        # q : md
                        # c : nd
                        qflat = q.reshape(-1)
                        cflat = c.reshape(-1)

                        return np.abs(np.min(mts.mass2(cflat, qflat)))

                    mass_num_c = int(config['mass']['num_c'])
                    mass_num_runs = int(config['mass']['num_runs'])
                    mass_num_q = int(config['mass']['num_q'])

                    run_baseline("MASS", mass_dist, q, test_dataset_c, mass_num_c, mass_num_runs, mass_num_q, SKIP)

                except Exception as e:
                    print(f"MASS failed with error: {e}")

            print(GLOBAL_TIMES, file=main_logfile)
            list_of_times.append(GLOBAL_TIMES)

    torch.save(list_of_times, f"../plots_and_figures/data/times_{args.dataset}.pkl")