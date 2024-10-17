import wandb
import numpy as np
import math
import h5py
import logging
import os, sys
import argparse
sys.path.append('/raid/infolab/bhavyakohli/parseq/')

import torch
import torch.nn as nn
import torch.nn.functional as F

from parseq.tstok.tokenizer import Tokenizer
from parseq.utils import make_A, AttributeDict
from torch.utils.data import DataLoader, Dataset
from parseq.gumbel_sinkhorn_ops import gumbel_sinkhorn

from time import time
from datetime import datetime
from configparser import ConfigParser

from scripts.main import *
from tqdm import tqdm  
from argparse import ArgumentParser, Namespace as AttributeDict


@torch.no_grad()
def embed_full_corpus(dataset, embed_model, preembed_model, inner_batch_size=800):
    C = torch.from_numpy(dataset.c).float()
    Cembed = []
    for batch in tqdm(range(0, len(C), inner_batch_size), disable=True):
        c = C[batch:batch+inner_batch_size].to(next(embed_model.parameters()).device)
        c = embed_if_image_and_normalize(c)
        c = embed_model(preembed_model(c))
        c = normalize(c)
        Cembed.append(c)
    C = torch.vstack(Cembed)
    return C

@torch.no_grad()
def prepare_batch(q, embed_model, preembed_model, noise_override=False):
    q = q.to(next(embed_model.parameters()).device) 
    if not noise_override:
        q = q + batch_get_white_noise(q, args.SNR)     # torch.randn_like(q) * noise
    # q is bmd, c is Nnd, l is bN
    q = embed_if_image_and_normalize(q)
    q = embed_model(preembed_model(q))
    q = normalize(q)
    return q

@torch.no_grad()
def get_batch_scores(q, l, C, model, scoremodel, embed_model, preembed_model, stagger=2, noise_override=False):
    """
    q: (b, m, d)
    l: (b, num_corpuses)
    C: (num_corpuses, n, d)

    Returns-
    netscore: (b, num_corpuses)
    l       : (b, num_corpuses)
    """
    q = prepare_batch(q, embed_model, preembed_model, noise_override=noise_override)

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

def eval_loader(loader, num_batches, C, model, scoremodel, embed_model, preembed_model, stagger=2, n_inner=5, noise_override=False):
    map, mrr = [], []
    for q, l in tqdm(loader):
        res = np.zeros((n_inner,2))
        for i in range(n_inner):
            netscores, true_labels = get_batch_scores(q, l, C, model, scoremodel, embed_model, preembed_model, stagger=stagger, noise_override=noise_override)
            map_, mrr_ = compute_map_mrr(netscores, true_labels)
            res[i] = map_, mrr_
        map_, mrr_ = res[np.argmax(res.sum(axis=1))]
        map.append(map_)
        mrr.append(mrr_)
        if len(map) >= num_batches:
            break
    return np.mean(map), np.mean(mrr)


@torch.no_grad()
def embed_if_image_and_normalize(c, image_embed_model=None):
    if len(c.shape) != 3:   
        # b x n x c x h x w instead of b x n x d
        c = torch.stack([embed_image(image_embed_model, c_) for c_ in c])
    return normalize(c)

@torch.no_grad()
def compute_metrics(dataset, model, scoremodel, embed_model, preembed_model, stagger=2, verbose=False):

    loader = dataset.get_dataloader(batch_size=200, shuffle=True)

    C = embed_full_corpus(dataset, embed_model, preembed_model)
    # C is (N, n, xoutdim)

    netscores = []
    true_labels = []
    for n, (q, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        q = q.to(next(model.parameters()).device) 
        # q is bmd, c is Nnd, l is bN
        q = q + batch_get_white_noise(q, args.SNR)     # torch.randn_like(q) * noise
        q = embed_if_image_and_normalize(q)
        q = embed_model(preembed_model(q))
        q = normalize(q)
        # q is (b, m, xoutdim)

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

        netscores.append(netscore.to('cpu'))
        true_labels.append(l.to('cpu'))

    netscores = torch.vstack(netscores)
    true_labels = torch.vstack(true_labels).squeeze() # 

    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    mRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    mAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    mAP /= (torch.arange(ranked_output.shape[1]) + 1)
    mAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    mAP = mAP.sum(dim=1).mean().item()

    return mAP, mRR


def load_models(root, name="best"):
    if name not in ["best", "last", "latest"]:
        raise NotImplementedError("Only `best`, `last`, and `latest` models are supported")
    
    if name=="best":
        suffix = ""
    else:
        suffix = f"_{name}"
    
    logging.info(f"Loading `{name}` model")

    model = torch.load(f"{root}/model{suffix}.pt", map_location=DEVICE)
    scoremodel = torch.load(f"{root}/scmodel{suffix}.pt", map_location=DEVICE)
    embed_model = torch.load(f"{root}/embed_model{suffix}.pt", map_location=DEVICE)
    if os.path.exists(f"{root}/preembed_model{suffix}.pt"):
        preembed_model = torch.load(f"{root}/preembed_model{suffix}.pt", map_location=DEVICE)
    else:
        tokenize_transform = lambda x: tokenize(x, args)[0]
        preembed_model = TransformInput(tokenize_transform).to(DEVICE)

    return model, scoremodel, embed_model, preembed_model

# bookkeeping functions


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
    
    logging.info = print

    logging.basicConfig(
        filename=f"inference.log",
        filemode="a+",
        level=logging.INFO,
        format="%(levelname)s (%(asctime)s): %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p"
    )
    config = ConfigParser()
    config.read("inference.config")

    parser = ArgumentParser('Inference Time Comparisons')
    parser.add_argument("experiment_id", type=str, help="Experiment ID")
    parser.add_argument("--skip", type=int, help="skip value", default=1)
    parser.add_argument("--device", type=int, help="cuda device, pass -1 for cpu", default=-1)
    args = parser.parse_args()
    
    SKIP = args.skip
    logging.info(f"RUNNING BASELINES WITH SKIP: {SKIP}")

    experiment_id = args.experiment_id 
    logging.info(experiment_id)

    if experiment_id.startswith("C"):
        image_embed_model_ckpt = "/raid/infolab/bhavyakohli/parseq/data/video_cifar/autoencoder/weights/cifar_ae_final.pkl"
        image_embed_model = Autoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False    

    elif experiment_id.startswith("L"):
        image_embed_model_ckpt = "/raid/infolab/bhavyakohli/parseq/data/video_lsun/autoencoder/weights2/lsun_ae_384.pkl"
        image_embed_model = LSUNAutoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

    root = f"/raid/infolab/bhavyakohli/parseq/scripts/models/{experiment_id}/"
    DEVICE = f'cuda:{args.device}' if args.device != -1 else "cpu"

    if experiment_id[0] in ["C", "L"]:
        image_embed_model = image_embed_model.to(DEVICE)
        embed_if_image_and_normalize = partial(embed_if_image_and_normalize, image_embed_model=image_embed_model)

    import pickle
    with open(root+"args.pkl", "rb") as f:
        args = pickle.load(f)

    TEST_FILE = f"../../final_data/{args.dataset}/dataset_test.hdf5"
    if experiment_id.startswith("L"):
        TEST_FILE = TEST_FILE.replace(".hdf5", "_orig.hdf5")
    print(TEST_FILE)

    logging.info(f"Running inference on device {DEVICE} using models from {root} using dataset {args.dataset} found at {TEST_FILE}")

    test_dataset = PairDatasetTest(TEST_FILE)


    model, scoremodel, embed_model, preembed_model = load_models(root, name="best")
    model.eval(); scoremodel.eval(); embed_model.eval(); preembed_model.eval()

    nwt, lamwt = next(scoremodel.parameters()).exp().detach().cpu()
    nwt, lamwt = nwt.item(), lamwt.item()

    logging.info(f"normscore weight: {nwt:.4f}, lamscore weight: {lamwt:.4f}")

    ##### PARSEQ HPARAMS ######
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

    A_mat, a_vec, Rm_mat = get_parseq_constants(M, N, DEVICE)

    CFG = AttributeDict(tau=1, n_sink_iter=20, n_samples=1)

    # inference times (and hparams) for all experiments stored for comparisons
    GLOBAL_TIMES = AttributeDict()

    ######################################################
    ######### CORPUS EMBEDDING FOR PARSEQ ################
    ######################################################
    num_runs = int(config['parseq_cembed']['num_runs'])

    ts = []
    for _ in tqdm(range(num_runs), desc="Corpus Embedding"):
        start = time()
        C = embed_full_corpus(test_dataset, embed_model, preembed_model, inner_batch_size=800)
        stop = time()
        ts.append(stop-start)

    mu, sig = get_mean_std(ts)
    GLOBAL_TIMES.corpus_embedding = get_time_entry(ts, num_runs)
    logging.info(f"Corpus embedding cost for {len(test_dataset.c)} corpus items (one-time cost): {mu:.4f}±{sig:.4f} s (averaged over {num_runs} runs)")

    ## PARSEQ GPU

    logging.info("\n\nRunning PARSEQ (GPU)")

    model, scoremodel, embed_model, preembed_model = model.to(DEVICE), scoremodel.to(DEVICE), embed_model.to(DEVICE), preembed_model.to(DEVICE)
    A_mat, a_vec, Rm_mat = get_parseq_constants(M, N, DEVICE)

    num_c = len(C)
    num_q = int(config['parseq_gpu']['num_q'])
    num_runs = int(config['parseq_gpu']['num_runs'])
    n_inner = 1
    
    # torch.manual_seed(69420)
    torch.manual_seed(6969)
    loader = test_dataset.get_dataloader(batch_size=num_q, shuffle=True)
    q, l = next(iter(loader))
    tensors = {'q': q, 'l': l, 'c': torch.from_numpy(test_dataset.c)}

    ts = []
    map, mrr = [], []
    for _ in tqdm(range(num_runs), desc="PARSEQ (GPU)"):
        start = time()
        netscore, true_labels = get_batch_scores(q, l, C, model, scoremodel, embed_model, preembed_model, stagger=stagger)
        map_, mrr_ = compute_map_mrr(netscore, true_labels)
        stop = time()

        ts.append(stop-start)
        map.append(map_)
        mrr.append(mrr_)

    print_stats(ts, num_runs, num_c, num_q, map, mrr)
    GLOBAL_TIMES.parseq_gpu = get_time_entry(ts, num_runs, num_c, num_q, map, mrr) 

    tensors['ptime'] = 1000 * np.mean(ts)/num_q/num_q 
    tensors['map'] = np.mean(map) 
    tensors['mrr'] = np.mean(mrr) 
    torch.save(tensors, f"tensors_{args.dataset}.pt")
    raise

    ## PARSEQ CPU

    if config['parseq_cpu'].getint('skip', 0)==0:
        logging.info("\n\nRunning PARSEQ (CPU)")

        num_runs = int(config['parseq_cpu']['num_runs'])

        testdevice = 'cpu'
        model, scoremodel, embed_model, preembed_model = model.to(testdevice), scoremodel.to(testdevice), embed_model.to(testdevice), preembed_model.to(testdevice)
        A_mat, a_vec, Rm_mat = get_parseq_constants(M, N, testdevice)
        n_inner = 1

        ts = []
        map, mrr = [], []
        for _ in tqdm(range(num_runs), desc="PARSEQ (CPU)"):
            start = time()
            netscore, true_labels = get_batch_scores(q, l, C.to('cpu'), model, scoremodel, embed_model, preembed_model, stagger=stagger)
            map_, mrr_ = compute_map_mrr(netscore, true_labels)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_)

        print_stats(ts, num_runs, num_c, num_q, map, mrr)
        GLOBAL_TIMES.parseq_cpu = get_time_entry(ts, num_runs, num_c, num_q, map, mrr)


    #######################################################
    ################## SHARP ##############################
    #######################################################

    if config['sharp'].getint('skip', 0)==0:
        logging.info("\n\nRunning SHARP")

        from sharp import sharp_sdtw_div

        sharp_num_c = len(C)
        sharp_num_runs = int(config['sharp']['num_runs'])
        sharp_num_q = int(config['sharp']['num_q'])

        inclusions = []
        for i in range(sharp_num_q):
            inclusions += list(l[i].nonzero().squeeze().numpy())    
        inclusions = np.unique(inclusions)
        other_idxs = list(set(range(sharp_num_c))-set(inclusions))

        ts = []
        map, mrr = [], []
        for _ in tqdm(range(sharp_num_runs), desc="SHARP"):
            sharp_q = q.clone()[:sharp_num_q]
            sharp_q = sharp_q + batch_get_white_noise(sharp_q, args.SNR)
            sharp_q = normalize(sharp_q)

            c = test_dataset.c[inclusions]
            
            random_idxs = np.random.choice(other_idxs, size=sharp_num_c-len(c), replace=False)
            c_ = test_dataset.c[random_idxs]
            sharp_c = normalize(torch.from_numpy(np.vstack((c, c_))))

            sharp_l = torch.hstack((l[:sharp_num_q,inclusions],l[:sharp_num_q,random_idxs]))
            sim = torch.zeros(sharp_q.shape[0], sharp_c.shape[0])

            start = time()
            for i, query in enumerate(sharp_q):
                for j, corpus in enumerate(sharp_c):
                    sim[i,j] = -sharp_sdtw_div(corpus.cpu().numpy()[:,::SKIP], query.cpu().numpy()[:,::SKIP])
            map_, mrr_ = compute_map_mrr(sim, sharp_l)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_)

        print_stats(ts, sharp_num_runs, sharp_num_c, sharp_num_q, map, mrr)
        GLOBAL_TIMES.sharp = get_time_entry(ts, sharp_num_runs, sharp_num_c, sharp_num_q, map, mrr)


    #######################################################
    ################## FASTDTW ############################
    #######################################################

    if config['fdtw'].getint('skip', 0)==0:
        logging.info("\n\nRunning FASTDTW")

        from fastdtw import fastdtw as dtw

        dtw_num_c = len(C)
        dtw_num_runs = int(config['fdtw']['num_runs'])
        dtw_num_q = int(config['fdtw']['num_q'])

        inclusions = []
        for i in range(dtw_num_q):
            inclusions += list(l[i].nonzero().squeeze().numpy())    
        inclusions = np.unique(inclusions)
        other_idxs = list(set(range(dtw_num_c))-set(inclusions))

        cs = lambda x, y: 1 - np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)

        ts = []
        map, mrr = [], []
        for run in tqdm(range(dtw_num_runs), desc="FASTDTW"):
            dtw_q = q.clone()[:dtw_num_q]
            dtw_q = dtw_q + batch_get_white_noise(dtw_q, args.SNR)
            dtw_q = normalize(dtw_q)

            c = test_dataset.c[inclusions]
            
            random_idxs = np.random.choice(other_idxs, size=dtw_num_c-len(c), replace=False)
            c_ = test_dataset.c[random_idxs]
            dtw_c = normalize(torch.from_numpy(np.vstack((c, c_))))

            dtw_l = torch.hstack((l[:dtw_num_q,inclusions],l[:dtw_num_q,random_idxs]))
            sim = torch.zeros(dtw_q.shape[0], dtw_c.shape[0])

            start = time()
            for i, query in enumerate(dtw_q):
                for j, corpus in enumerate(dtw_c):
                    if i == 0 and j == 0 and run == 0:
                        print(corpus.cpu().numpy()[:,::SKIP].shape, query.cpu().numpy()[:,::SKIP].shape)
                    sim[i,j] = -dtw(corpus.cpu().numpy()[:,::SKIP], query.cpu().numpy()[:,::SKIP], dist=cs)[0]
            map_, mrr_ = compute_map_mrr(sim, dtw_l)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_)

        print_stats(ts, dtw_num_runs, dtw_num_c, dtw_num_q, map, mrr)
        GLOBAL_TIMES.fastdtw = get_time_entry(ts, dtw_num_runs, dtw_num_c, dtw_num_q, map, mrr)

    #######################################################
    ##################### SDTW ############################
    #######################################################

    if config['sdtw'].getint('skip', 0)==0:
        logging.info("\n\nRunning SDTW")

        from sdtw import sdtw_div

        dtw_num_c = len(C)
        dtw_num_runs = int(config['sdtw']['num_runs'])
        dtw_num_q = int(config['sdtw']['num_q'])

        inclusions = []
        for i in range(dtw_num_q):
            inclusions += list(l[i].nonzero().squeeze().numpy())    
        inclusions = np.unique(inclusions)
        other_idxs = list(set(range(dtw_num_c))-set(inclusions))

        ts = []
        map, mrr = [], []
        for _ in tqdm(range(dtw_num_runs), desc="SDTW"):
            dtw_q = q.clone()[:dtw_num_q]
            dtw_q = dtw_q + batch_get_white_noise(dtw_q, args.SNR)
            dtw_q = normalize(dtw_q)

            c = test_dataset.c[inclusions]
            
            random_idxs = np.random.choice(other_idxs, size=dtw_num_c-len(c), replace=False)
            c_ = test_dataset.c[random_idxs]
            dtw_c = normalize(torch.from_numpy(np.vstack((c, c_))))

            dtw_l = torch.hstack((l[:dtw_num_q,inclusions],l[:dtw_num_q,random_idxs]))
            sim = torch.zeros(dtw_q.shape[0], dtw_c.shape[0])

            start = time()
            for i, query in enumerate(dtw_q):
                for j, corpus in enumerate(dtw_c):
                    sim[i,j] = -sdtw_div(corpus.cpu().numpy()[:,::SKIP], query.cpu().numpy()[:,::SKIP], gamma=1)
            map_, mrr_ = compute_map_mrr(sim, dtw_l)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_)

        print_stats(ts, dtw_num_runs, dtw_num_c, dtw_num_q, map, mrr)
        GLOBAL_TIMES.fastdtw = get_time_entry(ts, dtw_num_runs, dtw_num_c, dtw_num_q, map, mrr)

    #######################################################
    ################## MASS ###############################
    #######################################################

    if config['mass'].getint('skip', 0)==0:
        logging.info("\n\nRunning MASS")

        import mass_ts as mts
        from tqdm import tqdm

        def mass_dist(c, q, b=4000) :
            # q : md
            # c : nd
            qflat = q.reshape(-1).cpu().numpy()
            cflat = c.reshape(-1).cpu().numpy()

            return np.abs(np.min(mts.mass2(cflat, qflat)))

            dist = []
            for i in range(0, len(qflat), b):
                dist.append(np.abs(np.min(mts.mass2(cflat, qflat[i:i+b]))))
            return np.mean(dist)
        

        mass_num_c = int(config['mass']['num_c'])
        mass_num_runs = int(config['mass']['num_runs'])
        mass_num_q = int(config['mass']['num_q'])

        inclusions = []
        for i in range(mass_num_q):
            inclusions += list(l[i].nonzero().squeeze().numpy())    
        inclusions = np.unique(inclusions)
        other_idxs = list(set(range(mass_num_c))-set(inclusions))

        ts = []
        map, mrr = [], []
        for _ in tqdm(range(mass_num_runs), desc="MASS"):
            mass_q = q.clone()[:mass_num_q]
            mass_q = mass_q + batch_get_white_noise(mass_q, args.SNR)
            mass_q = normalize(mass_q)

            c = test_dataset.c[inclusions]
            
            random_idxs = np.random.choice(other_idxs, size=mass_num_c-len(c), replace=False)
            c_ = test_dataset.c[random_idxs]
            mass_c = normalize(torch.from_numpy(np.vstack((c, c_))))

            mass_l = torch.hstack((l[:mass_num_q,inclusions],l[:mass_num_q,random_idxs]))
            sim = torch.zeros(mass_q.shape[0], mass_c.shape[0])

            start = time()
            for i, query in enumerate(mass_q):
                for j, corpus in enumerate(mass_c):
                    sim[i,j] = -mass_dist(corpus[:,::SKIP], query[:,::SKIP])
            map_, mrr_ = compute_map_mrr(sim, mass_l)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_)

        print_stats(ts, mass_num_runs, mass_num_c, mass_num_q, map, mrr)
        GLOBAL_TIMES.mass = get_time_entry(ts, mass_num_runs, mass_num_c, mass_num_q, map, mrr)

    
    with open("inference_times.pkl", "wb") as f:
        pickle.dump(GLOBAL_TIMES, f)
