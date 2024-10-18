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
from main import *

from time import time
from argparse import Namespace as AttributeDict
from omegaconf import OmegaConf


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
    logging.basicConfig(
        filename=f"inference.log",
        filemode="a+",
        level=logging.INFO,
        format="%(levelname)s (%(asctime)s): %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p"
    )
    config = OmegaConf.load("configs/main.config")

    parser = argparse.ArgumentParser('Inference Time Comparisons')
    parser.add_argument("--expt_id", type=str, help="Experiment ID")
    parser.add_argument("--skip", type=int, help="skip value", default=10)
    parser.add_argument("--device", type=int, help="cuda device, pass -1 for cpu", default=-1)
    parser.add_argument("--skip_baselines", action="store_true", help="pass when only OPAS numbers are required")
    args = parser.parse_args()
    
    if args.skip_baselines:
        for baseline in ["sharp", "fdtw", "sdtw", "mass"]:
            config[baseline]['skip'] = 1
        logging.info(f"Skipping all baselines")
    else:
        SKIP = args.skip
        logging.info(f"RUNNING BASELINES WITH SKIP: {SKIP}")

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
    DEVICE = f'cuda:{args.device}' if args.device != -1 else "cpu"

    import pickle
    with open(root+"args.pkl", "rb") as f:
        args = pickle.load(f)

    TEST_FILE = f"../final_data/{args.dataset}/dataset_test.hdf5"
    if experiment_id[0] in ["C", "L"]:
        image_embed_model = image_embed_model.to(DEVICE)
        TEST_FILE = TEST_FILE.replace(".hdf5", "_orig.hdf5")
    print(TEST_FILE)

    logging.info(f"Running inference on device {DEVICE} using models from {root} using dataset {args.dataset} found at {TEST_FILE}")

    test_dataset = PairDatasetTest(TEST_FILE)

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

    ######################################################
    ######### CORPUS EMBEDDING FOR opas ##################
    ######################################################
    num_runs = int(config['opas_cembed']['num_runs'])

    ts = []
    for _ in tqdm(range(num_runs), desc="Corpus Embedding"):
        start = time()
        C = embed_full_corpus(test_dataset, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator, inner_batch_size=800)
        stop = time()
        ts.append(stop-start)

    mu, sig = get_mean_std(ts)
    GLOBAL_TIMES.corpus_embedding = get_time_entry(ts, num_runs)
    logging.info(f"Corpus embedding cost for {len(test_dataset.c)} corpus items (one-time cost): {mu:.4f}±{sig:.4f} s (averaged over {num_runs} runs)")

    logging.info("\nRunning OPAS (GPU)")

    A_mat, a_vec, Rm_mat = get_opas_constants(M, N, DEVICE)

    num_c = len(test_dataset.c)
    num_q = int(config['opas_gpu']['num_q'])
    num_runs = int(config['opas_gpu']['num_runs'])
    
    torch.manual_seed(6969)
    loader = test_dataset.get_dataloader(batch_size=num_q, shuffle=True)
    q, l = next(iter(loader))

    ts = []
    map, mrr = [], []
    for _ in tqdm(range(num_runs), desc="OPAS (GPU)"):
        start = time()
        netscore, true_labels = get_batch_scores(q, l, C, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator, stagger=stagger)
        map_, mrr_ = compute_map_mrr(netscore, true_labels)
        stop = time()

        ts.append(stop-start)
        map.append(map_)
        mrr.append(mrr_)

    print_stats(ts, num_runs, num_c, num_q, map, mrr)
    GLOBAL_TIMES.opas_gpu = get_time_entry(ts, num_runs, num_c, num_q, map, mrr) 

    ## opas CPU

    if config['opas_cpu'].get('skip', 0)==0:
        logging.info("\n\nRunning OPAS (CPU)")

        num_runs = int(config['opas_cpu']['num_runs'])

        testdevice = 'cpu'
        model, scoremodel, embed_model, preembed_model = model.to(testdevice), scoremodel.to(testdevice), embed_model.to(testdevice), preembed_model.to(testdevice)
        if aggregator is not None:
            aggregator = aggregator.to(testdevice)
        A_mat, a_vec, Rm_mat = get_opas_constants(M, N, testdevice)

        ts = []
        map, mrr = [], []
        for _ in tqdm(range(num_runs), desc="OPAS (CPU)"):
            start = time()
            netscore, true_labels = get_batch_scores(q, l, C.to('cpu'), model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator, stagger=stagger)
            map_, mrr_ = compute_map_mrr(netscore, true_labels)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_)

        print_stats(ts, num_runs, num_c, num_q, map, mrr)
        GLOBAL_TIMES.opas_cpu = get_time_entry(ts, num_runs, num_c, num_q, map, mrr)


    def run_baseline(name, distance_function, num_c, num_runs, num_q):
        logging.info(f"\nRunning {name}")

        inclusions = []
        for i in range(num_q):
            inclusions += list(l[i].nonzero().squeeze().numpy())    
        inclusions = np.unique(inclusions)
        other_idxs = list(set(range(num_c))-set(inclusions))

        ts = []
        map, mrr = [], []
        for _ in tqdm(range(num_runs), desc=name.upper()):
            method_q = q.clone()[:num_q].flatten(start_dim=2)
            method_q = method_q + batch_get_white_noise(method_q, args.SNR)
            method_q = normalize(method_q)

            c = test_dataset.c[inclusions]
            
            random_idxs = np.random.choice(other_idxs, size=num_c-len(c), replace=False)
            c_ = test_dataset.c[random_idxs]
            method_c = normalize(torch.from_numpy(np.vstack((c, c_)))).flatten(start_dim=2)

            method_l = torch.hstack((l[:num_q,inclusions],l[:num_q,random_idxs]))
            sim = torch.zeros(method_q.shape[0], method_c.shape[0])

            start = time()
            for i, query in enumerate(method_q):
                for j, corpus in enumerate(method_c):
                    sim[i,j] = -distance_function(corpus.cpu().numpy()[:,::SKIP], query.cpu().numpy()[:,::SKIP])
            map_, mrr_ = compute_map_mrr(sim, method_l)
            stop = time()

            t = (stop-start)
            ts.append(t)
            map.append(map_)
            mrr.append(mrr_)

        print_stats(ts, num_runs, num_c, num_q, map, mrr)
        setattr(GLOBAL_TIMES, name, get_time_entry(ts, num_runs, num_c, num_q, map, mrr))


    #######################################################
    ################## SHARP ##############################
    #######################################################

    if config['sharp'].get('skip', 0)==0:
        from sharp import sharp_sdtw_div

        sharp_num_c = len(test_dataset.c)
        sharp_num_runs = int(config['sharp']['num_runs'])
        sharp_num_q = int(config['sharp']['num_q'])

        run_baseline("SHARP", sharp_sdtw_div, sharp_num_c, sharp_num_runs, sharp_num_q)

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

        run_baseline("FASTDTW", dtw_func, dtw_num_c, dtw_num_runs, dtw_num_q)

    #######################################################
    ##################### SDTW ############################
    #######################################################

    if config['sdtw'].get('skip', 0)==0:
        logging.info("\n\nRunning SDTW")

        from sdtw import sdtw_div

        sdtw_num_c = len(test_dataset.c)
        sdtw_num_runs = int(config['sdtw']['num_runs'])
        sdtw_num_q = int(config['sdtw']['num_q'])

        run_baseline("SDTW", sdtw_div, sdtw_num_c, sdtw_num_runs, sdtw_num_q)

    #######################################################
    ################## MASS ###############################
    #######################################################

    if config['mass'].get('skip', 0)==0:
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

        run_baseline("MASS", mass_dist, mass_num_c, mass_num_runs, mass_num_q)

    torch.save(GLOBAL_TIMES, "inference_times.pkl")