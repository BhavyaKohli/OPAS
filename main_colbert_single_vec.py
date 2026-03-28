import h5py
import logging
import os, sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf

from opas.tstok.tsutils import TOKENIZER, tokenize, batch_get_white_noise
from opas.utils import AttributeDict, gumbel_sinkhorn, normalize, get_opas_constants, seed_everything, embed_full_corpus, embed_if_image_and_normalize, stagger_and_concat
from opas.data import PairDatasetTrain, PairDatasetTest

from opas.models.main import LamModel, ScoreModel, PositionalEncoding, TransformInput, DummyScoreModel, Attention_Layer
from opas.models.cifar_embed import Autoencoder
from opas.models.lsun_embed import Autoencoder as LSUNAutoencoder
from opas.models.deepset import DeepSetModel
from opas.models.sortlrl import SortLRL
from opas.models.model_utils import load_models

from functools import partial
from transformers import get_linear_schedule_with_warmup

from transformers import BertConfig
from colbert.modeling.colbert import ColBERT
from colbert.utils.amp import MixedPrecisionManager


tqdm = partial(tqdm, ncols=100)


@torch.no_grad()
def embed_full_corpus(dataset, colbert, image_embed_model=None, inner_batch_size=800, verbose=False):
    C = dataset.c
    if not isinstance(C, torch.Tensor):
        C = torch.from_numpy(C).float()    
    Cembed = []
    for batch in tqdm(range(0, len(C), inner_batch_size), disable=not verbose, leave=False, desc="Embedding..."):
        c = C[batch:batch+inner_batch_size].to(next(colbert.parameters()).device)
        c = embed_if_image_and_normalize(c, image_embed_model)[...,::samp_rate_step]
        c = torch.cat((colbert.doc_identifier.repeat(c.shape[0],1,1), c), dim=1)
        c = colbert.pretransform_to_bert(c)
        size_c = lambda: (len(c), c.shape[1])
        c.size = size_c
        c = colbert.doc(c, torch.ones(len(c), c.shape[1], device=c.device))[:,0,:]
        Cembed.append(c)
    C = torch.vstack(Cembed)
    return C


@torch.no_grad()
def compute_metrics(dataset, colbert, image_embed_model=None, stagger=2, verbose=False):

    loader = dataset.get_dataloader(batch_size=150, shuffle=True)

    C = embed_full_corpus(dataset, colbert, image_embed_model=image_embed_model)
    # C is (N, n, xoutdim)

    netscores = []
    true_labels = []
    for n, (q, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        q = q.to(next(colbert.parameters()).device) 
        # q is bmd, c is Nnd, l is bN
        q = q + batch_get_white_noise(q, args.SNR)     # torch.randn_like(q) * noise
        q = embed_if_image_and_normalize(q, image_embed_model)[...,::samp_rate_step]
        q = torch.cat((colbert.query_identifier.repeat(q.shape[0],1,1), q), dim=1)
        q = colbert.pretransform_to_bert(q)
        size_q = lambda: (len(q), q.shape[1])
        q.size = size_q
        q = colbert.query(q, torch.ones(len(q), q.shape[1], device=q.device))[:,0,:]

        # netscore = torch.einsum("bmd,Nnd->bNmn", q, C).max(-1).values.sum(-1)  # bNmn -> bNm -> bN
        netscore = q @ C.permute(1,0)   # bN
        netscores.append(netscore.to('cpu'))
        true_labels.append(l.to('cpu'))

    netscores = torch.vstack(netscores)
    true_labels = torch.vstack(true_labels).squeeze() # 

    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    MRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    MAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    MAP /= (torch.arange(ranked_output.shape[1]) + 1)
    MAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    MAP = MAP.sum(dim=1).mean().item()

    return MAP, MRR


if __name__ == '__main__':
    cli_conf = OmegaConf.from_cli()
    dataset = cli_conf.dataset
    if "audio" in dataset:
        spec_conf = OmegaConf.load("configs/audio.yaml")
    elif "speech" in dataset:
        spec_conf = OmegaConf.load("configs/speech.yaml")
    elif "cifar" in dataset:
        spec_conf = OmegaConf.load("configs/cifar.yaml")
    elif "lsun" in dataset:
        spec_conf = OmegaConf.load("configs/lsun.yaml")
    else:
        raise NotImplementedError(f"Check dataset name")
    
    if cli_conf.get("config", None):
        extra = OmegaConf.load(cli_conf.config)
        spec_conf = OmegaConf.merge(spec_conf, extra)

    base_conf = OmegaConf.load("configs/base.yaml")
    main_conf = OmegaConf.merge(base_conf, spec_conf, cli_conf)
    args = argparse.Namespace(**main_conf)

    DEVICE = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # torch.use_deterministic_algorithms(True)    #causes .backward() issues with adaptive pooling
    seed_everything(args.seed)

    experiment_id = datetime.now().strftime("%d%m%H%M%S")
    args.human = args.video = args.cifar = args.lsun = False
    image_embed_model = None
    if "audio" in args.dataset:
        experiment_id = f"A{experiment_id}" 
    elif "human" in args.dataset or "speech" in args.dataset:
        args.speech = True
        args.human = True
        experiment_id = f"S{experiment_id}"
    elif "video" in args.dataset:
        args.video = True
        experiment_id = f"V{experiment_id}"
    elif "cifar" in args.dataset:
        args.cifar = True
        experiment_id = f"C{experiment_id}" 
        image_embed_model_ckpt = "data/image_sequence/embedding_models/cifar_ae.pkl"
        image_embed_model = Autoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt, map_location=DEVICE))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

    elif "lsun" in args.dataset:
        args.lsun = True
        experiment_id = f"L{experiment_id}"

        image_embed_model_ckpt = "data/image_sequence/embedding_models/lsun_ae.pkl"
        image_embed_model = LSUNAutoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt, map_location=DEVICE))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

    else:
        raise NotImplementedError(f"Check dataset name")

    # args.debug = True   # debug mode perma on
    args.wandb_log = False
    TQDM_DISABLE = getattr(args, "tqdm_disable", False)

    EXPT_ROOT = f"models_colbert/{experiment_id}"
    if not args.debug:
        os.makedirs(f"{EXPT_ROOT}/", exist_ok=True)
        print(f"Experiment information at {EXPT_ROOT}")

        logging.basicConfig(
            filename=f"{EXPT_ROOT}/experiments.log",
            filemode="a+",
            level=logging.INFO,
            format="%(levelname)s (%(asctime)s): %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p"
        )
    
    logging.info("python " + " ".join(sys.argv))
    logging.info(f"Running with args: {args}")
    print(f"Running with args: {args}")

    NOLAMMODEL = args.no_lammodel

    if args.cifar or args.lsun:
        image_embed_model = image_embed_model.to(DEVICE)
    logging.info(f"Using device = {DEVICE}")
    EMBED = True
    DEEPSET = args.deepset

    # CFG for sinkhorn
    CFG = AttributeDict({
        'tau': 1,
        'n_sink_iter': args.n_sink_iter,
        'n_samples': 1,
    })

    PARAMS = AttributeDict({})

    logging.info(f"Dataset in use : {args.dataset}")

    if not args.debug:
        import pickle
        with open(f"{EXPT_ROOT}/args.pkl", "wb") as f:
            pickle.dump(args, f)
        with open(f"{EXPT_ROOT}/config.yaml", "w") as f:
            OmegaConf.save(main_conf, f)

    DATA_ROOT = f"final_data/{args.dataset}"

    TRAIN_FILE = f"{DATA_ROOT}/dataset_train.hdf5"
    TEST_FILE = f"{DATA_ROOT}/dataset_test.hdf5"
    VAL_FILE = f"{DATA_ROOT}/dataset_val.hdf5"

    if (args.cifar or args.lsun) and args.train_with_orig:
        TRAIN_FILE = f"{DATA_ROOT}/dataset_train_orig.hdf5"
        TEST_FILE = f"{DATA_ROOT}/dataset_test_orig.hdf5"
        VAL_FILE = f"{DATA_ROOT}/dataset_val_orig.hdf5"

    with h5py.File(TRAIN_FILE, "r") as dataset_file:
        if args.print_dataset : 
            for k in dataset_file.keys():
                print(k, dataset_file[k].shape)
            print("-"*80)
        M = dataset_file['Q'].shape[1]
        N = dataset_file['C'].shape[1]
        samp_rate = dataset_file['Q'].shape[-1]

    with h5py.File(TEST_FILE, "r") as dataset_file:
        if args.print_dataset : 
            for k in dataset_file.keys():
                print(k, dataset_file[k].shape)
            print("-"*80)
        assert dataset_file['Q'].shape[1]==M, 'M Shape mismatch in test'
        assert dataset_file['C'].shape[1]==N, 'N Shape mismatch in test'

    with h5py.File(VAL_FILE, "r") as dataset_file:
        if args.print_dataset : 
            for k in dataset_file.keys():
                print(k, dataset_file[k].shape)
        assert dataset_file['Q'].shape[1]==M, 'Shape mismatch in val'
        assert dataset_file['C'].shape[1]==N, 'N Shape mismatch in val'

    if N >= 50:
        print("Long sequence dataset. Please use main_long_seq.py instead. Main differences: redefined a_vec, LamModel4LongSeq, compute_metrics with smaller batch sizes and additional internal loops.")
        check = input(f"Delete expt dir at: {EXPT_ROOT}? y/n")
        if check.lower() == "y":
            import shutil
            shutil.rmtree(EXPT_ROOT)
        else:
            print(f"Keeping failed expt dir {EXPT_ROOT}")
        exit()
    
    # update params
    PARAMS.N = N            # length of corpus item
    PARAMS.M = M            # length of query
    PARAMS.samp_rate = samp_rate

    logging.info("*"*120+"\n"+"*"*120)

    A_mat, a_vec, Rm_mat = get_opas_constants(PARAMS.M, PARAMS.N, DEVICE)

    batch_size = args.batch_size
    positive_samples = 10

    neg_expl = args.neg_expl

    train_dataset = PairDatasetTrain(TRAIN_FILE, num_q=args.num_q, negative_exploration=neg_expl, seed=15)
    trainloader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    val_dataset = PairDatasetTest(VAL_FILE)
    valloader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    test_dataset = PairDatasetTest(TEST_FILE)
    testloader = test_dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    logging.info(f"Train dataset: {len(train_dataset)} query-corpus pairs, Val dataset: {len(val_dataset)} query-corpus pairs, Test dataset: {len(test_dataset)} queries")

    ##### HPARAMS ######
    b = args.b          # hinge margin for negative gap penalty (b-Apa)
    b1 = args.b1         # hinge margin for positive gap penalty (Apa-b)
    delta = args.delta     # hinge margin for contrastive loss
    lr = args.lr
    nepochs = args.nepochs
    stagger = args.stagger
    lamwt = args.lamwt    # loss coefficient for negative gap penalty
    gapwt = args.gapwt       # loss coefficient for positive gap penalty
    noise = args.noise
    internal_lamwt = getattr(args, "internal_lamwt", 1)
    ####################

    if args.wandb_log:
        import wandb 
        wandb.init(
            # set the wandb project where this run will be logged
            project = "Opas - Experiments",
            
            # track hyperparameters and run metadata
            config = {
                "learning_rate": lr,
                "dataset": f"{args.dataset}",
                "epochs": nepochs,
                "lamwt" : lamwt,
                "embed_model": EMBED,
                "delta" : delta,
                "stagger": stagger,
                "gapwt": gapwt,
                "xlr" : args.xlr,
                "xnumlayers" : args.xnumlayers,
                "xoutdim" : args.xoutdim,
                "xff" : args.xff,
                "noise": args.noise,
                "attention": args.add_attention,
                "single_xfmer": args.use_sing_xfmer,
                "noise SNR": args.SNR,
                "tokenize": not args.no_tokenize
            }
        )

    M, N = PARAMS.M, PARAMS.N

    samp_rate_step = getattr(args, "samp_rate_step", 1)
    samp_rate_override = int(samp_rate // samp_rate_step)

    tokenize_transform = lambda x: tokenize(x, args)[0].float()
    d_model = samp_rate_override
    lin_transform = nn.Sequential(
        nn.Linear(samp_rate, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
    )
    identity = lambda x: x

    if isinstance(image_embed_model, Autoencoder) or isinstance(image_embed_model, LSUNAutoencoder):
        d_model = 384

    colbert = ColBERT(
        BertConfig(hidden_size=d_model, num_attention_heads=args.xnhead, num_hidden_layers=args.xnumlayers),
        query_maxlen=M+1,
        doc_maxlen=N+1,
        dim=args.xoutdim,
        similarity_metric="cosine",
        mask_punctuation=False
    )
    colbert.pretransform_to_bert = TransformInput(tokenize_transform)
    colbert.register_buffer("query_identifier", nn.Parameter(torch.randn(1,d_model), requires_grad=True))
    colbert.register_buffer("doc_identifier", nn.Parameter(torch.randn(1,d_model), requires_grad=True))
    colbert = colbert.to(DEVICE)
    
    class Identity(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_ids, **kwargs):
            return input_ids.clone()

    def mask_doc(input_ids):
        return torch.ones(input_ids.shape[:2]).tolist()

    colbert.bert.embeddings = Identity()
    colbert.mask = mask_doc

    optimizer = torch.optim.AdamW(colbert.parameters(), lr=args.xlr, eps=1e-8)
    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(optimizer, args.xwarmupepochs * len(trainloader), nepochs * len(trainloader))

    logging.info(f"Trainable parameters: {sum(p.numel() for p in colbert.parameters() if p.requires_grad):,}")

    amp = MixedPrecisionManager(True)

    losslog = []
    logging.info("Training\n"+"*"*120+"\n"+"*"*120)
    pbar = tqdm(range(1,nepochs+1,1), disable=TQDM_DISABLE)
    best_val_MAP, best_val_MRR = 0, 0

    for i in pbar:
        if i == 1:
            # pass
            # sanity check on compute metrics
            _, _ = compute_metrics(val_dataset, colbert, image_embed_model=image_embed_model, stagger=stagger, verbose=False)
        
        colbert.train()
        wandb_losslog = []
        inner_pbar = tqdm(trainloader, disable=TQDM_DISABLE, leave=False)
        for n, (q, c, l) in enumerate(inner_pbar):
            qpos, cpos, lpos = train_dataset.get_positive_samples(positive_samples)
            q = torch.cat((q, qpos), dim=0).to(DEVICE)
            c = torch.cat((c, cpos), dim=0).to(DEVICE)
            l = torch.cat((l, lpos), dim=0)

            attn_qq = torch.ones(len(q), q.shape[1]).to(DEVICE)
            attn_cc = torch.ones(len(c), c.shape[1]).to(DEVICE)

            # add q and c identifiers:

            with amp.context():
                q = q + batch_get_white_noise(q, args.SNR)
                q = embed_if_image_and_normalize(q, image_embed_model)[...,::samp_rate_step]
                c = embed_if_image_and_normalize(c, image_embed_model)[...,::samp_rate_step]

                q = torch.cat((colbert.query_identifier.repeat(q.shape[0],1,1), q), dim=1)
                c = torch.cat((colbert.doc_identifier.repeat(c.shape[0],1,1), c), dim=1)

                q = colbert.pretransform_to_bert(q)
                c = colbert.pretransform_to_bert(c)

                size_q = lambda: (batch_size, M+1)
                size_c = lambda: (batch_size, N+1)
                q.size = size_q
                c.size = size_c

                # q, c, l are of shape bmd, bnd, b respectively
                Q = colbert.query(q, attn_qq)[:,0,:]
                D = colbert.doc(c, attn_cc)[:,0,:]      
                netscore = (Q @ D.T).diagonal()
                # netscore = (Q @ D.permute(0,2,1)).max(2).values.sum(1)   # b
        
                pos_score = netscore[torch.where(l==1)]
                neg_score = netscore[torch.where(l==0)]
                
                if not getattr(args, "train_with_labels", False):
                    neg_minus_pos = (neg_score.unsqueeze(0) - pos_score.unsqueeze(1)).reshape(-1)   # dim(pos_score) * dim(neg_score)
                    loss = F.relu(delta + neg_minus_pos).mean() 
                else:
                    # score == logit corresponding to probability 1 (match)
                    # softmax([0, score]) ==> [prob_0, prob_1]
                    # !!! ditch !!!

                    pos_score = torch.stack([torch.zeros_like(pos_score), pos_score]).T
                    neg_score = torch.stack([torch.zeros_like(neg_score), neg_score]).T
                    netscore = torch.cat([pos_score, neg_score], dim=0)
                    labels = torch.tensor([1]*len(pos_score) + [0]*len(neg_score)).long()

                    loss = F.cross_entropy(netscore.float(), labels.to(DEVICE), reduction="mean")

            amp.backward(loss)
            amp.step(colbert, optimizer)
            scheduler.step()

            pbar.set_postfix_str(f"loss: {loss:.4f}")
            wandb_losslog.append(loss.item())
            if args.wandb_log:
                wandb.log({"loss": loss.item()})


        # validation metrics and logging
        colbert.eval()
        MAP, MRR = compute_metrics(val_dataset, colbert, image_embed_model=image_embed_model, stagger=stagger, verbose=False)

        if MAP > best_val_MAP: 
            best_val_MAP = MAP
            val_MRR_at_best = MRR
            if not args.debug: torch.save(colbert.state_dict(), f"{EXPT_ROOT}/best.pkl")
        
        logging.info(f"[Epoch {i:2d}|{nepochs}] loss: {np.mean(wandb_losslog):.4f}, val MAP: {MAP:.4f}, val MRR: {MRR:.4f}, best MAP: {best_val_MAP:.4f}, MRR @ best val MAP: {val_MRR_at_best:.4f}")
    
        if args.wandb_log:
            wandb.log({"val MAP": MAP, "val MRR": MRR})
            wandb.log({"best val MAP": best_val_MAP, "best val MRR": val_MRR_at_best})
            wandb_losslog = []

        if not args.debug: torch.save(colbert.state_dict(), f"{EXPT_ROOT}/last.pkl")

    colbert.load_state_dict(torch.load(f"{EXPT_ROOT}/best.pkl", map_location=DEVICE))
    colbert.eval()

    MAP, MRR = compute_metrics(test_dataset, colbert, image_embed_model=image_embed_model, stagger=stagger, verbose=not TQDM_DISABLE)

    if getattr(args, "note", None):
        note = args.note
    else:
        note = ""        
    
    logging.info(f"Final test metrics: MAP,MRR: {MAP:.4f},{MRR:.4f}")
    if args.wandb_log: 
        wandb.log({"Test MAP": MAP, "Test MRR": MRR})
    print(f"Final test metrics {note}: MAP,MRR: {MAP:.4f},{MRR:.4f}")

    map_bst, mrr_bst = MAP, MRR

    colbert.load_state_dict(torch.load(f"{EXPT_ROOT}/last.pkl", map_location=DEVICE))
    colbert.eval()

    MAP, MRR = compute_metrics(test_dataset, colbert, image_embed_model=image_embed_model, stagger=stagger, verbose=not TQDM_DISABLE)
    
    logging.info(f"Final test metrics (last ckpt): MAP,MRR: {MAP:.4f},{MRR:.4f}")
    if args.wandb_log: 
        wandb.log({"Test MAP": MAP, "Test MRR": MRR})
    print(f"Final test metrics (last ckpt) {note}: MAP,MRR: {MAP:.4f},{MRR:.4f}")

    if note != "":
        with open("final_results_colbert_sv.txt", "a") as f:
            f.write(f"{note}: MAP, MRR (best): {map_bst:.4f}, {mrr_bst:.4f} | MAP, MRR (last): {MAP:.4f}, {MRR:.4f}\n")

    logging.info("*"*120+"\n"+"*"*120)