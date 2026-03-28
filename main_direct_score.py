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
from opas.models.sortlrl import SortLRL, LRLModel
from opas.models.model_utils import load_models

from functools import partial
from transformers import get_linear_schedule_with_warmup


tqdm = partial(tqdm, ncols=100)


class ScorerEmbedPhi(nn.Module):
    def __init__(self, d_model, args):
        super().__init__()
        self.embed_base = nn.Sequential(
            PositionalEncoding(d_model=d_model, max_seq_length=1000),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=args.xnhead, batch_first=True, dim_feedforward=args.xff), args.xnumlayers),
        )
        self.scorer = nn.Linear(d_model, 1)
    
    def forward(self, x):
        """
        input: (B x (M+N) x d_model)
        """
        x = self.embed_base(x)
        x = x.mean(dim=1)                   # B x d_model
        return self.scorer(x).squeeze()     # B


@torch.no_grad()
def compute_metrics(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False, aggregator=None):

    loader = dataset.get_dataloader(batch_size=150, shuffle=True)

    C = embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)
    # C is (N, n, xoutdim)

    netscores = []
    true_labels = []
    for n, (q, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        q = q.to(next(model.parameters()).device) 
        # q is bmd, c is Nnd, l is bN
        q = q + batch_get_white_noise(q, args.SNR)     # torch.randn_like(q) * noise
        q = embed_if_image_and_normalize(q, image_embed_model)
        q = embed_model(preembed_model(q))
        q = normalize(q)
        # q is (b, m, xoutdim)
        if aggregator is None:
            qct = torch.einsum("bmd,Nnd->bNmn", q, C)   # verified
            model_inputs = stagger_and_concat(qct, num_stagger=stagger)
            if stagger==0: 
                model_inputs = model_inputs.squeeze(1)
            lambdas = torch.stack([model(x) for x in model_inputs])
            
            F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))
            del qct

            P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)

            RmPC = Rm_mat @ P @ C.squeeze(-1)

            if args.no_lamrelu:
                lamscore = lamwt * (lambdas.transpose(2,3) @ (b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
            else:
                lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
            normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

            allscores = torch.stack([lamscore, normscore], dim=2)
            netscore = 2*scoremodel(-allscores).squeeze()      # b
            del P, F_mat, RmPC, q

        else:
            q = aggregator[0](q)
            # q is bd, C is Nd, we want bN scores
            # b1d - 1Nd = bNd --> sum across last dim to get bN scores
            netscore = 2 * F.sigmoid(-F.relu(q.unsqueeze(1) - C.unsqueeze(0)).sum(dim=-1))    # bN
            # TODO: fix this
            if args.deepset_mode == "cosine":     # 3
                netscore = 0.5 * (F.cosine_similarity(q.unsqueeze(1), C.unsqueeze(0), dim=-1) + 1)

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


@torch.no_grad()
def compute_metrics_direct_score_complete(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False, aggregator=None):
    C = torch.from_numpy(dataset.c).float()
    C = C.to(next(model.parameters()).device)
    C = normalize(C)

    loader = dataset.get_dataloader(batch_size=1, shuffle=True)
    inner_bsize = 800

    netscores = []
    true_labels = []
    for n, (q_, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        q_ = q_.to(next(model.parameters()).device) # 1xmxd
        q_ = q_ + batch_get_white_noise(q_, args.SNR)
        q_ = embed_if_image_and_normalize(q_, image_embed_model)
        q_ = normalize(q_)

        netscore = []
        for cbatch in range(0,len(C),inner_bsize):
            c = C[cbatch:cbatch+inner_bsize]
            c = embed_if_image_and_normalize(c, image_embed_model)
            q = q_.repeat_interleave(c.shape[0], dim=0)

            qc = torch.cat((q, c), dim=1)                   # inner_bsize x (6||20) x 4000
            netscore_ = embed_model(preembed_model(qc)).squeeze()     # inner_bsize
            
            netscore.append(netscore_)
        
        netscore = torch.cat(netscore).unsqueeze(0)  # 1 x num_c

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


@torch.no_grad()
def compute_metrics_direct_score(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False, aggregator=None):
    C = torch.from_numpy(dataset.c).float()
    C = C.to(next(model.parameters()).device)
    C = normalize(C)

    loader = dataset.get_dataloader(batch_size=1, shuffle=True)
    inner_bsize = 500

    netscores = []
    true_labels = []
    for n, (q_, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        q_ = q_.to(next(model.parameters()).device) # 1xmxd
        q_ = q_ + batch_get_white_noise(q_, args.SNR)
        q_ = embed_if_image_and_normalize(q_, image_embed_model)
        q_ = normalize(q_)

        samp_idxs = torch.randperm(len(C))[:inner_bsize]
        rand_c = C[samp_idxs]
        all_pos_c = C[torch.where(l[0])]
        
        c = torch.vstack((rand_c, all_pos_c))
        c = embed_if_image_and_normalize(c, image_embed_model)
        l = l[:,samp_idxs]      # 1 x (inner_bsize + num_pos)
        l = torch.cat((l, l.new_ones(1, len(all_pos_c))), dim=-1)

        q = q_.repeat_interleave(c.shape[0], dim=0)
        qc = torch.cat((q, c), dim=1)                   # inner_bsize x (6||20) x 4000
        netscore = embed_model(preembed_model(qc)).squeeze().unsqueeze(0)     # 1 x inner_bsize

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
    

def save_models(model, scoremodel, embed_model, preembed_model, aggregator=None, final=False, latest=False):
    if final:
        suffix = "_last"
        try:
            os.remove(f"{EXPT_ROOT}/model_latest.pt")
            os.remove(f"{EXPT_ROOT}/scmodel_latest.pt")
            os.remove(f"{EXPT_ROOT}/embed_model_latest.pt")
            if args.preembed != "tokenize":
                os.remove(f"{EXPT_ROOT}/preembed_model_latest.pt")
            if aggregator is not None:
                os.remove(f"{EXPT_ROOT}/aggregator_latest.pt")
        except:
            print("Couldn't delete `latest` models")
    elif latest:
        suffix = "_latest"
    else:
        suffix = ""
    
    logging.info(f"Saving models... experiment id: {experiment_id}")
    torch.save(model, f"{EXPT_ROOT}/model{suffix}.pt")
    torch.save(scoremodel, f"{EXPT_ROOT}/scmodel{suffix}.pt")
    torch.save(embed_model, f"{EXPT_ROOT}/embed_model{suffix}.pt")
    if args.preembed != "tokenize":
        torch.save(preembed_model, f"{EXPT_ROOT}/preembed_model{suffix}.pt")
    if aggregator is not None:
        torch.save(aggregator, f"{EXPT_ROOT}/aggregator{suffix}.pt")


if __name__ == '__main__':
    cli_conf = OmegaConf.from_cli()
    dataset = cli_conf.dataset
    if "audio" in dataset or "music" in dataset:
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

    experiment_id = datetime.now().strftime("%d%m%H%M")
    args.human = args.video = args.cifar = args.lsun = False
    image_embed_model = None
    if "audio" in args.dataset or "music" in args.dataset:
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

    EXPT_ROOT = f"models/{experiment_id}"
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
    else:
        print("Debug mode ON: Not saving logs or models")
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s (%(asctime)s): %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p"
        )
    
    logging.info("python " + " ".join(sys.argv))
    logging.info(f"Running with args: {args}")
    print(f"Running with args: {args}")

    if args.cifar or args.lsun:
        image_embed_model = image_embed_model.to(DEVICE)
    logging.info(f"Using device = {DEVICE}")
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
            project = "OPAS - Experiments",
            
            # track hyperparameters and run metadata
            config = {
                "learning_rate": lr,
                "dataset": f"{args.dataset}",
                "epochs": nepochs,
                "lamwt" : lamwt,
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

    tokenize_transform = lambda x: tokenize(x, args)[0]
    d_model = 256

    transform = tokenize_transform
    if args.lfcc:
        d_model = 320
    else:
        d_model = samp_rate
        if args.cifar:
            if isinstance(image_embed_model, Autoencoder):
                d_model = 384
            else:
                raise NotImplementedError
        elif args.lsun:
            if  isinstance(image_embed_model, LSUNAutoencoder):
                d_model = 384
            else:
                raise NotImplementedError    

    ################# Embedding flow: preembedding transform, embed model, embed optimizer and scheduler 
    preembed_model = TransformInput(transform).to(DEVICE)
    if args.preembed != "tokenize":
        preembed_optimizer = torch.optim.Adam(preembed_model.parameters(), amsgrad=True, lr=args.lr)

    embed_model = ScorerEmbedPhi(d_model, args).to(DEVICE)

    if args.skip_embed:
        d_model = args.xoutdim
        
        if args.skip_type == "lin":
            skip_transform = nn.Sequential(
                nn.Linear(samp_rate, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        elif args.skip_type == "conv":
            skip_transform = Conv1dTS(n_ch=4, latent=d_model)
        else:
            raise NotImplementedError(f"Skip type {args.skip_type} not implemented")
        
        args.no_tokenize = True     # turn off input tokenization
        with open(f"{EXPT_ROOT}/args.pkl", "wb") as f:
            pickle.dump(args, f)

        embed_model = TransformInput(skip_transform)
        embed_model = embed_model.to(DEVICE)
        preembed_model = TransformInput(nn.Identity())
        preembed_model.dummy_param = nn.Parameter(torch.empty(20), requires_grad=True)
        preembed_model = preembed_model.to(DEVICE)

    embed_optimizer = torch.optim.AdamW(embed_model.parameters(), lr=args.xlr,)
    scheduler = get_linear_schedule_with_warmup(embed_optimizer, args.xwarmupepochs * len(trainloader), nepochs * len(trainloader))
    loss_backward_interval = 1 # int(len(trainloader) / 4) # args.xnumlayers)

    ################# Optional aggregator (deepset) used after embed model, instead of OPAS QC^T scoring 
    if DEEPSET:
        use_sort_lrl = getattr(args, "use_sort_lrl", False)
        use_lrl_embed = getattr(args, "use_lrl_embed", False)
        if use_sort_lrl:
            qsort = SortLRL(args.xoutdim, seq_len=M, latent=args.xff, outdim=args.xoutdim).to(DEVICE)
            csort = SortLRL(args.xoutdim, seq_len=N, latent=args.xff, outdim=args.xoutdim).to(DEVICE)
            aggregator = nn.ModuleList([qsort, csort])
        elif use_lrl_embed:
            qembed = LRLModel(indim=args.xoutdim, seq_len=M, latent=args.xff, outdim=args.xoutdim).to(DEVICE)
            cembed = LRLModel(indim=args.xoutdim, seq_len=N, latent=args.xff, outdim=args.xoutdim).to(DEVICE)
            aggregator = nn.ModuleList([qembed, cembed])
        else:
            deepset = DeepSetModel(indim=args.xoutdim, latent=args.xoutdim//2, outdim=args.xoutdim//2).to(DEVICE)
            aggregator = nn.ModuleList([deepset, deepset])
        aggregator_optimizer = torch.optim.Adam(aggregator.parameters(), lr=getattr(args, "agglr", args.lr), weight_decay=1e-5)
    else:
        aggregator = None

    ################# Scoremodel and Lammodel, and their respective optimizers (weight decay and amsgrad are mostly randomly turned on due to some extremely old experiments, should not affect training)

    scoremodel = ScoreModel().to(DEVICE)
    model = LamModel(M, N, stagger).to(DEVICE)
    if getattr(args, "fix_lambdas", None) is not None:
        model = DummyLamModel(M, value=args.fix_lambdas).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    sc_optimizer = torch.optim.Adam(scoremodel.parameters(), lr=lr, amsgrad=True, weight_decay=1e-2)

    losslog = []
    logging.info("Training\n"+"*"*120+"\n"+"*"*120)

    pbar = tqdm(range(1,nepochs+1,1), disable=False)
    best_val_MAP, best_val_MRR = 0, 0

    for i in pbar:
        if i == 1:
            # sanity check on compute metrics
            _, _ = compute_metrics_direct_score(val_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)
        
        model.train(), scoremodel.train(), embed_model.train(), preembed_model.train()
        if DEEPSET:
            aggregator.train()

        wandb_losslog = []
        inner_pbar = tqdm(trainloader, disable=False, leave=False)
        for n, (q, c, l) in enumerate(inner_pbar):
            qpos, cpos, lpos = train_dataset.get_positive_samples(positive_samples)
            q = torch.cat((q, qpos), dim=0).to(DEVICE)
            c = torch.cat((c, cpos), dim=0).to(DEVICE)
            l = torch.cat((l, lpos), dim=0)
            q = q + batch_get_white_noise(q, args.SNR)
            q, c = embed_if_image_and_normalize(q, image_embed_model), embed_if_image_and_normalize(c, image_embed_model)

            qc = torch.cat((q, c), dim=1)
            netscore = embed_model(preembed_model(qc))
            # import ipdb; ipdb.set_trace()

            pos_score = netscore[torch.where(l==1)]
            neg_score = netscore[torch.where(l==0)]
            
            neg_minus_pos = (neg_score.unsqueeze(0) - pos_score.unsqueeze(1)).reshape(-1)   # dim(pos_score) * dim(neg_score)

            loss = F.relu(delta + neg_minus_pos).mean() 

            optimizer.zero_grad()
            sc_optimizer.zero_grad()
            if not args.pretrain_embedding:
                embed_optimizer.zero_grad()
            if args.preembed != "tokenize": 
                preembed_optimizer.zero_grad()
            if DEEPSET:
                aggregator_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sc_optimizer.step()
            if not args.pretrain_embedding:
                embed_optimizer.step()
                scheduler.step()
            if args.preembed != "tokenize": 
                preembed_optimizer.step()
            if DEEPSET:
                aggregator_optimizer.step()

            pbar.set_postfix_str(f"loss: {loss:.4f}")
            wandb_losslog.append(loss.item())
            if args.wandb_log:
                wandb.log({"loss": loss.item()})

        ################# Validation metrics and logging 
        model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
        if DEEPSET:
            aggregator.eval()
        MAP, MRR = compute_metrics_direct_score(val_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=False, aggregator=aggregator)

        if MAP > best_val_MAP: 
            best_val_MAP = MAP
            val_MRR_at_best = MRR
            if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator)
            
        logging.info(f"[Epoch {i:2d}|{nepochs}] loss: {np.mean(wandb_losslog):.4f}, val MAP: {MAP:.4f}, val MRR: {MRR:.4f}, best MAP: {best_val_MAP:.4f}, MRR @ best val MAP: {val_MRR_at_best:.4f}")
    
        if args.wandb_log:
            wandb.log({"val MAP": MAP, "val MRR": MRR})
            wandb.log({"best val MAP": best_val_MAP, "best val MRR": val_MRR_at_best})
            wandb_losslog = []

        ################# Save models if not in debug mode
        if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator, latest=True)

    ################# Loading saved models and running test-set evaluations 
    model, scoremodel, embed_model, preembed_model, aggregator = load_models(name="best", expt_root=EXPT_ROOT, device=DEVICE, args=args)
    model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
    if DEEPSET:
        aggregator.eval()

    MAP, MRR = compute_metrics_direct_score_complete(test_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)
    
    logging.info(f"Final test metrics: MAP,MRR: {MAP:.4f},{MRR:.4f}")
    if args.wandb_log: 
        wandb.log({"Test MAP": MAP, "Test MRR": MRR})
    print(f"Final test metrics: MAP,MRR: {MAP:.4f},{MRR:.4f}")

    if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator, final=True)