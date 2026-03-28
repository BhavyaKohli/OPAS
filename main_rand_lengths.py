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
# from opas.models.ts_encoders import Conv1dTS, EncConv1dTS, MelConv1dTS
from opas.models.deepset import DeepSetModel
from opas.models.sortlrl import SortLRL
from opas.models.model_utils import load_models

from functools import partial
from transformers import get_linear_schedule_with_warmup#, AdamW

# import torchaudio.transforms as T


tqdm = partial(tqdm, ncols=100)


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
            if not NOLAMMODEL:
                qct = torch.einsum("bmd,Nnd->bNmn", q, C)   # verified
                if args.use_linear_lammodel:
                    lambdas = []
                    for xx in range(len(q)):
                        q_ = q[xx].unsqueeze(0)
                        q_ = q_.repeat_interleave(C.shape[0], dim=0)
                        lambdas_ = model(torch.cat((q_, C), dim=1).flatten(start_dim=1)).unsqueeze(-1)
                        lambdas.append(lambdas_)
                    lambdas = torch.stack(lambdas)
                else:
                    model_inputs = stagger_and_concat(qct, num_stagger=stagger)
                    if stagger==0: 
                        model_inputs = model_inputs.squeeze(1)
                    lambdas = torch.stack([model(x) for x in model_inputs])
                
                F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))
                del qct
            else:
                F_mat = Rm_mat.T @ (
                    torch.stack([-(q[i][None].unsqueeze(2) - C.unsqueeze(1)).relu().sum(-1) for i in range(len(q))])
                )
                lambdas = torch.ones((len(q), len(C), M, 1), device=F_mat.device)

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
def compute_metrics_sing(dataset, model, scoremodel, embed_model, preembed_model, stagger=2, verbose=False):
    C = torch.from_numpy(dataset.c).float()
    C = C.to(next(model.parameters()).device)
    C = normalize(C)

    loader = dataset.get_dataloader(batch_size=1, shuffle=True)
    inner_bsize = 100

    netscores = []
    true_labels = []
    for n, (q_, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        q_ = q_.to(next(model.parameters()).device) # 1xmxd
        q_ = q_ + batch_get_white_noise(q_, args.SNR)
        q_ = normalize(q_)

        for cbatch in range(0,len(C),inner_bsize):
            c = C[cbatch:cbatch+inner_bsize]
            l_ = l[:, cbatch:cbatch+inner_bsize]
            q = q_.repeat_interleave(c.shape[0], dim=0)

            qc = torch.cat((q, c), dim=1)
            qc = embed_model(preembed_model(qc))
            q, c = qc[:, :M], qc[:, M:]
            q, c = normalize(q, c)

            qct = torch.einsum("bmd,bnd->bmn", q, c)    # verified
            model_inputs = stagger_and_concat(qct, num_stagger=stagger)
            
            lambdas = model(model_inputs)

            F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(1,2) @ A_mat).transpose(1,2))

            P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)
            # RmPC = torch.einsum("mn,bnn,bnd->bmd", Rm_mat, P, c)    # sanity fail
            RmPC = Rm_mat @ torch.bmm(P, c)

            lamscore = lamwt * (lambdas.transpose(1,2) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
            normscore = torch.norm(q - RmPC, dim=[1,2])

            allscores = torch.stack([lamscore, normscore], dim=1)
            netscore = 2*scoremodel(-allscores).squeeze()

            netscores.append(netscore.to('cpu'))
            true_labels.append(l_.to('cpu'))

    netscores = torch.vstack(netscores[:-1])
    true_labels = torch.vstack(true_labels[:-1]).squeeze() # 

    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    MRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    MAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    MAP /= (torch.arange(ranked_output.shape[1]) + 1)
    MAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    MAP = MAP.sum(dim=1).mean().item()

    return MAP, MRR


@torch.no_grad()
def compute_odc(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False):

    loader = dataset.get_dataloader(batch_size=1, shuffle=True)

    num_samples = 20
    C = embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model=image_embed_model)
    # C is (N, n, xoutdim)

    odc = []
    for n, (q, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        if n > 100: break
        q_, l_ = q[0], l[0]

        shuffles = torch.stack([torch.arange(len(q_))] + [s for s in [torch.randperm(len(q_),) for _ in range(num_samples)] if not torch.equal(s, torch.arange(len(q_)))]) 

        q = q_[shuffles]
        l = l_[None].repeat_interleave(len(q_), dim=0)
        true_c = C[torch.where(l[0])[0]]

        q = q.to(next(model.parameters()).device) 
        # q is bmd, c is Nnd, l is bN
        q = q + batch_get_white_noise(q, args.SNR)     # torch.randn_like(q) * noise
        q = embed_if_image_and_normalize(q, image_embed_model)
        q = embed_model(preembed_model(q))
        q = normalize(q)
        # q is (b, m, xoutdim)

        qct = torch.einsum("bmd,Nnd->bNmn", q, true_c)  # verified
        model_inputs = stagger_and_concat(qct, num_stagger=stagger) # bNsmn  s = num_stagger+1

        lambdas = torch.stack([model(x) for x in model_inputs])
        # bNm1

        F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))

        P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)

        RmPC = Rm_mat @ P @ true_c.squeeze(-1)

        lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
        normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

        allscores = torch.stack([lamscore, normscore], dim=2)
        netscore = 2*scoremodel(-allscores).squeeze()      # b
        
        pos = netscore[0]
        neg = netscore[1:]
        diff = (pos - neg)

        odc.append((100 * (pos-neg > 0).sum() / np.prod(diff.shape)).item())

    return np.mean(odc)


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


class LRLModel(nn.Module):
    def __init__(self, indim, seq_len, latent, outdim):
        super().__init__()
        self.lrl = nn.Sequential(
            nn.Linear(indim * seq_len, latent),
            nn.ReLU(),
            nn.Linear(latent, outdim)
        )
    
    def forward(self, x):
        return self.lrl(x.flatten(start_dim=1))


class DummyLamModel(nn.Module):
    def __init__(self, M, value=1):
        super().__init__()
        self.M = M
        self.value = value
        self.dummy_param = nn.Parameter(torch.empty(1), requires_grad=True)
        
    def forward(self, x):
        return self.value * torch.ones((x.shape[0], self.M, 1), device=x.device)


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

    experiment_id = datetime.now().strftime("%d%m%H%M")
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

    if getattr(args, "dummy_scoremodel", False):
        scoremodel = DummyScoreModel().to(DEVICE)
    else:
        scoremodel = ScoreModel().to(DEVICE)

    tokenize_transform = lambda x: tokenize(x, args)[0]
    d_model = 256
    lin_transform = nn.Sequential(
        nn.Linear(samp_rate, d_model),
        nn.Tanh(),
        nn.Linear(d_model, d_model),
    )
    identity = lambda x: x
    # conv_transform = Conv1dTS(n_ch=4, latent=d_model)
    # conv_transform = EncConv1dTS(latent=d_model, samp_rate=samp_rate)
    # conv_transform = MelConv1dTS(latent=d_model, samp_rate=samp_rate)
    
    if args.preembed == "tokenize":
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
    elif args.preembed == "conv":
        transform = conv_transform
    elif args.preembed == "linear":
        transform = lin_transform
    else:
        raise NotImplementedError(f"Pre-embed model {args.preembed} not implemented")

    preembed_model = TransformInput(transform).to(DEVICE)
    if args.preembed != "tokenize":
        preembed_optimizer = torch.optim.Adam(preembed_model.parameters(), amsgrad=True, lr=args.lr)

    embed_model = nn.Sequential(
        PositionalEncoding(d_model=d_model, max_seq_length=500),
        nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=args.xnhead, batch_first=True, dim_feedforward=args.xff), args.xnumlayers),
        nn.Linear(d_model, args.xoutdim)
    ).to(DEVICE)

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

    if args.add_attention:
        # unused in current script version
        attention_model = Attention_Layer(M+N).to(DEVICE)
        attend_optimizer = torch.optim.Adam(attention_model.parameters(), lr=args.xlr, weight_decay=1e-5)
    else:
        attention_model = None

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

    model = LamModel(M, N, stagger).to(DEVICE)
    if args.use_linear_lammodel:
        model = nn.Sequential(nn.Linear((M+N)*args.xoutdim, M), nn.Sigmoid()).to(DEVICE)
    if getattr(args, "fix_lambdas", None) is not None:
        model = DummyLamModel(M, value=args.fix_lambdas).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    sc_optimizer = torch.optim.Adam(scoremodel.parameters(), lr=lr, amsgrad=True, weight_decay=1e-2)

    # tf = T.LFCC(sample_rate=4000, n_lfcc=40, log_lf=False, speckwargs={"n_fft": 1024}).to(DEVICE)
    
    ################ NOT USED ######################
    if args.lfcc: 
        TRANSFORM = lambda x: tf(x).flatten(start_dim=-2)
    else:
        TRANSFORM = lambda x: x
    ################ NOT USED ######################

    if args.pretrain_embedding:
        from copy import deepcopy
        from torch.utils.data import Dataset, DataLoader

        embed_model.add_module(
            "reconstruction", 
            nn.Sequential(
                nn.Linear(args.xoutdim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            ).to(DEVICE)
        )

        logging.info("Pretraining embedding model")
        qsamples = np.random.choice(range(len(train_dataset.q)), int(len(train_dataset.q) * args.pretrain_budget), replace=False)
        csamples = np.random.choice(range(len(train_dataset.c)), int(len(train_dataset.c) * args.pretrain_budget), replace=False)
        pretrain_q = torch.from_numpy(train_dataset.q[qsamples])
        pretrain_c = torch.from_numpy(train_dataset.c[csamples])

        class PairDatasetPretrain(Dataset):
            def __init__(self, q, c):
                self.q = q
                self.c = c

            def __len__(self):
                return len(self.q) * len(self.c)

            def __getitem__(self, idx):
                id1 = idx // len(self.c)
                id2 = idx % len(self.c)
                return self.q[id1], self.c[id2]

        pretrain_dataset = PairDatasetPretrain(pretrain_q, pretrain_c)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=800, shuffle=True, num_workers=16)

        bestloss = 10
        bestwts = None
        es = 0
        pbar = tqdm(range(1,args.pretrain_epochs+1,1), disable=False)
        for epoch in pbar:
            eloss = []
            for q, c in tqdm(pretrain_loader, leave=False):
                q, c = q.to(DEVICE), c.to(DEVICE)
                q = q + batch_get_white_noise(q, args.SNR)
                qorig, corig = embed_if_image_and_normalize(q, image_embed_model), embed_if_image_and_normalize(c, image_embed_model)

                q, c = embed_model(preembed_model(qorig)), embed_model(preembed_model(corig))
                q, c = normalize(q, c)
                loss = F.mse_loss(qorig, q) + F.mse_loss(corig, c)
                
                embed_optimizer.zero_grad()
                if args.preembed != "tokenize": 
                    preembed_optimizer.zero_grad()
                loss.backward()
                embed_optimizer.step()
                if args.preembed != "tokenize": 
                    preembed_optimizer.step()
                scheduler.step()

                pbar.set_postfix_str(f"loss: {loss:.4f}, bestloss: {bestloss:.4f}, es: {es}")
                eloss.append(loss.item())

            if np.mean(eloss) <= bestloss:
                bestloss = np.mean(eloss)
                bestwts = deepcopy(embed_model.state_dict())
            else:
                es += 1
                if es >= 5:
                    break

        logging.info(f"Pretraining done")
        embed_model.load_state_dict(bestwts)
        embed_model.reconstruction = nn.Identity()
        embed_model.eval()
        for param in embed_model.parameters():
            param.requires_grad = False

    losslog = []

    logging.info("Training\n"+"*"*120+"\n"+"*"*120)

    pbar = tqdm(range(1,nepochs+1,1), disable=False)
    best_val_MAP, best_val_MRR = 0, 0

    enforce_order = args.enforce_order

    for i in pbar:
        # if i == 1:
        #     if not args.use_sing_xfmer:
        #         # sanity check on compute metrics
        #         _, _ = compute_metrics(val_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)
        
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
            if enforce_order:
                qneg = []   # shuffled
                cneg = []   # repeats
                lneg = []   # 0
                for q_,c_,l_ in zip(qpos,cpos,lpos):
                    for _ in range(5):
                        qneg.append(q_[torch.randperm(q_.shape[0])])
                        cneg.append(c_)
                        lneg.append(torch.zeros_like(l_))
                qneg, cneg, lneg = torch.stack(qneg).to(DEVICE), torch.stack(cneg).to(DEVICE), torch.stack(lneg)
                q = torch.cat((q, qneg), dim=0)
                c = torch.cat((c, cneg), dim=0)
                l = torch.cat((l, lneg), dim=0)

            cmask = torch.isclose(c, torch.tensor(0.)).sum(dim=-1) != samp_rate


            q = q + batch_get_white_noise(q, args.SNR)
            q, c = embed_if_image_and_normalize(q, image_embed_model), embed_if_image_and_normalize(c, image_embed_model)

            if args.use_sing_xfmer:
                qc = torch.cat((q, c), dim=1)
                qc = embed_model(preembed_model(qc))
                q, c = qc[:, :M], qc[:, M:]
                q, c = normalize(q, c)
            else:
                # main forward pass, !! Change this in compute_metrics as well !!
                q = embed_model(preembed_model(q))
                c = embed_model(preembed_model(c))
                # c = embed_model[0](c)
                # c = embed_model[1](c, mask=cmask)
                # c = embed_model[2](c)
                
                q, c = normalize(q, c)
                c = c * cmask.unsqueeze(-1)

            # import ipdb; ipdb.set_trace()

            if not DEEPSET:
                if not NOLAMMODEL:
                    qct = torch.einsum("bmd,bnd->bmn", q, c)
                    
                    if args.use_linear_lammodel:
                        lambdas = model(torch.cat((q, c), dim=1).flatten(start_dim=1)).unsqueeze(-1)
                    else:
                        model_inputs = stagger_and_concat(qct, num_stagger=stagger)
                        if stagger==0: 
                            model_inputs = model_inputs.squeeze(1)
                        lambdas = model(model_inputs)

                    F_mat = Rm_mat.T @ (2*qct + internal_lamwt * (a_vec @ lambdas.transpose(1,2) @ A_mat).transpose(1,2))
                else:
                    F_mat = Rm_mat.T @ -(q.unsqueeze(2) - c.unsqueeze(1)).relu().sum(-1)
                    lambdas = torch.ones((q.shape[0], M, 1), device=DEVICE)
                
                if args.single_step_norm == 1:
                    P = F_mat / F_mat.sum(dim=-2, keepdims=True)    # normalizing on the row dimension
                elif args.single_step_norm == 2:
                    P = F_mat.exp() / F_mat.exp().sum(dim=-2, keepdims=True)
                else:
                    P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)

                # RmPC = torch.einsum("mn,bnn,bnd->bmd", Rm_mat, P, c)  # sanity fail
                RmPC = Rm_mat @ torch.bmm(P, c)

                if args.no_lamrelu:
                    lamscore = lamwt * (lambdas.transpose(1,2) @ (b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
                else:
                    lamscore = lamwt * (lambdas.transpose(1,2) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
                normscore = torch.norm(q - RmPC, dim=[1,2])

                allscores = torch.stack([lamscore, normscore], dim=1)
                netscore = 2*scoremodel(-allscores).squeeze()
            else:
                q, c = aggregator[0](q), aggregator[1](c)
                if args.deepset_mode == "normalized":   # 2
                    netscore = 2 * F.sigmoid(-F.relu(q - c).sum(dim=-1))    # normalized to 0-1, 0 for worst, 1 for best (==0 loss)
                elif args.deepset_mode == "base":       # 1
                    netscore = -F.relu(q - c).sum(dim=-1)                   # un-normalized scores
                elif args.deepset_mode == "cosine":     # 3
                    netscore = 0.5 * (F.cosine_similarity(q, c, dim=-1) + 1)

            if not getattr(args, "train_with_labels", False):
                pos_score = netscore[torch.where(l==1)]
                neg_score = netscore[torch.where(l==0)]
                
                neg_minus_pos = (neg_score.unsqueeze(0) - pos_score.unsqueeze(1)).reshape(-1)   # dim(pos_score) * dim(neg_score)

                loss = F.relu(delta + neg_minus_pos).mean() 
                if gapwt > 0 and not DEEPSET:
                    loss += gapwt * F.relu(A_mat @ Rm_mat @ P @ a_vec - b1).mean()
            else:
                loss = F.binary_cross_entropy(netscore, l.to(DEVICE), reduction="mean")

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


        # validation metrics and logging
        model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
        if DEEPSET:
            aggregator.eval()
        if not args.use_sing_xfmer:
            MAP, MRR = compute_metrics(val_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)

            if MAP > best_val_MAP: 
                best_val_MAP = MAP
                val_MRR_at_best = MRR
                if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator)

            if enforce_order:
                odc = compute_odc(val_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True)
                logging.info(f"odc at epoch: {i:2d} = {odc:.2f}")
                if args.wandb_log:
                    wandb.log({"odc": odc})
        else:
            MAP, MRR = 0, 0
            val_MRR_at_best = 0
            if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator)
            
        logging.info(f"[Epoch {i:2d}|{nepochs}] loss: {np.mean(wandb_losslog):.4f}, val MAP: {MAP:.4f}, val MRR: {MRR:.4f}, best MAP: {best_val_MAP:.4f}, MRR @ best val MAP: {val_MRR_at_best:.4f}")
    
        if args.wandb_log:
            wandb.log({"val MAP": MAP, "val MRR": MRR})
            wandb.log({"best val MAP": best_val_MAP, "best val MRR": val_MRR_at_best})
            wandb_losslog = []

        if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator, latest=True)

    model, scoremodel, embed_model, preembed_model, aggregator = load_models(name="best", expt_root=EXPT_ROOT, device=DEVICE, args=args)
    model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
    if DEEPSET:
        aggregator.eval()

    if args.use_sing_xfmer:
        MAP, MRR = compute_metrics_sing(test_dataset, model, scoremodel, embed_model, preembed_model, stagger=stagger, verbose=True)
    else:
        MAP, MRR = compute_metrics(test_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)
    
    logging.info(f"Final test metrics: MAP,MRR: {MAP:.4f},{MRR:.4f}")
    if args.wandb_log: 
        wandb.log({"Test MAP": MAP, "Test MRR": MRR})
    print(f"Final test metrics: MAP,MRR: {MAP:.4f},{MRR:.4f}")

    if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator, final=True)

    if not args.train_with_orig:
        cmd = f"python run_inference.py --dataset {args.dataset} --expt_id {experiment_id} --device {DEVICE[-1]}"
        ret = os.system(cmd)
        if ret != 0:
            print(f"Error in running inference script")

    logging.info("*"*120+"\n"+"*"*120)