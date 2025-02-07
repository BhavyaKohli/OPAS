import wandb
import numpy as np
import h5py
import logging
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from tqdm.auto import tqdm

from opas.tstok.tokenizer import Tokenizer
from opas.utils import AttributeDict, gumbel_sinkhorn, normalize, get_opas_constants
from opas.data import PairDatasetTrain, PairDatasetTest

from opas.models.main import LamModel, ScoreModel, PositionalEncoding
from opas.models.cifar_embed import Autoencoder
from opas.models.lsun_embed import Autoencoder as LSUNAutoencoder
from opas.models.ts_encoders import Conv1dTS, EncConv1dTS, MelConv1dTS
from opas.models.deepset import DeepSetModel

from transformers import get_linear_schedule_with_warmup, AdamW

from functools import partial

import torchaudio.transforms as T


def get_tokenizer():
    data_config = AttributeDict({
        "max_seq_len": 160,
        "batch_size": 64,
        "bin_size": 0.005,
        "max_coverage": .9998,
        "vocab_size": 512
    })
    tokenizer = Tokenizer(data_config)
    
    return tokenizer

TOKENIZER = get_tokenizer()


def tokenize(x, args):
    if args.no_tokenize:
        return x, None
        
    orig_shape = x.shape
    device = x.device
    x = x.squeeze().cpu()
    if len(x.shape) == 3:
        x = x.reshape(-1, orig_shape[-1])   
    ids, p = TOKENIZER.encode(x)
    ids = torch.from_numpy(ids).long()
    ids = ids.reshape(orig_shape)
    return ids.to(device), p


def get_white_noise(signal, SNR):
    RMS_s = torch.sqrt(torch.mean(signal**2))
    RMS_n = torch.sqrt(RMS_s**2 / (pow(10, SNR/10)))
    STD_n = RMS_n
    noise = torch.distributions.Normal(0, STD_n).sample(signal.shape)
    return noise


def batch_get_white_noise(x, SNR):
    # x is B x m/n x signal
    return get_white_noise(x, SNR)


def stagger_and_concat(model_inputs, num_stagger=1):
    # input of shape bmn OR bNmn
    if len(model_inputs.shape) == 3:
        model_inputs = model_inputs.unsqueeze(1)

    model_inputs_staggered = [model_inputs]
    for i in range(1,num_stagger+1,1):
        model_inputs_staggered.append(torch.dstack((model_inputs[:,:,i:,:], torch.zeros_like(model_inputs)[:,:,:i,:])))
    if num_stagger==0 : return torch.stack(model_inputs_staggered, dim=2)
    return torch.stack(model_inputs_staggered, dim=2).squeeze() 


class Attention_Layer(nn.Module):
    def __init__(self, n_feats: int) -> None:
        super().__init__()
        self.w = nn.Linear(
            in_features=n_feats,
            out_features=n_feats
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        w = self.w(X)
        output = F.softmax(torch.mul(X, w), dim=1)
        return output
    

class TransformInput(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform(x)


@torch.no_grad()
def embed_image(model, x):
    x = x.to(next(model.parameters()).device)
    out = model.encoder(x) 
    out = out.flatten(start_dim=-3)
    return out


@torch.no_grad()
def embed_if_image_and_normalize(c, image_embed_model=None):
    if len(c.shape) != 3:   
        # b x n x c x h x w instead of b x n x d
        if image_embed_model is None:
            raise ValueError("Image embed model not provided")
        c = torch.stack([embed_image(image_embed_model, c_) for c_ in c])
    return normalize(c)


def embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model=None, inner_batch_size=800, aggregator=None):
    C = torch.from_numpy(dataset.c).float()
    Cembed = []
    for batch in tqdm(range(0, len(C), inner_batch_size), disable=True):
        c = C[batch:batch+inner_batch_size].to(next(embed_model.parameters()).device)
        c = embed_if_image_and_normalize(c, image_embed_model)
        c = embed_model(preembed_model(c))
        c = normalize(c)
        if aggregator is not None:
            c = aggregator(c)
        Cembed.append(c)
    C = torch.vstack(Cembed)
    return C


@torch.no_grad()
def compute_metrics(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False, aggregator=None):

    loader = dataset.get_dataloader(batch_size=200, shuffle=True)

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
                qct = torch.einsum("bmd,Nnd->bNmn", q, C)
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
            else:
                F_mat = Rm_mat.T @ (
                    torch.stack([-(q[i][None].unsqueeze(2) - C.unsqueeze(1)).relu().sum(-1) for i in range(len(q))])
                )
                lambdas = torch.ones((len(q), len(C), M, 1), device=F_mat.device)

            P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)

            RmPC = Rm_mat @ P @ C.squeeze(-1)

            lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
            normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

            allscores = torch.stack([lamscore, normscore], dim=2)
            netscore = 2*scoremodel(-allscores).squeeze()      # b

        else:
            q = aggregator(q)
            # q is bd, C is Nd, we want bN scores
            # b1d - 1Nd = bNd --> sum across last dim to get bN scores
            netscore = 2 * F.sigmoid(-F.relu(q.unsqueeze(1) - C.unsqueeze(0)).sum(dim=-1))    # bN

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

            qct = torch.einsum("bmd,bnd->bmn", q, c)
            model_inputs = stagger_and_concat(qct, num_stagger=stagger)
            
            lambdas = model(model_inputs)

            F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(1,2) @ A_mat).transpose(1,2))

            P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)
            RmPC = torch.einsum("mn,bnn,bnd->bmd", Rm_mat, P, c)

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

    mRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    mAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    mAP /= (torch.arange(ranked_output.shape[1]) + 1)
    mAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    mAP = mAP.sum(dim=1).mean().item()

    return mAP, mRR


@torch.no_grad()
def compute_odr(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False):

    loader = dataset.get_dataloader(batch_size=1, shuffle=True)

    num_samples = 20
    C = embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model=image_embed_model)
    # C is (N, n, xoutdim)

    odc = []
    for n, (q, l) in enumerate(tqdm(loader, leave=True, disable=not verbose)):
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

        qct = torch.einsum("bmd,Nnd->bNmn", q, true_c)
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
            os.remove(f"models/{experiment_id}/model_latest.pt")
            os.remove(f"models/{experiment_id}/scmodel_latest.pt")
            os.remove(f"models/{experiment_id}/embed_model_latest.pt")
            if args.preembed != "tokenize":
                os.remove(f"models/{experiment_id}/preembed_model_latest.pt")
            if aggregator is not None:
                os.remove(f"models/{experiment_id}/aggregator_latest.pt")
        except:
            print("Couldn't delete `latest` models")
    elif latest:
        suffix = "_latest"
    else:
        suffix = ""
    
    logging.info(f"Saving models... experiment id: {experiment_id}")
    torch.save(model, f"models/{experiment_id}/model{suffix}.pt")
    torch.save(scoremodel, f"models/{experiment_id}/scmodel{suffix}.pt")
    torch.save(embed_model, f"models/{experiment_id}/embed_model{suffix}.pt")
    if args.preembed != "tokenize":
        torch.save(preembed_model, f"models/{experiment_id}/preembed_model{suffix}.pt")
    if aggregator is not None:
        torch.save(aggregator, f"models/{experiment_id}/aggregator{suffix}.pt")


def load_models(name="best", expt_id=None, device="cpu"):
    if expt_id is None:
        expt_id = experiment_id

    if name not in ["best", "last", "latest"]:
        raise NotImplementedError("Only `best`, `last`, and `latest` models are supported")
    
    if name=="best":
        suffix = ""
    else:
        suffix = f"_{name}"
    
    print(f"Loading `{name}` model")

    model = torch.load(f"models/{expt_id}/model{suffix}.pt", map_location=device)
    scoremodel = torch.load(f"models/{expt_id}/scmodel{suffix}.pt", map_location=device)
    embed_model = torch.load(f"models/{expt_id}/embed_model{suffix}.pt", map_location=device)
    if os.path.exists(f"models/{expt_id}/preembed_model{suffix}.pt"):
        preembed_model = torch.load(f"models/{expt_id}/preembed_model{suffix}.pt", map_location=device)
    else:
        tokenize_transform = lambda x: tokenize(x, args)[0]
        preembed_model = TransformInput(tokenize_transform).to(device)
    
    if os.path.exists(f"models/{expt_id}/aggregator{suffix}.pt"):
        aggregator = torch.load(f"models/{expt_id}/aggregator{suffix}.pt", map_location=device)
    else:
        aggregator = None

    return model, scoremodel, embed_model, preembed_model, aggregator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="name of dataset, stored in `final_data/`")
    parser.add_argument("--print_dataset", action="store_true", help="prints dataset array shapes when passed")
    parser.add_argument("--num_q", type=int, default=300, help="number of queries to be sampled from the dataset")
    parser.add_argument("--wandb_log", action="store_true", help="whether to log to wandb")

    parser.add_argument("--b", type=float, default=1.0, help="hinge margin for negative gap penalty (b-Apa)")
    parser.add_argument("--b1", type=float, default=0.0, help="hinge margin for positive gap penalty (Apa-b)")
    parser.add_argument("--delta", type=float, default=0.7, help="hinge margin for contrastive loss")
    parser.add_argument("--stagger", type=int, default=0, help="number of staggers when creating model input")
    parser.add_argument("--lamwt", type=float, default=1e-2, help="loss coefficient for negative gap penalty")
    parser.add_argument("--gapwt", type=float, default=0.0, help="loss coefficient for positive gap penalty")
    parser.add_argument("--SNR", type=int, default=1, help="input white noise SNR")
    parser.add_argument("--noise", type=float, default=0, help="input white noise")
    parser.add_argument("--preembed", type=str, help="pre-embed model to use (tokenize, linear, conv, none)")

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--nepochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use (default: 0)")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size for training")

    parser.add_argument("--xnhead", type=int, default=4, help="nhead for transformer encoder layer")
    parser.add_argument("--xnumlayers", type=int, default=2, help="num layers for transformer encoder")
    parser.add_argument("--xoutdim", type=int, default=128, help="out dim for transformer encoder model (4000 -> outdim)")
    parser.add_argument("--xlr", type=float, default=1e-5, help="learning rate for transformer encoder model")
    parser.add_argument("--xwarmupepochs", type=int, default=20, help="warmup epochs for transformer encoder model")
    parser.add_argument("--xff", type=int, default=2048, help="feedforward dim of transformer encoder model")

    parser.add_argument("--use_sing_xfmer", action="store_true", help="pass when a single transformer is to be used, for both q and c")
    parser.add_argument("--no_tokenize", action="store_true", help="pass when transformer pre-inputs should NOT be tokenized")
    parser.add_argument("--enforce_order", action="store_true", help="pass when enforce order during training")
    parser.add_argument("--skip_embed", action="store_true", help="pass to skip embed_model completely (embed_model will be replaced with nn.Identity())")
    parser.add_argument("--train_with_orig", action="store_true", help="when using the cifar or lsun datasets, pass this when training should be done using the original images and not the embeddings directly (not recommended for LSUN)")
    parser.add_argument("--deepset", action="store_true", help="pass when query and corpus sequences are to be embedded into the same latent space, for using a (hq-hc)+ loss instead of the usual sinkhorn->permutation->normscore+lamscore components")
    parser.add_argument("--deepset_mode", type=int, default=2, help="mode 2 is normalized, mode 1 is not (only used when deepset is passed)")
    parser.add_argument("--add_attention", action="store_true", help="[unused in current script version] pass when an additional attention module is required before the Xfmer")
    parser.add_argument("--lfcc", action="store_true", help="[unused in current script version] pass when LFCC features should be used in place of audio")

    # sinkhorn params
    parser.add_argument("--n_sink_iter", type=int, default=20, help="number of sinkhorn iterations")
    parser.add_argument("--single_step_norm", type=int, default=-1, choices=[-1,1,2], help="controls how P is obtained from F during training. -1 means no single step normalization (sinkhorn iterations), 1 means single step normalization using sum, 2 means single step normalization using exp (attn)")

    # post-review
    parser.add_argument("--internal_lamwt", type=float, default=1, help="coefficient for A^T @ lambda @ a^T inside F_mat")
    parser.add_argument("--use_linear_lammodel", action="store_true", help="pass when a linear model should be used for lambda instead of a transformer")
    parser.add_argument("--debug", action="store_true", help="no logging to file")

    # new experiments
    parser.add_argument("--no_lammodel", action="store_true", help="pass when sinkhorn matrix is to be computed without lammodel, using -relu(hq-hc)")

    args = parser.parse_args()

    experiment_id = datetime.now().strftime("%d%m%H%M")
    args.human = args.video = args.cifar = args.lsun = False
    image_embed_model = None
    if "audio" in args.dataset:
        experiment_id = f"A{experiment_id}" 
    elif "human" in args.dataset or "speech" in args.dataset:
        args.human = True
        experiment_id = f"AH{experiment_id}"
    elif "video" in args.dataset:
        args.video = True
        experiment_id = f"V{experiment_id}"
    elif "cifar" in args.dataset:
        args.cifar = True
        experiment_id = f"C{experiment_id}" 
        image_embed_model_ckpt = "data/image_sequence/embedding_models/cifar_ae.pkl"
        image_embed_model = Autoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

    elif "lsun" in args.dataset:
        args.lsun = True
        experiment_id = f"L{experiment_id}"

        image_embed_model_ckpt = "data/image_sequence/embedding_models/lsun_ae.pkl"
        image_embed_model = LSUNAutoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

    else:
        raise NotImplementedError(f"Check dataset name")

    if not args.debug:
        os.makedirs(f"models/{experiment_id}/", exist_ok=True)
        print(f"Experiment information at models/{experiment_id}")

        logging.basicConfig(
            filename=f"models/{experiment_id}/experiments.log",
            filemode="a+",
            level=logging.INFO,
            format="%(levelname)s (%(asctime)s): %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p"
        )
    
    logging.info("python " + " ".join(sys.argv))
    logging.info(f"Running with args: {args}")
    print(f"Running with args: {args}")

    NOLAMMODEL = args.no_lammodel

    DEVICE = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
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
        with open(f"models/{experiment_id}/args.pkl", "wb") as f:
            pickle.dump(args, f)

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

    neg_expl = 300 if getattr(args, "video", None) else 800

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

    if args.wandb_log : 
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

    scoremodel = ScoreModel().to(DEVICE)
    
    tokenize_transform = lambda x: tokenize(x, args)[0]
    d_model = 256
    lin_transform = nn.Sequential(
        nn.Linear(samp_rate, d_model),
        nn.Tanh(),
        nn.Linear(d_model, d_model),
    )
    identity = lambda x: x
    conv_transform = Conv1dTS(n_ch=4, latent=d_model)
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
        lin_transform = nn.Sequential(
            nn.Linear(samp_rate, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        args.no_tokenize = True     # turn off input tokenization
        with open(f"models/{experiment_id}/args.pkl", "wb") as f:
            pickle.dump(args, f)

        embed_model = TransformInput(lin_transform)
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
        aggregator = DeepSetModel(indim=args.xoutdim, latent=args.xoutdim//2, outdim=args.xoutdim//2).to(DEVICE)
        aggregator_optimizer = torch.optim.Adam(aggregator.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        aggregator = None

    model = LamModel(M, N, stagger).to(DEVICE)
    if args.use_linear_lammodel:
        model = nn.Sequential(nn.Linear((M+N)*args.xoutdim, M), nn.Sigmoid()).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    sc_optimizer = torch.optim.Adam(scoremodel.parameters(), lr=lr, amsgrad=True, weight_decay=1e-2)

    tf = T.LFCC(sample_rate=4000, n_lfcc=40, log_lf=False, speckwargs={"n_fft": 1024}).to(DEVICE)
    
    ################ NOT USED ######################
    if args.lfcc: 
        TRANSFORM = lambda x: tf(x).flatten(start_dim=-2)
    else:
        TRANSFORM = lambda x: x
    ################ NOT USED ######################

    losslog = []

    logging.info("Training\n"+"*"*120+"\n"+"*"*120)

    pbar = tqdm(range(1,nepochs+1,1), disable=False)
    best_val_map, best_val_mrr = 0, 0

    enforce_order = args.enforce_order

    for i in pbar:
        if i == 1:
            if not args.use_sing_xfmer:
                # sanity check on compute metrics
                _, _ = compute_metrics(val_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)
        
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

            q = q + batch_get_white_noise(q, args.SNR)
            q, c = embed_if_image_and_normalize(q, image_embed_model), embed_if_image_and_normalize(c, image_embed_model)

            if args.use_sing_xfmer:
                qc = torch.cat((q, c), dim=1)
                qc = embed_model(preembed_model(qc))
                q, c = qc[:, :M], qc[:, M:]
                q, c = normalize(q, c)
            else:
                # main forward pass, !! Change this in compute_metrics as well !!
                q, c = embed_model(preembed_model(q)), embed_model(preembed_model(c))
                q, c = normalize(q, c)

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

                RmPC = torch.einsum("mn,bnn,bnd->bmd", Rm_mat, P, c)

                lamscore = lamwt * (lambdas.transpose(1,2) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
                normscore = torch.norm(q - RmPC, dim=[1,2])

                allscores = torch.stack([lamscore, normscore], dim=1)
                netscore = 2*scoremodel(-allscores).squeeze()
            elif DEEPSET:
                q, c = aggregator(q), aggregator(c)
                if args.deepset_mode == 2:
                    netscore = 2 * F.sigmoid(-F.relu(q - c).sum(dim=-1))    # normalized to 0-1, 0 for worst, 1 for best (==0 loss)
                elif args.deepset_mode == 1:
                    netscore = -F.relu(q - c).sum(dim=-1)                 # un-normalized scores

            pos_score = torch.atleast_1d(netscore[torch.where(l==1)])
            neg_score = netscore[torch.where(l==0)]
            
            neg_minus_pos = (neg_score.unsqueeze(0) - pos_score.unsqueeze(1)).reshape(-1)   # dim(pos_score) * dim(neg_score)

            loss = F.relu(delta + neg_minus_pos).mean() 
            if gapwt > 0 and not DEEPSET:
                loss += gapwt * F.relu(A_mat @ Rm_mat @ P @ a_vec - b1).mean()

            optimizer.zero_grad()
            sc_optimizer.zero_grad()
            embed_optimizer.zero_grad()
            if args.preembed != "tokenize": 
                preembed_optimizer.zero_grad()
            if DEEPSET:
                aggregator_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sc_optimizer.step()
            embed_optimizer.step()
            if args.preembed != "tokenize": 
                preembed_optimizer.step()
            if DEEPSET:
                aggregator_optimizer.step()
            scheduler.step()

            pbar.set_postfix_str(f"loss: {loss:.4f}")
            wandb_losslog.append(loss.item())
            if args.wandb_log:
                wandb.log({"loss": loss.item()})


        # validation metrics and logging
        model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
        if DEEPSET:
            aggregator.eval()
        if not args.use_sing_xfmer:
            mAP, mRR = compute_metrics(val_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=False, aggregator=aggregator)

            if mAP > best_val_map: 
                best_val_map = mAP
                val_mrr_at_best = mRR
                if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator)

            if enforce_order:
                odr = compute_odr(val_dataset, model, scoremodel, embed_model, preembed_model, stagger=stagger, verbose=False)
                logging.info(f"ODR at epoch: {i:2d} = {odr:.2f}")
                if args.wandb_log:
                    wandb.log({"ODR": odr})
        else:
            mAP, mRR = 0, 0
            val_mrr_at_best = 0
            if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator)
            
        logging.info(f"[Epoch {i:2d}|{nepochs}] loss: {np.mean(wandb_losslog):.4f}, val mAP: {mAP:.4f}, val mRR: {mRR:.4f}, best mAP: {best_val_map:.4f}, mRR @ best val mAP: {val_mrr_at_best:.4f}")
    
        if args.wandb_log:
            wandb.log({"val MAP": mAP, "val MRR": mRR})
            wandb.log({"best val MAP": best_val_map, "best val MRR": val_mrr_at_best})
            wandb_losslog = []

        if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator, latest=True)

    model, scoremodel, embed_model, preembed_model, aggregator = load_models(name="best", expt_id=experiment_id, device=DEVICE)
    model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
    if DEEPSET:
        aggregator.eval()

    if args.use_sing_xfmer:
        mAP, mRR = compute_metrics_sing(test_dataset, model, scoremodel, embed_model, preembed_model, stagger=stagger, verbose=True)
    else:
        mAP, mRR = compute_metrics(test_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)
    
    logging.info(f"Final test metrics: mAP: {mAP:.4f}, mRR: {mRR:.4f}")
    if args.wandb_log: 
        wandb.log({"Test MAP": mAP, "Test MRR": mRR})
    print(f"Final test metrics: mAP: {mAP:.4f}, mRR: {mRR:.4f}")

    if not args.debug: save_models(model, scoremodel, embed_model, preembed_model, aggregator, final=True)
    logging.info("*"*120+"\n"+"*"*120)