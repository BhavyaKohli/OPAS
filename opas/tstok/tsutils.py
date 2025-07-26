import torch

from .tokenizer import Tokenizer
from ..utils import AttributeDict


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

def tokenize_og(x, args):
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


def tokenize(x, args):
    if args.no_tokenize:
        return x, None
    ids, p = TOKENIZER.encode_pt(x)
    return ids, p


def get_white_noise(signal, SNR):
    RMS_s = torch.sqrt(torch.mean(signal**2))
    RMS_n = torch.sqrt(RMS_s**2 / (pow(10, SNR/10)))
    STD_n = RMS_n
    noise = torch.distributions.Normal(0, STD_n).sample(signal.shape)
    return noise


def batch_get_white_noise(x, SNR):
    # x is B x m/n x signal
    return get_white_noise(x, SNR)