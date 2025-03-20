import os
import torch

from .main import *
from .cifar_embed import Autoencoder
from .lsun_embed import Autoencoder as LSUNAutoencoder
from ..tstok.tsutils import TOKENIZER, tokenize


def load_models(name="best", expt_root=None, device="cpu", args=None):
    if expt_root is None:
        raise ValueError("Please provide a valid experiment root folder in the inference script")

    if name not in ["best", "last", "latest"]:
        raise NotImplementedError("Only `best`, `last`, and `latest` models are supported")
    
    if name=="best":
        suffix = ""
    else:
        suffix = f"_{name}"
    
    print(f"Loading `{name}` model")

    model = torch.load(f"{expt_root}/model{suffix}.pt", map_location=device, weights_only=False)
    scoremodel = torch.load(f"{expt_root}/scmodel{suffix}.pt", map_location=device, weights_only=False)
    embed_model = torch.load(f"{expt_root}/embed_model{suffix}.pt", map_location=device, weights_only=False)
    if os.path.exists(f"{expt_root}/preembed_model{suffix}.pt"):
        preembed_model = torch.load(f"{expt_root}/preembed_model{suffix}.pt", map_location=device, weights_only=False)
    else:
        tokenize_transform = lambda x: tokenize(x, args)[0]
        preembed_model = TransformInput(tokenize_transform).to(device)
    
    if os.path.exists(f"{expt_root}/aggregator{suffix}.pt"):
        aggregator = torch.load(f"{expt_root}/aggregator{suffix}.pt", map_location=device, weights_only=False)
    else:
        aggregator = None

    return model, scoremodel, embed_model, preembed_model, aggregator


def get_image_embed_model(dataset, dataset_file_paths, device):
    """
    dataset: str
    dataset_file_paths: list of file paths (TRAIN_FILE, VAL_FILE, TEST_FILE)
    """

    image_embed_model = None
    name_replace = lambda x: x.replace(".hdf5", "_orig.hdf5")
    
    if "cifar" in dataset:
        image_embed_model_ckpt = "data/image_sequence/embedding_models/cifar_ae.pkl"
        image_embed_model = Autoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt, map_location='cpu'))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

        image_embed_model = image_embed_model.to(device)
        dataset_file_paths = [name_replace(file_path) for file_path in dataset_file_paths]
    
    elif "lsun" in dataset:
        image_embed_model_ckpt = "data/image_sequence/embedding_models/lsun_ae.pkl"
        image_embed_model = LSUNAutoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt, map_location='cpu'))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

        image_embed_model = image_embed_model.to(device)
        dataset_file_paths = [name_replace(file_path) for file_path in dataset_file_paths]

    return image_embed_model, *dataset_file_paths 