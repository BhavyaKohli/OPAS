import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import os
import h5py
import time
import argparse

from tqdm import tqdm
import sys
sys.path.append("../../")
from opas.models.cifar_embed import Autoencoder
from opas.models.lsun_embed import Autoencoder as LSUNAutoencoder


def sample(dataset, classes, num_samples):
    labels = np.array(dataset.targets)
    samps = []
    slabs = []
    for cls in classes:
        cls_locs = np.where(labels == cls)[0]
        cls_locs = np.random.choice(cls_locs, size=num_samples, replace=False)
        samps.append(np.stack([dataset[i][0].cpu().numpy() for i in cls_locs]))
        slabs.append(np.ones((len(samps[-1]),1)) * cls)
    return np.vstack(samps), np.vstack(slabs)


@torch.no_grad()
def embed(model, x):
    x = x.to(DEVICE)
    x = model.encoder(x).flatten(start_dim=-3)
    return x.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=["CIFAR", "LSUN"])
    parser.add_argument('--ae_weights', type=str, help="Path to autoencoder weights, stored in ./embedding_models by default if using `train_ae.py` script")
    parser.add_argument('--device', type=int, default=-1, help="Cuda device index, pass -1 to run on cpu (not recommended)")
    parser.add_argument('--seq_len', type=int, default=20, help="Length of sequence (default 20)")
    args = parser.parse_args()

    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu"
    LSUN_ROOT = "lsun_gt/"
    CIFAR_ROOT = "cifar_gt/"

    embed_model_ckpt = torch.load(args.ae_weights)
    if args.dataset_name == "CIFAR":
        embed_model = Autoencoder()
    elif args.dataset_name == "LSUN":
        embed_model = LSUNAutoencoder()

    embed_model.load_state_dict(embed_model_ckpt)
    embed_model = embed_model.to(DEVICE)
    embed_model.eval()

    if args.dataset_name == "CIFAR":
        transform_eval = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=True, download=False, transform=transform_eval)
        testset = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=False, download=False, transform=transform_eval)

    elif args.dataset_name == "LSUN":
        transform_eval = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.4846, 0.5057, 0.5165), (0.2738, 0.2705, 0.3113)),
        ])

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(LSUN_ROOT, "train"), transform=transform_eval)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(LSUN_ROOT, "test"), transform=transform_eval)

    gen = torch.Generator().manual_seed(69)
    trainset, valset = torch.utils.data.random_split(trainset, [0.8, 0.2], generator=gen)
    trainset.targets = torch.tensor(trainset.dataset.targets)[trainset.indices]
    valset.targets = torch.tensor(valset.dataset.targets)[valset.indices]

    np.random.seed(15)
    if args.dataset_name == "CIFAR":
        classes = [0, 1, 2, 3, 4, 5]
        num_samples = 20
    elif args.dataset_name == "LSUN":
        classes = [0]
        num_samples = 200

    train_samples = sample(trainset, classes, num_samples)
    val_samples = sample(valset, classes, num_samples//2)
    test_samples = sample(testset, classes, num_samples//2)

    dataset_save_path = f"rotated_sequences_{args.dataset_name}.hdf5"
    if os.path.exists(dataset_save_path):
        print("Dataset exists at path, removing in 2 seconds..")
        time.sleep(2)
        os.remove(dataset_save_path)

    maxlim = 180 if args.seq_len <= 20 else 270
    ANGLES = np.linspace(0, maxlim, args.seq_len, endpoint=True)
    for split, dset in zip(["train", "val", "test"], [train_samples, val_samples, test_samples]):
        data, _ = dset
        data = torch.from_numpy(data)
        dset_orig = []
        dset_embeds = []
        dset_labels = []
        pos = 0
        for i in tqdm(range(len(data))):
            images = []
            labels = []
            for _ in range(10):
                images_ = []
                for angle in ANGLES:
                    angle = angle + 20 * np.random.rand()
                    images_.append(TF.rotate(data[i], angle))
                images.append(torch.stack(images_))
                labels.append(torch.tensor([pos+k for k in range(10)]))
            pos += len(labels[0])

            dset_orig.append(torch.stack(images))
            images = torch.stack([embed(embed_model, sequence) for sequence in images])
            labels = torch.stack(labels)

            dset_embeds.append(images)
            dset_labels.append(labels)

        dset_orig = torch.vstack(dset_orig).float()
        dset_embeds = torch.vstack(dset_embeds).float()
        dset_labels = torch.vstack(dset_labels).long()

        with h5py.File(dataset_save_path, "a") as f:
            f.create_dataset(f"{split}/orig", data=dset_orig)
            f.create_dataset(f"{split}/data", data=dset_embeds)
            f.create_dataset(f"{split}/labels", data=dset_labels)
            print(f"Saved {split} with {len(dset_embeds)} samples")