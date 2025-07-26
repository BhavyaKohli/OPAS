import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from opas.randomaug import RandAugment
from opas.models.cifar_embed import Autoencoder
from opas.models.lsun_embed import Autoencoder as LSUNAutoencoder


SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=["CIFAR", "LSUN", "IMAGENET"])
    parser.add_argument('--device', type=int, default=-1, help="CUDA device index, pass -1 to run on cpu (not recommended)")
    args = parser.parse_args()

    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu"
    LSUN_ROOT = "data/image_sequence/lsun_gt/"
    CIFAR_ROOT = "data/image_sequence/cifar_gt/"
    IMAGENET_ROOT = "data/image_sequence/imagenet_gt/"

    # Create model
    if args.dataset_name == "CIFAR":
        autoencoder = Autoencoder()

        size = (32, 32)
        # Load data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Add RandAugment with N, M(hyperparameter)
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))


        trainset = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                                shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=False,
                                            download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                                shuffle=False, num_workers=8)

    elif args.dataset_name == "LSUN":
        autoencoder = LSUNAutoencoder()

        size = (256, 256)
        # Load data
        transform_train = transforms.Compose([
            transforms.RandomCrop(198, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize((0.4846, 0.5057, 0.5165), (0.2738, 0.2705, 0.3113)),
        ])

        # Add RandAugment with N, M(hyperparameter)
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(LSUN_ROOT, "train"), transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                                shuffle=True, num_workers=8)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(LSUN_ROOT, "test"), transform=transform_train)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                                shuffle=False, num_workers=8)
        
    elif args.dataset_name == "IMAGENET":
        ImageNetAutoencoder = LSUNAutoencoder
        autoencoder = ImageNetAutoencoder()

        size = (256, 256)
        # Load data
        transform_train = transforms.Compose([
            transforms.RandomCrop(198, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(270),
            transforms.ToTensor(),
            transforms.Normalize([0.4735, 0.4494, 0.4040], [0.2741, 0.2659, 0.2775]),
        ])

        # Add RandAugment with N, M(hyperparameter)
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(IMAGENET_ROOT, "train"), transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                                shuffle=True, num_workers=8)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(IMAGENET_ROOT, "test"), transform=transform_train)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                                shuffle=False, num_workers=8)

    # Define an optimizer and criterion
    autoencoder.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5)
    
    pbar = tqdm(range(300))
    for epoch in pbar:
        running_loss = 0.0
        autoencoder.train()
        for i, (inputs, _) in enumerate(trainloader, 0):
            inputs = inputs.to(DEVICE)
            inputs_noise = inputs + 0.5 * torch.randn_like(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs_noise)
            if epoch == 0 and i == 0:
                assert outputs.shape == inputs.shape, "recon != input shape"
            loss = criterion(outputs, inputs)

            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
        
        trainloss = running_loss / len(trainloader)
        running_loss = 0.0

        # ============ Validation ============
        autoencoder.eval()
        valid_loss = 0.0
        for i, (inputs, _) in enumerate(testloader, 0):
            inputs = inputs.to(DEVICE)
            inputs_noise = inputs + 0.5 * torch.randn_like(inputs)
            encoded, outputs = autoencoder(inputs_noise)
            loss = criterion(outputs, inputs)
            valid_loss += loss.data
        scheduler.step(valid_loss)

        testloss = valid_loss / len(testloader)

        pbar.set_postfix_str(f"{trainloss:.4f}, {testloss:.4f}")

    print('Finished Training')
    print('Saving Model...')
    os.makedirs("data/image_sequence/embedding_models/", exist_ok=True)
    torch.save(autoencoder.state_dict(), f"data/image_sequence/embedding_models/{args.dataset_name}_ae.pkl")