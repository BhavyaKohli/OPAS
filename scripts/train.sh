#!/bin/bash

dataset=$1
device=$2

if [ $dataset = "audio" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 128 --batch_size 400 --xlr 5e-6 --xff 512 --delta 0.3 --dataset audio --stagger 0 --print_dataset
elif [ $dataset = "human" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 128 --batch_size 400 --xlr 5e-6 --xff 512 --delta 0.3 --dataset human --stagger 0 --print_dataset
elif [ $dataset = "cifar" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 32 --batch_size 800 --xlr 5e-5 --xff 256 --delta 0.7 --num_q 800 --dataset cifar --print_dataset --train_with_orig
elif [ $dataset = "lsun384" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 64 --batch_size 400 --xlr 5e-5 --lr 5e-4 --xff 256 --delta 0.3 --num_q 800 --dataset lsun384 --print_dataset
else
    echo "No pre-existing training script for this dataset, please refer to the README for instructions on how to train on a new dataset."
fi