#!/bin/bash

dataset=$1
device=$2
extra=${@:3}

if [ "$dataset" = "a" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 128 --batch_size 400 --xlr 5e-6 --xff 512 --delta 0.3 --dataset audio --stagger 0 --print_dataset $extra
elif [ "$dataset" = "s" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 128 --batch_size 400 --xlr 5e-6 --xff 512 --delta 0.3 --dataset speech --stagger 0 --print_dataset $extra
elif [ "$dataset" = "c" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 32 --batch_size 800 --xlr 5e-5 --xff 256 --delta 0.7 --num_q 800 --dataset cifar --print_dataset --train_with_orig $extra
elif [ "$dataset" = "l" ]; then
    python main.py --nepochs 30 --preembed tokenize --device $device --xoutdim 64 --batch_size 400 --xlr 5e-5 --lr 5e-4 --xff 256 --delta 0.3 --num_q 800 --dataset lsun --print_dataset $extra
else
    echo "Invalid dataset option. Please choose from 'a', 's', 'c', or 'l'."
fi