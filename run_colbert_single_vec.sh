#!/bin/bash

export TQDM_DISABLE=1

DATASET_ARRAY=("speech" "cifar")
XNL=(2 4)
HEADS=(10 8)
STEPS=(2 1)
let g=0

for i in "${!DATASET_ARRAY[@]}"; do
    for layers in "${XNL[@]}"; do
        DATASET=${DATASET_ARRAY[$i]}
        GPU="$(($g % 8))"
        CUDA_VISIBLE_DEVICES=$GPU python main_colbert_single_vec.py dataset=$DATASET device=0 print_dataset=False note="layers=$layers" tqdm_disable=True xnhead=${HEADS[$i]} samp_rate_step=${STEPS[$i]} xnumlayers=$layers &
        let g+=1
        sleep 1
        GPU="$(($g % 8))"
        CUDA_VISIBLE_DEVICES=$GPU python main_colbert.py dataset=$DATASET device=0 print_dataset=False note="layers=$layers" tqdm_disable=True xnhead=${HEADS[$i]} samp_rate_step=${STEPS[$i]} xnumlayers=$layers &
        sleep 1
        let g+=1
    done
done