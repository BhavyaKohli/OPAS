#!/bin/bash

DATASET_ARRAY=("audio" "speech" "cifar")
EXPT_ID_ARRAY=("A01091914" "S28021253" "C07040027")

for i in "${!DATASET_ARRAY[@]}"; do
    DATASET=${DATASET_ARRAY[$i]}
    EXPT_ID=${EXPT_ID_ARRAY[$i]}
    python run_inference_odc.py --dataset=$DATASET --device=3 --expt_id=$EXPT_ID --n_samp=20 &
    python run_inference_odc.py --dataset=$DATASET --device=4 --expt_id=$EXPT_ID --n_samp=50 &
    python run_inference_odc.py --dataset=$DATASET --device=5 --expt_id=$EXPT_ID --n_samp=100 &
done
