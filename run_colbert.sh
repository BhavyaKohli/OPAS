#!/bin/bash

export TQDM_DISABLE=1

DATASET_ARRAY=("audio" "speech" "cifar")
EXPT_ID_ARRAY=("2411194526" "2411225240" "2411225244")
HEADS=(10 10 8)
STEPS=(2 2 1)

for i in "${!DATASET_ARRAY[@]}"; do
    DATASET=${DATASET_ARRAY[$i]}
    EXPT_ID=${EXPT_ID_ARRAY[$i]}
    HEAD=${HEADS[$i]}
    STEP=${STEPS[$i]}
    
    CUDA_VISIBLE_DEVICES=1 python main_colbert_for_odr.py dataset=$DATASET device=0 print_dataset=False xnhead=$HEAD samp_rate_step=$STEP xnumlayers=2 expt_id=$EXPT_ID n_samp_odc=50 &
    CUDA_VISIBLE_DEVICES=2 python main_colbert_for_odr.py dataset=$DATASET device=0 print_dataset=False xnhead=$HEAD samp_rate_step=$STEP xnumlayers=2 expt_id=$EXPT_ID n_samp_odc=100&
done
