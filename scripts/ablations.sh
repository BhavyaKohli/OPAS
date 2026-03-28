#!/bin/bash

# This script is meant for reference, and is not meant to be run directly

for dataset in "audio" "speech" "cifar-large" "lsun"; do
    python main.py dataset=$dataset device=$device fix_lambdas=1                            # fixed lambda_j=1 for all j
    python main.py dataset=$dataset device=$device lamwt=0.                                 # disable s_lambda
    python main.py dataset=$dataset device=$device no_lammodel=True                         # pairwise hinge score instead of LamModel
    
    python main.py dataset=$dataset device=$device skip_embed=True skip_type=lrl            # ephi lrl
    python main.py dataset=$dataset device=$device skip_embed=True skip_type=conv           # ephi conv
    python main.py dataset=$dataset device=$device pretrain_embedding=True                  # pretrain ephi

    python main.py dataset=$dataset device=$device no_lamrelu=False                         # relu in s_lambda
    python main.py dataset=$dataset device=$device single_step_norm=1                       # SS_sum
    python main.py dataset=$dataset device=$device single_step_norm=2                       # SS_exp

    python main.py dataset=$dataset device=$device use_linear_lammodel=True                 # simple LamModel
    python main.py dataset=$dataset device=$device deepset=True deepset_mode="base"         # deepset instead of LamModel, unnormalized hinge
    python main.py dataset=$dataset device=$device deepset=True deepset_mode="normalized"   # deepset instead of LamModel, normalized hinge
    python main.py dataset=$dataset device=$device deepset=True deepset_mode="cosine"       # deepset instead of LamModel, cosine
    
    python main.py dataset=$dataset device=$device stagger=0                                # stagger=0
    python main.py dataset=$dataset device=$device stagger=2                                # stagger=2 

    python main.py dataset=$dataset device=$device dummy_scoremodel=True                    # remove trainable beta_1, beta_2 from scoremodel. score will be computed as sigmoid(s_F + s_\lambda)
    python main_direct_score.py dataset=$dataset device=$device                             # using e_\phi to output scores directly, without sinkhorn and OPAS scoring framework

    for niter in 20 15 10 5; do
        python main.py dataset=$dataset device=$device n_sink_iter=$niter                   # sinkhorn sensitivity
    done

    for neg_expl in 800 400 200; do
        python main.py dataset=$dataset device=$device neg_expl=$neg_expl                   # negative exploration
    done

    # python main.py dataset=long$dataset device=$device config=configs/long$dataset.yaml   # long datasets
done