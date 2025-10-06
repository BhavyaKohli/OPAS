#!/bin/bash

device=$1

ln -s final_data/cifar final_data/cifar-large       # name compatibility for some notebooks which expect cifar-large, in case the command has not been run already

python run_inference.py --device $1 --expt_id A01091914 --dataset audio
python run_inference.py --device $1 --expt_id S28021253 --dataset speech
python run_inference.py --device $1 --expt_id C07040027 --dataset cifar