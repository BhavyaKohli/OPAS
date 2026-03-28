# OPAS

This is the official code for the AISTATS 2026 Paper: Learning Right Monotone Permutation Matrices for Neural Subsequence Search

## Main Note
The main branch of this repository contains the main code which has been stripped off of the effect of many script switches which enable/disable certain parameters. The dev branch contains the original versions of the code, which were used during the actual development. Although there is no difference in command usage, in case there are any unforseen issues, please raise an Issue and I will fix it immediately.

## Environment
Please refer to the paper for the server specs. The main requirements are a CUDA version compatible with CUDA 11.8 torch versions, and a valid conda installation Use `conda env create -f env.yaml` to create an environment named "opas" at the default conda env folder (no forced prefix), and activate it using `conda activate opas`. Non-pytorch requirements are provided in `requirements.txt`, in case the pip route is preferred. Torch will need to be installed separately in this case.

## Datasets
Download `final_data.zip` from [here](https://drive.google.com/drive/folders/1EVI_t2oObqU0uCeHaX1NrBUvg9Uvn3_H?usp=sharing) and unzip its contents into `final_data`. Due to size constraints, we provide only the processed test subset from the Music, Speech and CIFAR datasets. This will enable the execution of `run_inference.py` along with the timing and memory comparisons (on datasets other than LSUN) in `baselines/`.

For custom datasets, refer to `data/audio` for audio-based datasets, and `data/image_sequence` for image sequence datasets.
**The training scripts might require updating to accept new datasets, and to configure the prefix for the logging directories.**

The directory structure should be as follows:
```
.
├── baselines
│   ├── configs
│   │   ├── main.config
│   │   ├── main_rev.config
│   │   └── opas_mem.config
│   ├── naivedl
│   │   .
│   │   └── train.py
│   ├── tensors
│   │   ├── get_tensors.py
│   │   └── get_tensors_rev.py
│   ├── README.md
│   .
│   └── timing_comparisons.py
├── configs
│   ├── README.md
│   ├── audio.yaml
│   .
│   └── speech.yaml
├── data
│   ├── README.md
│   ├── audio
│   │   ├── music_raw
│   │   │   └── .
│   │   └── speech_raw
│   │   │   └── .
│   │   ├── generate_dataset_audio.py
│   │   .
│   │   └── process_and_save_data.py
│   ├── image_sequence
│   │   ├── embedding_models
│   │   │   ├── cifar_ae.pkl
│   │   │   └── lsun_ae.pkl
│   │   ├── get_rotated_sequences.py
│   │   ├── generate_dataset_image_sequence.py
│   │   └── nb_extend_cifar_dataset.ipynb
│   └── README.md
├── final_data
│   └── README.md
├── models
│   └── README.md
├── notebooks
│   .
│   └── nb_new_expts.ipynb
├── opas
│   ├── data.py
│   ├── metrics
│   │   ├── soft_dtw_cuda.py
│   │   └── soft_dtw.py
│   ├── models
│   │   ├── cifar_embed.py
│   │   ├── deepset.py
│   │   ├── lsun_embed.py
│   │   ├── main.py
│   │   ├── model_utils.py
│   │   ├── set_transformer.py
│   │   ├── sortlrl.py
│   │   └── ts_encoders.py
│   ├── randomaug.py
│   ├── tstok
│   │   ├── generic.py
│   │   ├── tokenizer.py
│   │   └── tsutils.py
│   └── utils.py
├── plots_and_figures
│   ├── data
│   │   .
│   │   └── times_speech.pkl
│   .
│   └── set_xfm_vs_deepset.pdf
├── scripts
│   .
│   └── ablations.sh
├── README.md
├── env.yaml
├── requirements.txt
├── train_ae.py                 # base script to train autoencoders for image-sequence datasets
│
├── main_colbert.py             # main colbert training script
├── main_colbert_for_odc.py     # clone of main script, used only for computing ODC
├── main_colbert_for_time.py    # clone of main script, used only for computing eval time
├── main_colbert_single_vec.py  # [exp] colbert ablation, using single vector scoring    
│
├── main.py                     # main opas training script
├── main_long_seq.py            # opas training script for long datasets
├── main_direct_score.py        # [exp] ablation, using ephi to generate scores directly
├── main_early_interaction.py   # [exp] ablation, using early interaction to get alignments
├── main_bert.py                # [exp] ablation, using bert-based ephi
│
├── run_inference.py            # main inference script
├── run_inference_long_seq.py   # inference for long datasets
├── run_inference_odc.py        # inference with ODC computation
├── run_inference_early.py      # [exp] inference with early interaction
│
├── train_prehash.py            # main script for training SetAggr
├── train_hyperplanes.py        # train hyperplanes using trained SetAggr
├── eval_lsh.py                 # evaluate trained SetAggr and hyperplanes
├── train_xfmprehash.py         # [exp] train some SetAggr while training OPAS
├── run_lsh_multi.py            # [utility] parallel hyperplane training
└── run_lsh_eval_multi.py       # [utility] parallel lsh evaluations
```

## Sanity Check
After installing the environment, extracting the datasets and the models uploaded [here](https://drive.google.com/drive/folders/1EVI_t2oObqU0uCeHaX1NrBUvg9Uvn3_H?usp=sharing) to `final_data/` and `models/` respectively, run `bash scripts/sanity_eval.sh <gpu_id>` to run the inference script `run_inference.py` on three experiment ids (lsun dataset was too large to share), on their respective datasets.

## Training
Use script `main.py` for training OPAS given the dataset is in the correct format in `final_data`

Scripts used to train OPAS on the Music, Speech, CIFAR, and LSUN datasets (numbers used in the paper) are given below. They use the defualt configurations for the respective datasets stored in `configs/`.  

```python
python main.py dataset=audio device=$device

python main.py dataset=speech device=$device

python main.py dataset=cifar-large device=$device

python main.py dataset=lsun device=$device
```

### Note: alternate training scripts
The scripts `main_direct_score.py` and `main_long_seq.py` are essentially the same script as `main.py`, with some special modifications for their resp. use cases. The direct score ablation discussed in Appendix L.7 uses the former, and long sequence datasets with N>=50 use the latter. If there is an attempt to run `main.py` with a long sequence dataset, an error will be raised and the script will abort. 

## Evaluation
For a trained model with experiment id `<expt_id>`, run only evaluation using

```python
python run_inference.py --expt_id <expt_id> --device $device --dataset $dataset
```

For cross-task results, simply change `$dataset` to a different dataset (assuming embedding dimensions, $d_{SR}$, etc are as described in the paper) at the time of running this command.

Note: there is a unique `run_inference_long_seq.py` as a companion script to `main_long_seq.py`. There is no companion inference script for `main_direct_score.py`. Please refer to the logfiles for those experiments to check test set metrics.

## Ablations
Refer to script `ablations.sh` for commands used to run the reported ablation studies.

## Timing and Memory
Refer to the `baselines/` folder for obtaining timing and memory data.

## Plots and Figures
Refer to the `plots_and_figures/` folder for the notebooks and code used for generating the plots for OPAS.

## OPAS and LSH (SetAggr and hyperplane training)

### SetAggr
The script `train_prehash.py` contains the code for training SetAggr. The command used for training SetAggr for the audio, speech and cifar datasets are given below:

```python
# audio
python train_prehash.py amsgrad=False device=$device expt_id=A01091914 hash_latent=1252 hash_outdim=256 hasher_type=SetTransformer loss_type=mse lr=1e-4 wandb=False

# speech
python train_prehash.py amsgrad=False device=$device expt_id=S28021253 hash_latent=1252 hash_outdim=256 hasher_type=SetTransformer loss_type=mse lr=1e-4 wandb=False

# cifar
python train_prehash.py amsgrad=False device=$device expt_id=C07040027 hash_latent=1252 hash_outdim=64 hasher_type=SetTransformer loss_type=mse lr=1e-5 wandb=False
```

Note: 
- there are multiple hasher_types which were tried, SortLRL, SortNoLRL, DeepSet and SetTransformer
- this script also uses negative exploration to sample negative pairs
- the main parameters for the models is `hash_latent` and `hash_outdim`, which refer to the latent dim and output dim of the SetAggr model being trained


### Hyperplanes
The script `train_hyperplanes.py` contains the code for training hyperplanes $W$. The script requires two main components to be trained using their respective scripts:
1. OPAS models stored in `models/<expt_id>/`, created using `main.py` 
2. SetAggr model stored in `hashing/<expt_id>/<hasher_expt_num>`, created using `train_prehash.py`

The commands used for training hyperplanes for audio, speech and cifar are given below:
```python
# audio
python train_hyperplanes.py expt_id=A01091914 hexpt_num=<hasher_expt_num> device=<device> nplanes=30 nbits=10 lr=1e-3 track_metric=index_spread neg_expl=800 l2v=2 l1=0.05 l2=0.05 l3=0.001 run_lsh_eval=True

# speech
python train_hyperplanes.py expt_id=S28021253 hexpt_num=<hasher_expt_num> device=<device> nplanes=30 nbits=10 lr=1e-3 track_metric=index_spread neg_expl=800 l2v=2 l1=0.01 l2=0.05 l3=0.001 run_lsh_eval=True

# cifar
python train_hyperplanes.py expt_id=S28021253 hexpt_num=<hasher_expt_num> device=<device> nplanes=30 nbits=10 lr=1e-3 track_metric=index_spread neg_expl=800 l2v=2 l1=0.001 l2=0.1 l3=0.001 run_lsh_eval=True
```

Note: 
- l2v refers to the version of bit balance loss, l2v=1 is the original, and l2v=2 is the modified, with the kronecker regularizer addition. 
- l1, l2, and l3 in the commands refer to the loss weights (denoted by $\kappa_i$ in the paper)
- the script `run_lsh_multi.py` and `run_lsh_eval_multi.py` are meant exclusively for parallel hyperparameter tuning the LSH loss weights. Extra parameters are being passed into the scripts `train_hyperplanes.py` and `eval_lsh.py` to maintain separation of runs which are running in parallel. These are left for archival purposes, for individual experiments please use the individual scripts only.


## Early interaction
The two scripts `main_early_interaction.py` and `run_inference_early.py` train and evaluate, respectively, OPAS models using `T` early interaction rounds. This has not been included in the paper due to poor performance.

# Questions
Please feel free to reach out with any specific questions or clarifications via GitHub by submitting an Issue, or, if preferred, on this email: opas76969@gmail.com