# OPAS

## datasets
Download `final_data.zip` from [here](https://drive.google.com/drive/folders/1oT5lfTuG-VnygbuUwOOQmbkLvQncN_vu?usp=sharing) and unzip its contents into `final_data`. Due to size constraints, we provide only the processed test subset from the Music, Speech and CIFAR datasets. This will enable the execution of `run_inference.py` along with the timing and memory comparisons (on datasets other than LSUN) in `baselines/`.

For custom datasets, refer to `data/audio` for audio-based datasets, and `data/image_sequence` for image sequence datasets.

The directory structure after downloading the datasets and model files should be as follows:
```
.
├── baselines
│   ├── baselines_mem_vs_perf_SINGLE.py
│   ├── baselines_time_vs_perf.py
│   ├── configs
│   │   ├── main.config
│   │   └── opas_mem.config
│   ├── get_opas_mem.py
│   ├── inference.log
│   ├── README.md
│   ├── runall_mem_vs_perf.py
│   ├── sdtw.py
│   ├── sharp.py
│   ├── tensors
│   │   └── get_tensors.py
│   └── timing_comparisons.py
├── data
│   ├── lsun_gt
│   ├── cifar_gt
│   ├── audio
│   │   └── generate_dataset_audio.py
│   ├── image_sequence
│   │   ├── embedding_models
│   │   │   ├── cifar_ae.pkl
│   │   │   └── lsun_ae.pkl
│   │   ├── generate_dataset_image_sequence.py
│   │   └── get_rotated_sequences.py
│   └── README.md
├── final_data
│   ├── audio
│   │   └── dataset_test.hdf5
│   ├── cifar
│   │   └── dataset_test_orig.hdf5
│   ├── README.md
│   └── speech
│       └── dataset_test.hdf5
├── main.py
├── models
│   ├── A01091914
│   │   ├── args.pkl
│   │   .
│   │   └── scmodel.pt
│   ├── AH05092001
│   │   ├── args.pkl
│   │   .
│   │   └── scmodel.pt
│   ├── C30082224
│   │   ├── args.pkl
│   │   .
│   │   └── scmodel.pt
│   ├── L13090014
│   │   ├── args.pkl
│   │   .
│   │   └── scmodel.pt
│   └── README.md
├── opas
│   ├── data.py
│   ├── gumbel_sinkhorn_ops.py
│   ├── metrics
│   │   ├── soft_dtw_cuda.py
│   │   └── soft_dtw.py
│   ├── models
│   │   ├── cifar_embed.py
│   │   ├── deepset.py
│   │   ├── lsun_embed.py
│   │   ├── main.py
│   │   └── ts_encoders.py
│   ├── tstok
│   │   ├── generic.py
│   │   ├── __pycache__
│   │   │   └── tokenizer.cpython-38.pyc
│   │   ├── tokenizer.py
│   │   └── tsutils.py
│   └── utils.py
├── plots_and_figures
│   ├── arial.ttf
│   ├── data
│   │   ├── memory_fastdtw.log
│   │   .
│   │   └── times_speech.pkl
│   ├── map_vs_memory.pdf
│   ├── map_vs_time.pdf
│   ├── plots_map_vs_mem.ipynb
│   └── plots_map_vs_time.ipynb
├── README.md
├── run_inference.py
├── scripts
│   └── train.sh
└── train_ae.py
```

## training
Use script `main.py` for training OPAS given the dataset is in the correct format in `final_data`

Scripts used to train OPAS on the Music, Speech, CIFAR, and LSUN datasets (numbers used in the paper) are pre-set in `scripts/train.sh`. This can be invoked simply by running `bash scripts/train.sh <dataset> <device>` where `<dataset>` is one of--"audio", "speech", "cifar", "lsun", and `<device>` is the GPU id for the GPU to use for training.