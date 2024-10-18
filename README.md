# OPAS

## datasets
Download `final_data.zip` from <here> and unzip its contents into `final_data`

For custom datasets, refer to `data/audio` for audio-based datasets, and `data/image_sequence` for image sequence datasets.

## training
Use script `main.py` for training OPAS given the dataset is in the correct format in `final_data`

Scripts used to train OPAS on the Music, Speech, CIFAR, and LSUN datasets (numbers used in the paper) are pre-set in `scripts/train.sh`. This can be invoked simply by running `bash scripts/train.sh <dataset> <device>` where `<dataset>` is one of--"audio", "speech", "cifar", "lsun", and `<device>` is the GPU id for the GPU to use for training. Example: `bash scripts/train.sh audio 0`