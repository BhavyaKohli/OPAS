# Getting the data

## Audio

Given a list of youtube playlists, first download all the original songs to a temporary directory using `yt-dlp` (official repository [here](https://github.com/yt-dlp/yt-dlp)) and split them into fragments of `N` seconds each, keeping track of the id's of the original songs the fragments are taken from. This file (in `hdf5` format) containing fragments and song ids can be passed as an argument to `audio/generate_dataset_audio.py` to generate the required query-corpus dataset which will be stored in `../final_data`.

## Image sequence

For CIFAR-10, running `train_ae.py` with `dataset_name` set to "CIFAR" will automatically download the torchvision CIFAR10 dataset and store it in `cifar_gt`. Running `image_sequence/get_rotated_sequences.py` will also download the dataset to the same location.

For LSUN-C, please refer to the official repository [here](https://github.com/fyu/lsun) for download instructions. For compatibility with our dataset generation, store the train and test splits under `lsun_gt/train` and `lsun_gt/test` respectively. (In principle, a user could download any class from the LSUN data repository, as long as they are stored in the desired locations. We chose churches arbitrarily.)

After the respective autoencoder has been trained using `train_ae.py`, the weights will be stored in `image_sequence/embedding_models/`. Then, the script `image_sequence/get_rotated_sequences.py` needs to be executed giving the dataset name and the path to the autoencoder weights as the inputs. This will generate a `rotated_sequences_<dataset_name>.hdf5` file, which can be passed as an input to `image_sequence/generate_dataset_image_sequence.py`, which will create the required query-corpus dataset in `../final_data`.

Since the autoencoder weights are provided, to generate the cifar dataset, 
1. cd into `image_sequence`
2. run `python get_rotated_sequences.py --dataset_name CIFAR --autoencoder_weights embedding_models/cifar_ae.pkl --device 0` for processing the data on GPU 0 (change if required)
3. run `python generate_dataset_image_sequence.py --input_file rotated_sequences_CIFAR.hdf5 --dataset_name cifar --log`