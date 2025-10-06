# Getting the data

## Audio

Given a list of youtube playlists, first download all the original songs to a temporary directory using `yt-dlp` (official repository [here](https://github.com/yt-dlp/yt-dlp)). For downloading the music and speech datasets we used, `audio/get_youtube_videos.py` downloads all audio files in `.wav` format into `music_raw/` and `speech_raw`. The following processing steps needs to be carried out for any folder filled with audio files is as follows:

1. For each song (identified by id "i") Load each file, resample it to a uniform same sampling rate (we chose 4000) and split it into fragments of `N` seconds each, discarding any excess. Example: a 170s song will give 8 fragments of 20s each, with 10s excess which is discarded. These 8 seconds in tensor form will have a shape ($20\times d_{SR}$)
2. Store these fragments, along with the song id "i" in the format `fragment_1,...,fragment_k` & `i,...,i` (k times). 
3. Create a  file (in `hdf5` format) containing fragments and song ids. This file can be passed as an argument to `audio/generate_dataset_audio.py` to generate the required query-corpus dataset which will be stored in `../final_data`.
4. For this, use the `process_and_save_data.py` script either directly, or as reference.
5. Sample usage of the generate dataset script: `python generate_dataset_audio.py --input_file audio_processed/samp4000dur20.hdf5 --dataset_name audio --M 6 --log`


## Image sequence

For CIFAR-10, running `train_ae.py` with `dataset_name` set to "CIFAR" will automatically download the torchvision CIFAR10 dataset and store it in `image_sequence/cifar_gt`. Running `image_sequence/get_rotated_sequences.py` will also download the dataset to the same location.

For LSUN-C, please refer to the official repository [here](https://github.com/fyu/lsun) for download instructions. For compatibility with our dataset generation, store the train and test splits under `lsun_gt/train` and `lsun_gt/test` respectively.

After the respective autoencoder has been trained using `train_ae.py`, the weights will be stored in `image_sequence/embedding_models/`. Then, the script `image_sequence/get_rotated_sequences.py` needs to be executed giving the dataset name and the path to the autoencoder weights as the inputs. This will generate a `rotated_sequences_<dataset_name>.hdf5` file, which can be passed as an input to `image_sequence/generate_dataset_image_sequence.py`, which will create the required query-corpus dataset in `../final_data`.

Since the autoencoder weights are provided, to generate the cifar dataset, 
1. cd into `image_sequence`
2. run `python get_rotated_sequences.py --dataset_name CIFAR --autoencoder_weights embedding_models/cifar_ae.pkl --device 0` for processing the data on GPU 0 (change if required)
3. run `python generate_dataset_image_sequence.py --input_file rotated_sequences_CIFAR.hdf5 --dataset_name cifar --log`


## Long datasets

For generating the long sequence datasets, run `process_and_save_data.py` with `single_dur = 40` (line 40), then run the generate_dataset script as usual.