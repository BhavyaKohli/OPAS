import os
import sys
import glob
import h5py
import librosa
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def get_song(path, target_samp_rate=None):
    song, samp_rate = librosa.load(path, sr=target_samp_rate)

    num_sec = len(song) // samp_rate
    if num_sec >= 600:
        return None, None
    song_og = song[:samp_rate * (num_sec)]
    song = np.stack(np.array_split(song_og, num_sec))

    # song = MinMaxScaler(feature_range=(0, 1)).fit_transform(song)
    return samp_rate, song

dataset = sys.argv[1] 
root = f"{dataset}_raw/"
ls = glob.glob(os.path.join(root, "*.wav"))

samp_rates, songs = [], []
samp_rate_override = 4000

for path in tqdm(ls):
    samp, song = get_song(path, target_samp_rate=samp_rate_override)
    if samp is None:
        continue
    samp_rates.append(samp)
    songs.append(song)

SONGS = []
SONGS_id = []
single_dur = 20
print(f"Splitting songs into {single_dur} second pieces")

for n,song in tqdm(enumerate(songs)):
    pieces = [i for i in np.array_split(song, [single_dur*i for i in range(len(song) // single_dur + 1)]) if i.shape[0] == single_dur]
    SONGS += pieces
    SONGS_id += [n]*len(pieces)

# randomly permuting the corpuses
np.random.seed(0)
perm = np.random.permutation(range(len(SONGS)))
SONGS = np.array(SONGS)[perm]
SONGS_id = np.array(SONGS_id)[perm]


save_dir = f"{dataset}_processed/"
os.makedirs(save_dir, exist_ok=True)

file = h5py.File(os.path.join(save_dir, f"samp{samp_rate_override}_dur{single_dur}.hdf5"), "w")
file.create_dataset("audio", data=SONGS, dtype=np.float64)
file.create_dataset("labels", data=SONGS_id, dtype=int)
file.close()