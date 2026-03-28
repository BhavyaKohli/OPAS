import numpy as np
import argparse
import logging
import torch
import h5py
import os
from time import sleep

from tqdm import tqdm


logging.basicConfig(
    filename="datasets.log",
    filemode="a+",
    level=logging.INFO,
    format="%(levelname)s (%(asctime)s): %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S %p"
)


def mask_sequence_and_reconstruct(corpus_item, mask_prob, max_mask_len, replace, start_pos=None, seed=None):
    # corpus item of shape (n x samp_rate)
    if torch.rand(1)[0] > mask_prob:
        return corpus_item
    
    np.random.seed(seed)
    item = corpus_item.flatten().clone()
    mask_idx1 = start_pos if start_pos is not None else np.random.choice(range(len(item)-max_mask_len))
    mask_idx2 = np.random.choice(np.linspace(mask_idx1, mask_idx1+max_mask_len, 11, endpoint=True)[1:]).astype(int)
    
    if replace: 
        item[mask_idx1:mask_idx2] = 0
    else:
        item = torch.cat([item[:mask_idx1], torch.zeros(mask_idx2-mask_idx1), item[mask_idx2:]])[:len(item)]
        
    return item.reshape(corpus_item.shape)


class QueryDatasetBinary:
    def __init__(
            self, corpus_sequence_set, num_queries_per_corpus, dataset_save_path, 
            query_size=5, seed=10, consecutive=0, verbose=False, split='train',
            test_size=0, sort_fraction=0.15
        ):

        self.C_list = corpus_sequence_set
        print(self.C_list.shape)           # (600,20,4000)
        self.Qsize = query_size
        self.Qnumtrain = num_queries_per_corpus # int((1-test_size) * num_queries_per_corpus) 
        self.Qnumtest = 0 # int(test_size * num_queries_per_corpus)
        self.Qtotal = num_queries_per_corpus
        self.sort_fraction = sort_fraction
        self.consecutive = consecutive

        np.random.seed(seed)

        if os.path.exists(dataset_save_path):
            print("Dataset exists at path, removing in 2 seconds..")
            sleep(2)
            os.remove(dataset_save_path)
        dataset_file = h5py.File(dataset_save_path, 'a')

        test, train = [], []
        for n, c in enumerate(tqdm(range(len(self.C_list)), disable=not verbose, ncols=100)):
            corpus_item = torch.from_numpy(self.C_list[c]) # (20, 4000)

            Nmax = corpus_item.shape[0]
            randlength = np.random.randint(Nmax-5, Nmax)
            corpus_item = corpus_item[:randlength]

            samp_rate = corpus_item.shape[-1]
            citemlist = [corpus_item]

            # 15 random masks, no specific position, max 0.2 sec
            for _ in range(15):
                citemlist.append(mask_sequence_and_reconstruct(corpus_item, mask_prob=1, max_mask_len=samp_rate//5, replace=True))
            citemlist = torch.stack(citemlist)

            # citemlist is now (16 x randlength x samp_rate), need to pad to Nmax
            citemlist = torch.cat([citemlist, torch.zeros_like(citemlist)[:, :(Nmax-randlength), :]], dim=1)

            # citemlist now has 1 + 15 = 16 items => (16 x N x samp_rate)
            # all of these are "positives" for all queries generated from the original corpus_item

            tridx, _ = self.get_idxs_from_corpus(corpus_item)
            tr = torch.stack([corpus_item[idx] for idx in tridx])

            # tr : (num_q_per_c x M x samp_rate)
            # the label of the original corpus item is "n"
            # need to assign this value to all corpuses in citemlist
            
            tr_labels = torch.from_numpy(np.array([i+n*len(citemlist) for i in range(len(citemlist))])).unsqueeze(0).repeat_interleave(len(tr), dim=0)
            # c_labels = torch.zeros(len(citemlist)).fill_(n*len(citemlist))
            # 0-15, then 16-31, then 32-47, will be corpus items 0, 1, 2 respectively 
            # (with their shifted variants)

            if n==0:
                dataset_file.create_dataset(f"Q", data=tr, compression="gzip", chunks=True, maxshape=[None]*len(tr.shape))
                dataset_file.create_dataset(f"labels", data=tr_labels, compression="gzip", chunks=True, maxshape=[None]*len(tr_labels.shape))
                dataset_file.create_dataset(f"C", data=citemlist, compression="gzip", chunks=True, maxshape=[None]*len(citemlist.shape))
            
            else:
                dataset_file[f"Q"].resize((dataset_file[f"Q"].shape[0] + tr.shape[0]), axis=0)
                dataset_file[f"Q"][-tr.shape[0]:] = tr
                dataset_file[f"labels"].resize((dataset_file[f"labels"].shape[0] + tr_labels.shape[0]), axis=0)
                dataset_file[f"labels"][-tr_labels.shape[0]:] = tr_labels
                dataset_file[f"C"].resize((dataset_file[f"C"].shape[0] + citemlist.shape[0]), axis=0)
                dataset_file[f"C"][-citemlist.shape[0]:] = citemlist


            if (n+1)%(len(self.C_list)//10) == 0:
                trshape = dataset_file[f"Q"].shape
                if args.log : logging.info(f"At corpus item {n+1}, currently saved {trshape} training data indices")

        dataset_file.close()

    def get_idxs_from_corpus(self, corpus_item):
        # corpus_item : (20, 4000)
        Qidxs = np.stack([np.random.choice(range(len(corpus_item)), size=self.Qsize, replace=False) for _ in range(self.Qtotal+20)])
        Qidxs = np.unique(Qidxs, axis=0)[:self.Qtotal]

        Qidxs = Qidxs[np.random.permutation(len(Qidxs))]
        train, test = Qidxs[:self.Qnumtrain], Qidxs[self.Qnumtrain:]

        train, train_labels = self.sort_some(train, self.sort_fraction)

        return train, train_labels

    @staticmethod
    def sort_some(Qidxs, sort_fraction):
        if sort_fraction == 0:
            return Qidxs

        labels = torch.zeros(len(Qidxs))
        
        sort_idx = np.random.choice(range(len(Qidxs)), size=int(len(Qidxs)*sort_fraction), replace=False)
        Qidxs[sort_idx] = np.sort(Qidxs[sort_idx], axis=1)
        labels[sort_idx] = 1

        return torch.from_numpy(Qidxs), labels
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="Path to input hdf5 file with the following keys: `audio`- containing a long list with items of shape (N x samp_rate) and `labels`- containing the original song labels for each item in `audio`")
    parser.add_argument('--dataset_name', type=str, help="Name of the dataset to be created")
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--num_q_per_c', type=int, default=10, help="Number of queries to be generated per corpus item")
    parser.add_argument('--num_songs', type=int, default=-1)
    parser.add_argument('--M', type=int, default=6, help="Length of each query")
    parser.add_argument('--consecutive', type=int, default=0, help="[deprecated, unused] non-zero when grouped queries are desired")
    parser.add_argument('--log', action="store_true")
    args = parser.parse_args()

    if args.num_songs == -1:
        num_songs = np.inf
    else :
        num_songs = args.num_songs

    if args.consecutive!=0 : 
        assert args.consecutive > 0
        args.consecutive -= args.M % args.consecutive 

    DEVICE = 'cpu'
    if args.log : logging.info(f"{DEVICE=}")

    root = "../../final_data/"
    dataset_name = args.dataset_name
    dataset_dir = os.path.join(root, dataset_name)
    
    os.makedirs(dataset_dir, exist_ok=True)
    if args.log : logging.info(f"Created dataset directory at {dataset_dir}")

    # expected input file structure:
    # keys: audio, labels
    # file['audio'] : (N x samp_rate)
    # these N-length items are collected from different songs, each with a different label
    # file['labels']: (N) 
    # these labels are the original song labels for each item in `audio`
    # example: 2 songs of length 80 seconds each, with 20 second clips, will have 8 items with labels 0,0,0,0,1,1,1,1

    file = h5py.File(args.input_file, 'r')
    SONGS = file['audio'] 
    SONGS_id = file['labels']

    if args.log : logging.info(f"Found {len(SONGS)} songs, tensor of shape: {SONGS.shape}")
    
    seed = 15
    main_dataset = QueryDatasetBinary(
        corpus_sequence_set=SONGS[:600],
        num_queries_per_corpus=args.num_q_per_c,
        dataset_save_path=os.path.join(dataset_dir, "dataset_train.hdf5"),
        test_size=0, sort_fraction=1,
        query_size=args.M,
        consecutive = args.consecutive,
        seed=seed,
        verbose=1
    )

    subs = [i for i in np.unique(SONGS_id) if i not in np.unique(SONGS_id[:600])]
    if SONGS.shape[-1] != 4000:
        subs_val = subs[:90]
        subs_test = subs[90:180]
    else:
        subs_val = subs[:30]
        subs_test = subs[30:60]
    
    idxs = []
    for i in range(len(SONGS_id)):
        if SONGS_id[i] in subs_test:
            idxs.append(i)
    idxs = np.array(idxs)

    main_dataset = QueryDatasetBinary(
        corpus_sequence_set=SONGS[idxs],
        num_queries_per_corpus=args.num_q_per_c,
        dataset_save_path=os.path.join(dataset_dir, "dataset_test.hdf5"),
        test_size=0, sort_fraction=1,
        query_size=args.M,
        consecutive = args.consecutive,
        seed=seed,
        verbose=1,
        split='test'
    )

    idxs = []
    for i in range(len(SONGS_id)):
        if SONGS_id[i] in subs_val:
            idxs.append(i)
    idxs = np.array(idxs)

    main_dataset = QueryDatasetBinary(
        corpus_sequence_set=SONGS[idxs],
        num_queries_per_corpus=args.num_q_per_c,
        dataset_save_path=os.path.join(dataset_dir, "dataset_val.hdf5"),
        test_size=0, sort_fraction=1,
        query_size=args.M,
        consecutive = args.consecutive,
        seed=seed,
        verbose=1,
        split='val'
    )