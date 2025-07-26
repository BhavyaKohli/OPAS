import numpy as np
import argparse
import logging
import torch
import h5py
import os
from time import sleep

from tqdm import tqdm

import torch.nn.functional as F


logging.basicConfig(
    filename="datasets.log",
    filemode="a+",
    level=logging.INFO,
    format="%(levelname)s (%(asctime)s): %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S %p"
)


class QueryDatasetBinary:
    def __init__(
            self, corpus_sequence_set, num_queries_per_corpus, dataset_save_path, 
            query_size=5, seed=10, consecutive=0, verbose=False, split='train',
            test_size=0, sort_fraction=0.15
        ):
        file, split = corpus_sequence_set
        self.C_labels, self.C_orig = file[f"{split}/labels"], file[f"{split}/orig"]
        print(self.C_orig.shape)
        self.Qsize = query_size
        self.Qnumtrain = num_queries_per_corpus # int((1-test_size) * num_queries_per_corpus) 
        self.Qnumtest = 0 # int(test_size * num_queries_per_corpus)
        self.Qtotal = num_queries_per_corpus
        self.sort_fraction = sort_fraction
        self.consecutive = consecutive

        self.C_list = file[f"{split}/data"]

        np.random.seed(seed)
        orig_dataset_save_path = dataset_save_path.replace(".hdf5", "_orig.hdf5")

        if os.path.exists(dataset_save_path):
            print("Dataset exists at path, removing in 2 seconds...")
            sleep(2)
            os.remove(dataset_save_path)
        
        if os.path.exists(orig_dataset_save_path):
            print("Dataset exists at path, removing...")
            os.remove(orig_dataset_save_path)
        
        dataset_file = h5py.File(dataset_save_path, 'a')
        orig_dataset_file = h5py.File(orig_dataset_save_path, 'a')
        dataset_file.create_dataset(f"C", data=self.C_list)
        orig_dataset_file.create_dataset(f"C", data=self.C_orig)

        for n, c in enumerate(tqdm(range(len(self.C_list)), disable=not verbose, ncols=100)):
            # c : 0 to 599
            corpus_item = torch.from_numpy(self.C_list[c]) # (20, 384)
            corpus_item_orig = torch.from_numpy(self.C_orig[c]) # (20, 3, 32, 32)
            labels = torch.from_numpy(self.C_labels[c])    # (10) 10 positives per query

            tridx, _ = self.get_idxs_from_corpus(corpus_item)
            tr = torch.stack([corpus_item[idx] for idx in tridx])
            tr_orig = torch.stack([corpus_item_orig[idx] for idx in tridx])

            tr_labels = labels[None].repeat_interleave(len(tr), dim=0)

            if n==0:
                dataset_file.create_dataset(f"Q", data=tr, maxshape=[None]*len(tr.shape))
                dataset_file.create_dataset(f"labels", data=tr_labels, maxshape=[None]*len(tr_labels.shape))
                orig_dataset_file.create_dataset(f"Q", data=tr_orig, maxshape=[None]*len(tr_orig.shape))
                orig_dataset_file.create_dataset(f"labels", data=tr_labels, maxshape=[None]*len(tr_labels.shape))
            
            else:
                dataset_file[f"Q"].resize((dataset_file[f"Q"].shape[0] + tr.shape[0]), axis=0)
                dataset_file[f"Q"][-tr.shape[0]:] = tr
                orig_dataset_file[f"Q"].resize((orig_dataset_file[f"Q"].shape[0] + tr_orig.shape[0]), axis=0)
                orig_dataset_file[f"Q"][-tr_orig.shape[0]:] = tr_orig
                dataset_file[f"labels"].resize((dataset_file[f"labels"].shape[0] + tr_labels.shape[0]), axis=0)
                dataset_file[f"labels"][-tr_labels.shape[0]:] = tr_labels
                orig_dataset_file[f"labels"].resize((orig_dataset_file[f"labels"].shape[0] + tr_labels.shape[0]), axis=0)
                orig_dataset_file[f"labels"][-tr_labels.shape[0]:] = tr_labels


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
    parser.add_argument('--input_file', type=str, help="Path to input `rotated_sequences` hdf5 file with the following keys: `data`- containing sequences of image embeddings of size `N x embed_dim`, `orig`- containing the sequence of original images of size (N x 3xWxH), and `labels`- containing the locations of all rotated sequences which have been generated from the same source image (for finding positives for a given query)")
    parser.add_argument('--dataset_name', type=str, help="Name of the dataset to be created")
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--num_q_per_c', type=int, default=2, help="Number of queries to be generated per corpus item")
    parser.add_argument('--num_songs', type=int, default=-1)
    parser.add_argument('--M', type=int, default=6, help="Length of each query")
    parser.add_argument('--consecutive', type=int, default=0, help="[deprecated, unused] non-zero when grouped queries are desired")
    parser.add_argument('--log', action="store_true")
    args = parser.parse_args()

    DEVICE = 'cpu'
    if args.log : logging.info(f"{DEVICE=}")

    if args.num_songs == -1:
        num_songs = np.inf
    else :
        num_songs = args.num_songs

    if args.consecutive!=0 : 
        assert args.consecutive > 0
        args.consecutive -= args.M % args.consecutive 

    root = "../../final_data/"
    dataset_name = args.dataset_name
    dataset_dir = os.path.join(root, dataset_name)
    
    os.makedirs(dataset_dir, exist_ok=True)
    if args.log : logging.info(f"Created dataset directory at {dataset_dir}")
    
    file = h5py.File(args.input_file, "r")

    seed = 15
    main_dataset = QueryDatasetBinary(
        corpus_sequence_set=[file, "train"],
        num_queries_per_corpus=args.num_q_per_c,
        dataset_save_path=os.path.join(dataset_dir, "dataset_train.hdf5"),
        test_size=0, sort_fraction=1,
        query_size=args.M,
        consecutive = args.consecutive,
        seed=seed,
        verbose=1
    )
    
    main_dataset = QueryDatasetBinary(
        corpus_sequence_set=[file, "test"],
        num_queries_per_corpus=args.num_q_per_c,
        dataset_save_path=os.path.join(dataset_dir, "dataset_test.hdf5"),
        test_size=0, sort_fraction=1,
        query_size=args.M,
        consecutive = args.consecutive,
        seed=seed,
        verbose=1,
        split='test'
    )

    main_dataset = QueryDatasetBinary(
        corpus_sequence_set=[file, "val"],
        num_queries_per_corpus=args.num_q_per_c,
        dataset_save_path=os.path.join(dataset_dir, "dataset_val.hdf5"),
        test_size=0, sort_fraction=1,
        query_size=args.M,
        consecutive = args.consecutive,
        seed=seed,
        verbose=1,
        split='val'
    )