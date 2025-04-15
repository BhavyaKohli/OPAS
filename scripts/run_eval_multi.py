import os
import argparse
import numpy as np

from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_id", type=str, default="A01091914")
    parser.add_argument("--hexpt_num", type=int, default=24)
    parser.add_argument("--gpus", nargs="+")
    args = parser.parse_args()

    gpus = [int(i) for i in args.gpus]
    expt_id = args.expt_id
    hexpt_num = args.hexpt_num

    l1s = [1e-4, 1e-3, 1e-2, 5e-2]
    l2s = [1e-2, 5e-2, 1e-1, 5e-1, 1]
    l3s = [1e-4, 1e-3]
    combs = [[x, y, z] for x in l1s for y in l2s for z in l3s]

    for i, c in enumerate(combs):
        combs[i] = [i, gpus[i % (len(gpus))]] + c

    with tqdm(total=len(combs), desc="Running experiments...") as pbar:
        def run_expt(i, device, l1, l2, l3):
            cmd = f"python eval_lsh.py expt_id={expt_id} hexpt_num={hexpt_num} device={device} m=10 L=30 hplanes=hyperplanes_{i} kbits=[10,8,7,6,5,4,2,1] seed=69 save_expt_num={i}{i} skip_logging=True"
            ret = os.system(cmd)
            if ret == 2:
                print("Interrupting...")
                exit()

            try:
                perf = np.load(f"tmp/multi/tmp_{i}{i}.npy").tolist()
                perf = [l1, l2, l3] + perf
                np.save(f"tmp/multi/tmp_{i}{i}.npy", perf)
            except:
                with open("tmp/multi/fails.txt", "a") as f:
                    f.write(f"{i} {device} {l1} {l2} {l3} failed\n")
                    
            try:
                with open(f"tmp/lsh_multi_final_{expt_id[0]}.txt", "a+") as f:
                    f.write(f"{perf}\n")
            except:
                pass
            pbar.update(1)

        with Pool(6*len(gpus)) as pool:
            pool.starmap(run_expt, combs, chunksize=1)