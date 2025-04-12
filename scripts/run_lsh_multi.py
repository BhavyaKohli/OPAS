import os
import argparse

from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_id", type=str, default="A01091914")
    parser.add_argument("--hexpt_num", type=int, default=24)
    parser.add_argument("--gpus", nargs="+")
    args = parser.parse_args()

    logger.remove(0)
    logger.add("tmp/lsh_multi.log", level="INFO", format="{time:D-MM-YYYY HH:mm:ss} | {level} | {message}")

    device = 0

    l1s = [1e-4, 1e-3, 1e-2, 5e-2]
    l2s = [1e-2, 5e-2, 1e-1, 5e-1, 1]
    l3s = [1e-4, 1e-3]
    combs = [[x, y, z] for x in l1s for y in l2s for z in l3s]
    
    expt_id = args.expt_id
    hexpt_num = args.hexpt_num
    gpus = [int(i) for i in args.gpus]

    def run_expt(i, device, l1, l2, l3):
        cmd = f"python train_hyperplanes.py expt_id={expt_id} hexpt_num={hexpt_num} device={device} nplanes=30 nbits=10 lr=0.0001 track_metric=index_spread neg_expl=800 l2v=2 l1={l1} l2={l2} l3={l3} hplanes_id_override={i} disable_logging=True run_lsh_eval=True save_expt_num={i}"
        ret = os.system(cmd)
        if ret == 2:
            print("Interrupting...")
            exit()
    
    for i, c in enumerate(combs):
        combs[i] = [i, gpus[i%(len(gpus))]] + c

    with Pool(3*len(gpus)) as pool:
        pool.starmap(run_expt, tqdm(combs, total=len(combs), desc="Running experiments..."))