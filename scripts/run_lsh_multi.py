import os

from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool

def run_expt(i, device, l1, l2, l3):
    cmd = f"python train_hyperplanes.py expt_id=A01091914 hexpt_num=24 device={device} nplanes=30 nbits=10 lr=0.0001 track_metric=index_spread neg_expl=1200 l2v=2 l1={l1} l2={l2} l3={l3} hplanes_id_override={i} disable_logging=True run_lsh_eval=True save_expt_num={i}"
    ret = os.system(cmd)
    if ret == 2:
        print("Interrupting...")
        exit()


if __name__ == "__main__":
    logger.remove(0)
    logger.add("tmp/lsh_multi.log", level="INFO", format="{time:D-MM-YYYY HH:mm:ss} | {level} | {message}")

    device = 0

    l1s = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    l2s = [1e-2, 5e-2, 1e-1, 5e-1, 1]
    l3s = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
    combs = [[x, y, z] for x in l1s for y in l2s for z in l3s]
    
    gpus = [0, 1, 2, 6]
    
    for i, c in enumerate(combs):
        combs[i] = [i, gpus[i%4]] + c

    with Pool(16) as pool:
        pool.starmap(run_expt, tqdm(combs, total=len(combs), desc="Running experiments..."))