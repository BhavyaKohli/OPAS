import os

for dataset in ["cifar", "lsun", "imagenet", "audio"]:
    for method in ["sharp", "sdtw", "mass", "fastdtw"]:
        os.makedirs("../plots_and_figures/data/", exist_ok=True)
        with open(f"../plots_and_figures/data/memory_{method}.log", "a") as f:
            print(f"Dataset: {dataset}", file=f)

        SKIP_LIST = [10, 50, 100, 200, 500, 750, 1000]
        if dataset == "lsun":
            SKIP_LIST = [500, 1000, 15000, 20000, 30000, 40000, 50000, 90000]

        for SKIP in SKIP_LIST:
            cmd = f"python baselines_mem_vs_perf_SINGLE.py --method {method} --dataset {dataset} --skip {SKIP}"
            ret = os.system(cmd)
            if ret == 1:
                print("Error in execution, exiting...")
                break