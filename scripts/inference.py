import sys, os
import subprocess

if __name__ == "__main__":
    dataset = sys.argv[1]
    device = sys.argv[2]
    early = sys.argv[3]
    expt_ids = sys.argv[4:]

    script = "run_inference_early.py" if early else "run_inference.py"

    for expt_id in expt_ids:
        ret = subprocess.run(f"python {script} --device {device} --expt_id {expt_id} --dataset {dataset}", shell=True)
        print(ret)
