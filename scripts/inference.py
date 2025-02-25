import sys, os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("device", type=str)
    parser.add_argument("expt_ids", type=str, nargs="+")
    parser.add_argument("--early", action="store_true")
    args = parser.parse_args()

    script = "run_inference_early.py" if args.early else "run_inference.py"

    for expt_id in args.expt_ids:
        ret = os.system(f"python {script} --device {args.device} --expt_id {expt_id} --dataset {args.dataset}")
        if ret == 2:
            break
