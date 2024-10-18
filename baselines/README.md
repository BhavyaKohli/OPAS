# Baselines and timing comparisons with OPAS

`timing_comparisons.py` requires:
1. valid experiment folder in `../models/`
2. the corresponding dataset in `../final_data/` (the same one used when running the experiment)

`baselines_time_vs_perf.py` requires:
1. dataset tensors in `tensors/`, either generated using `tensors/get_tensors.py`
2. valid dataset in `../final_data` (used by `tensors/get_tensors.py`)
   
`baselines_mem_vs_perf_SINGLE.py` is not expected to be run as is, please run `runall_mem_vs_perf.py` instead, which requires:
1. dataset tensors in `tensors/` (same as above)