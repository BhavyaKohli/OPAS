# Configuration files

## Main OPAS training

For training OPAS models, the configuration file `base.yaml` contains all expected arguments. For each individual dataset, we have one config file with overrides. Additionally, when running the script, any argument may be overridden by passing through CLI using OmegaConf syntax (`python script.py argument_name=argument_value`).

## Hashing

For training SetAggr and to run hyperplane training, we have `hash_base.yaml` and `hyperplane_base.yaml` as the base parameters. For custom parameters, any argument may be overridden by passing through CLI using OmegaConf syntax (`python script.py argument_name=argument_value`).