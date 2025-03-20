import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, required=True, help="name of dataset, stored in `final_data/`")
parser.add_argument("--print_dataset", action="store_true", help="prints dataset array shapes when passed")
parser.add_argument("--num_q", type=int, default=300, help="number of queries to be sampled from the dataset")
parser.add_argument("--wandb_log", action="store_true", help="whether to log to wandb")
parser.add_argument("--seed", type=int, default=69, help="random seed")
parser.add_argument("--reproducible", type=int, default=1, help="pass 0 to disable reproducible training")

parser.add_argument("--b", type=float, default=1.0, help="hinge margin for negative gap penalty (b-Apa)")
parser.add_argument("--b1", type=float, default=0.0, help="hinge margin for positive gap penalty (Apa-b)")
parser.add_argument("--delta", type=float, default=0.7, help="hinge margin for contrastive loss")
parser.add_argument("--stagger", type=int, default=0, help="number of staggers when creating model input")
parser.add_argument("--lamwt", type=float, default=1e-2, help="loss coefficient for negative gap penalty")
parser.add_argument("--gapwt", type=float, default=0.0, help="loss coefficient for positive gap penalty")
parser.add_argument("--SNR", type=int, default=1, help="input white noise SNR")
parser.add_argument("--noise", type=float, default=0, help="input white noise")
parser.add_argument("--preembed", type=str, help="pre-embed model to use (tokenize, linear, conv, none)")

parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--nepochs", type=int, default=30, help="number of epochs")
parser.add_argument("--device", type=int, default=0, help="CUDA device to use (default: 0)")
parser.add_argument("--batch_size", type=int, default=200, help="batch size for training")

parser.add_argument("--xnhead", type=int, default=4, help="nhead for transformer encoder layer")
parser.add_argument("--xnumlayers", type=int, default=2, help="num layers for transformer encoder")
parser.add_argument("--xoutdim", type=int, default=128, help="out dim for transformer encoder model (4000 -> outdim)")
parser.add_argument("--xlr", type=float, default=1e-5, help="learning rate for transformer encoder model")
parser.add_argument("--xwarmupepochs", type=int, default=20, help="warmup epochs for transformer encoder model")
parser.add_argument("--xff", type=int, default=2048, help="feedforward dim of transformer encoder model")

parser.add_argument("--use_sing_xfmer", action="store_true", help="pass when a single transformer is to be used, for both q and c")
parser.add_argument("--no_tokenize", action="store_true", help="pass when transformer pre-inputs should NOT be tokenized")
parser.add_argument("--enforce_order", action="store_true", help="pass when enforce order during training")
parser.add_argument("--train_with_orig", action="store_true", help="when using the cifar or lsun datasets, pass this when training should be done using the original images and not the embeddings directly (not recommended for LSUN)")
parser.add_argument("--deepset", action="store_true", help="pass when query and corpus sequences are to be embedded into the same latent space, for using a (hq-hc)+ loss instead of the usual sinkhorn->permutation->normscore+lamscore components")
parser.add_argument("--deepset_mode", type=int, default=2, help="mode 2 is normalized, mode 1 is not (only used when deepset is passed)")
parser.add_argument("--add_attention", action="store_true", help="[unused in current script version] pass when an additional attention module is required before the Xfmer")
parser.add_argument("--lfcc", action="store_true", help="[unused in current script version] pass when LFCC features should be used in place of audio")

# sinkhorn params
parser.add_argument("--n_sink_iter", type=int, default=20, help="number of sinkhorn iterations")
parser.add_argument("--single_step_norm", type=int, default=-1, choices=[-1,1,2], help="controls how P is obtained from F during training. -1 means no single step normalization (sinkhorn iterations), 1 means single step normalization using sum, 2 means single step normalization using exp (attn)")

# post-review
parser.add_argument("--internal_lamwt", type=float, default=1, help="coefficient for A^T @ lambda @ a^T inside F_mat")
parser.add_argument("--use_linear_lammodel", action="store_true", help="pass when a linear model should be used for lambda instead of a transformer")
parser.add_argument("--debug", action="store_true", help="no logging to file")

# new experiments
parser.add_argument("--no_lammodel", action="store_true", help="pass when sinkhorn matrix is to be computed without lammodel, using -relu(hq-hc)")
parser.add_argument("--no_lamrelu", action="store_true", help="pass when lamscore should not be relu'd")
parser.add_argument("--pretrain_embedding", action="store_true", help="pass when transformer encoder should be pretrained")
parser.add_argument("--pretrain_budget", type=float, default=0.25, help="fraction of queries and corpus to be used for pretraining")
parser.add_argument("--pretrain_epochs", type=int, default=30, help="number of epochs for pretraining")
parser.add_argument("--skip_embed", action="store_true", help="pass to skip embed_model completely (embed_model will be replaced with nn.Identity())")
parser.add_argument("--skip_type", type=str, default="lin", help="type of model to use (lin, conv, lstm)")
