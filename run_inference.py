import os

from main import *
from opas.utils import get_opas_constants


@torch.no_grad()
def compute_metrics(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False, aggregator=None):

    loader = dataset.get_dataloader(batch_size=200, shuffle=True)

    C = embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)
    # C is (N, n, xoutdim)

    netscores = []
    true_labels = []
    for n, (q, l) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
        q = q.to(next(model.parameters()).device) 
        # q is bmd, c is Nnd, l is bN
        q = q + batch_get_white_noise(q, args.SNR)     # torch.randn_like(q) * noise
        q = embed_if_image_and_normalize(q, image_embed_model)
        q = embed_model(preembed_model(q))
        q = normalize(q)
        # q is (b, m, xoutdim)
        if aggregator is None:
            qct = torch.einsum("bmd,Nnd->bNmn", q, C)
            if isinstance(model, LamModel):
                model_inputs = stagger_and_concat(qct, num_stagger=stagger) # bNsmn  s = num_stagger+1

                lambdas = torch.stack([model(x) for x in model_inputs])
                # bNm1
            else:
                lambdas = []
                for xx in range(len(q)):
                    q_ = q[xx].unsqueeze(0)
                    q_ = q_.repeat_interleave(C.shape[0], dim=0)
                    lambdas_ = model(torch.cat((q_, C), dim=1).flatten(start_dim=1)).unsqueeze(-1)
                    lambdas.append(lambdas_)
                lambdas = torch.stack(lambdas)

            F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))

            P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)

            RmPC = Rm_mat @ P @ C.squeeze(-1)

            lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
            normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

            allscores = torch.stack([lamscore, normscore], dim=2)
            netscore = 2*scoremodel(-allscores).squeeze()      # b

        else:
            q = aggregator(q)
            # q is bd, C is Nd, we want bN scores
            # b1d - 1Nd = bNd --> sum across last dim to get bN scores
            netscore = 2 * F.sigmoid(-F.relu(q.unsqueeze(1) - C.unsqueeze(0)).sum(dim=-1))    # bN

        netscores.append(netscore.to('cpu'))
        true_labels.append(l.to('cpu'))

    netscores = torch.vstack(netscores)
    true_labels = torch.vstack(true_labels).squeeze() # 

    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    mRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    mAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    mAP /= (torch.arange(ranked_output.shape[1]) + 1)
    mAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    mAP = mAP.sum(dim=1).mean().item()

    return mAP, mRR


def load_models(name="best", expt_root=None, device="cpu"):
    if expt_root is None:
        raise ValueError("Please provide a valid experiment root folder in the inference script")

    if name not in ["best", "last", "latest"]:
        raise NotImplementedError("Only `best`, `last`, and `latest` models are supported")
    
    if name=="best":
        suffix = ""
    else:
        suffix = f"_{name}"
    
    print(f"Loading `{name}` model")

    model = torch.load(f"{expt_root}/model{suffix}.pt", map_location=device, weights_only=False)
    scoremodel = torch.load(f"{expt_root}/scmodel{suffix}.pt", map_location=device, weights_only=False)
    embed_model = torch.load(f"{expt_root}/embed_model{suffix}.pt", map_location=device, weights_only=False)
    if os.path.exists(f"{expt_root}/preembed_model{suffix}.pt"):
        preembed_model = torch.load(f"{expt_root}/preembed_model{suffix}.pt", map_location=device, weights_only=False)
    else:
        tokenize_transform = lambda x: tokenize(x, args)[0]
        preembed_model = TransformInput(tokenize_transform).to(device)
    
    if os.path.exists(f"{expt_root}/aggregator{suffix}.pt"):
        aggregator = torch.load(f"{expt_root}/aggregator{suffix}.pt", map_location=device, weights_only=False)
    else:
        aggregator = None

    return model, scoremodel, embed_model, preembed_model, aggregator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_id", type=str, help="Experiment ID to load models from")
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset, stored in `final_data/`")
    parser.add_argument("--device", type=str, default=-1, help="gpu id to run on,  pass -1 to run on cpu")
    parser.add_argument("--old", action="store_true", help="use old model checkpoint folder")
    args = parser.parse_args()

    experiment_id = args.expt_id
    folder = "models" if not args.old else "models_old"
    expt_root = f"{folder}/{experiment_id}/"

    dataset = args.dataset
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu"
    
    import pickle
    with open(os.path.join(expt_root, "args.pkl"), "rb") as file:
        args = pickle.load(file)

    model, scoremodel, embed_model, preembed_model, aggregator = load_models(name="best", expt_root=expt_root, device=DEVICE)
    model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
    if aggregator is not None:
        aggregator.eval()

    nwt, lamwt = next(scoremodel.parameters()).exp().detach().cpu()
    nwt, lamwt = nwt.item(), lamwt.item()

    print(f"normscore weight: {nwt:.4f}, lamscore weight: {lamwt:.4f}")
    
    DATA_ROOT = f"final_data/{dataset}"
    TEST_FILE = f"{DATA_ROOT}/dataset_test.hdf5"

    image_embed_model = None
    if "cifar" in dataset:
        image_embed_model_ckpt = "data/image_sequence/embedding_models/cifar_ae.pkl"
        image_embed_model = Autoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

        image_embed_model = image_embed_model.to(DEVICE)
        TEST_FILE = f"{DATA_ROOT}/dataset_test_orig.hdf5"
    
    if "lsun" in dataset:
        image_embed_model_ckpt = "data/image_sequence/embedding_models/lsun_ae.pkl"
        image_embed_model = LSUNAutoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

        image_embed_model = image_embed_model.to(DEVICE)
        TEST_FILE = f"{DATA_ROOT}/dataset_test_orig.hdf5"

    test_dataset = PairDatasetTest(TEST_FILE)
    print(f"test subset of dataset: \"{dataset}\" loaded from {TEST_FILE}")

    ##### HPARAMS ######
    b = args.b          # hinge margin for negative gap penalty (b-Apa)
    b1 = args.b1         # hinge margin for positive gap penalty (Apa-b)
    delta = args.delta     # hinge margin for contrastive loss
    lr = args.lr
    nepochs = args.nepochs
    stagger = args.stagger
    lamwt = args.lamwt    # loss coefficient for negative gap penalty
    gapwt = args.gapwt       # loss coefficient for positive gap penalty
    M = 6
    N = 20
    ####################

    A_mat, a_vec, Rm_mat = get_opas_constants(M, N, DEVICE)

    CFG = AttributeDict({
        'tau': 1,
        'n_sink_iter': 20,
        'n_samples': 1,
    })

    MAP, MRR = compute_metrics(test_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)

    print(f"Test metrics on dataset \"{dataset}\" with model loaded from experiment {experiment_id} -- MAP: {MAP:.4f}, MRR: {MRR:.4f}")