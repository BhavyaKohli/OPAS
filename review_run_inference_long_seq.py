import os

from review_main_long_seq import *
from opas.utils import get_opas_constants
from opas.models.model_utils import load_models, get_image_embed_model


@torch.no_grad()
def compute_metrics(dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=None, stagger=2, verbose=False, aggregator=None):

    loader = dataset.get_dataloader(batch_size=100, shuffle=True)

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

        inner_score, inner_labels = [], []
        for cmini in range(0,len(C),50):
            C_ = C[cmini:cmini+50]

            if aggregator is None:
                if True:
                    qct = torch.einsum("bmd,Nnd->bNmn", q, C_)   # verified
                    if args.use_linear_lammodel:
                        lambdas = []
                        for xx in range(len(q)):
                            q_ = q[xx].unsqueeze(0)
                            q_ = q_.repeat_interleave(C_.shape[0], dim=0)
                            lambdas_ = model(torch.cat((q_, C_), dim=1).flatten(start_dim=1)).unsqueeze(-1)
                            lambdas.append(lambdas_)
                        lambdas = torch.stack(lambdas)
                    else:
                        model_inputs = stagger_and_concat(qct, num_stagger=stagger)
                        if stagger==0: 
                            model_inputs = model_inputs.squeeze(1)
                        lambdas = torch.stack([model(x) for x in model_inputs])
                    
                    F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))
                    del qct
                else:
                    F_mat = Rm_mat.T @ (
                        torch.stack([-(q[i][None].unsqueeze(2) - C_.unsqueeze(1)).relu().sum(-1) for i in range(len(q))])
                    )
                    lambdas = torch.ones((len(q), len(C_), M, 1), device=F_mat.device)

                P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)

                RmPC = Rm_mat @ P @ C_.squeeze(-1)

                if args.no_lamrelu:
                    lamscore = lamwt * (lambdas.transpose(2,3) @ (b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
                else:
                    lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
                normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

                allscores = torch.stack([lamscore, normscore], dim=2)
                netscore = 2*scoremodel(-allscores).squeeze()      # b
                del P, F_mat, RmPC

            else:
                q = aggregator[0](q)
                # q is bd, C is Nd, we want bN scores
                # b1d - 1Nd = bNd --> sum across last dim to get bN scores
                netscore = 2 * F.sigmoid(-F.relu(q.unsqueeze(1) - C_.unsqueeze(0)).sum(dim=-1))    # bN
                # TODO: fix this
                if args.deepset_mode == "cosine":     # 3
                    netscore = 0.5 * (F.cosine_similarity(q.unsqueeze(1), C_.unsqueeze(0), dim=-1) + 1)

            inner_score.append(netscore.to('cpu'))
        netscore = torch.cat(inner_score, dim=-1)        
        del q

        netscores.append(netscore.to('cpu'))
        true_labels.append(l.to('cpu'))

    netscores = torch.vstack(netscores)
    true_labels = torch.vstack(true_labels).squeeze() # 

    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    MRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    MAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    MAP /= (torch.arange(ranked_output.shape[1]) + 1)
    MAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    MAP = MAP.sum(dim=1).mean().item()

    return MAP, MRR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_id", type=str, help="Experiment ID to load models from")
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset, stored in `final_data/`")
    parser.add_argument("--device", type=str, default=-1, help="gpu id to run on,  pass -1 to run on cpu")
    parser.add_argument("--old", action="store_true", help="use old model checkpoint folder")
    parser.add_argument("--fix_lambdas", default=None, help="fix lambdas to this value, do not use LamModel")
    args = parser.parse_args()

    experiment_id = args.expt_id
    folder = "models" if not args.old else "models_old"
    expt_root = f"{folder}/{experiment_id}/"
    fix_lambdas = args.fix_lambdas

    dataset = args.dataset
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu"
    
    import pickle
    with open(os.path.join(expt_root, "args.pkl"), "rb") as file:
        args = pickle.load(file)

    model, scoremodel, embed_model, preembed_model, aggregator = load_models(name="best", expt_root=expt_root, device=DEVICE, args=args)
    model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
    if aggregator is not None:
        aggregator.eval()

    nwt, lamwt = next(scoremodel.parameters()).exp().detach().cpu()
    nwt, lamwt = nwt.item(), lamwt.item()

    print(f"normscore weight: {nwt:.4f}, lamscore weight: {lamwt:.4f}")
    
    DATA_ROOT = f"final_data_rev/{dataset}"
    TEST_FILE = f"{DATA_ROOT}/dataset_test.hdf5"

    image_embed_model, TEST_FILE = get_image_embed_model(dataset, [TEST_FILE], device=DEVICE)

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
    M = test_dataset.q[0].shape[0]
    N = test_dataset.c[0].shape[0]
    ####################
    if fix_lambdas is not None:
        fix_lambdas = float(fix_lambdas)
        model = DummyLamModel(M,value=fix_lambdas)
        model.to(DEVICE)
        model.eval()
        print(f"Using fixed lambdas: {fix_lambdas}")

    A_mat, a_vec, Rm_mat = get_opas_constants(M, N, DEVICE)
    a_vec = torch.arange(N).reshape(a_vec.shape).float().to(DEVICE) # for long sequences, cant use exponentiation

    CFG = AttributeDict({
        'tau': 1,
        'n_sink_iter': 20,
        'n_samples': 1,
    })

    MAP, MRR = compute_metrics(test_dataset, model, scoremodel, embed_model, preembed_model, image_embed_model=image_embed_model, stagger=stagger, verbose=True, aggregator=aggregator)

    print(f"Test metrics on dataset \"{dataset}\" with model loaded from experiment {experiment_id} -- MAP,MRR: {MAP:.4f},{MRR:.4f}")
