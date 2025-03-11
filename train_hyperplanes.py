from train_prehash import *
from torch.utils.data import DataLoader, TensorDataset


class DummyDataset:
    def __init__(self, arr):
        self.c = arr


@torch.no_grad()
def get_all_pair_scores(q_emb, c_emb, **kwargs):
    model, scoremodel, embed_model, preembed_model, hasher = kwargs['models']
    device = next(model.parameters()).device
    args = kwargs['args']

    A_mat, a_vec, Rm_mat = get_opas_constants(M=q_emb[0].shape[0], N=c_emb[0].shape[0], device=device)
    CFG = AttributeDict({
        'tau': 1,
        'n_sink_iter': getattr(args, "n_sink_iter", 20),
        'n_samples': 1,
    })
    lamwt = getattr(args, "lamwt", 1e-2)
    b = getattr(args, "b", 1)

    qcscores = []
    _batch_size = 150
    _inner_loader = range(0, len(q_emb), _batch_size)
    for i, idx in enumerate(tqdm(_inner_loader, desc="Computing scores", leave=False)):
        q = q_emb[idx:idx+_batch_size].to(device) 
        qct = torch.einsum("bmd,Nnd->bNmn", q, c_emb)     # verified
        model_inputs = stagger_and_concat(qct, num_stagger=0)   # !! hardcoded 0 stagger !!
        model_inputs = model_inputs.squeeze(1)                  # !! hardcoded 0 stagger !!
        lambdas = torch.stack([model(x) for x in model_inputs])
        
        F_mat = Rm_mat.T @ (2*qct + (a_vec @ lambdas.transpose(2,3) @ A_mat).transpose(2,3))

        P = gumbel_sinkhorn(F_mat, CFG.tau, CFG.n_sink_iter, noise=False)
        RmPC = Rm_mat @ P @ c_emb.squeeze(-1)
        lamscore = lamwt * (lambdas.transpose(2,3) @ F.relu(b-A_mat @ Rm_mat @ P @ a_vec)).squeeze()
        normscore = torch.norm(q.unsqueeze(1) - RmPC, dim=[-1,-2])

        allscores = torch.stack([lamscore, normscore], dim=2)
        netscore = 2*scoremodel(-allscores).squeeze()      # b
        qcscores.append(netscore.to('cpu'))
        
        del qct, P, F_mat, RmPC, q
    return torch.vstack(qcscores).cpu()


class PairDatasetTrainHPlane(PairDatasetTrain):
    def __init__(self, filepath, models, args, num_q=300, negative_exploration=800, seed=15):
        # models: (model, scoremodel, embed_model, preembed_model, hasher)
        super().__init__(filepath, num_q, negative_exploration, seed)
        with torch.no_grad():
            self.q_emb = embed_full_corpus(DummyDataset(self.q), embed_model, preembed_model)
            self.c_emb = embed_full_corpus(DummyDataset(self.c), embed_model, preembed_model)
            self.qcscores = get_all_pair_scores(self.q_emb, self.c_emb, models=models, args=args)
            self.q = hasher[0](self.q_emb).cpu()
            self.c = hasher[1](self.c_emb).cpu()

    def __getitem__(self, idx):
        qloc, cloc = self.get_pair_ids(idx)
        cloc = self.exploration[qloc][cloc]
        q, c = self.q[qloc], self.c[cloc]
        l = self.l[qloc]

        label = 1 if cloc in l else 0
        return q, c, self.qcscores[qloc][cloc]

    def get_positive_samples(self, num_samples=2):
        if num_samples == 1:
            raise ValueError("num_samples must be greater than 1")
        positive_samples = np.random.randint(len(self.q), size=num_samples)
        q, l = self.q[positive_samples], self.l[positive_samples]
        # l has 16 indices, need to pick one
        l = [np.random.choice(l_, p=[0.1] + [0.9/(len(l_)-1)]*(len(l_)-1)) for l_ in l]
        c = torch.stack([self.c[l_] for l_ in l])
        label = torch.ones(num_samples)

        scores = torch.tensor([self.qcscores[qloc][cloc] for qloc, cloc in zip(positive_samples, l)])
        return q, c, scores


class PairDatasetTestHPlane(PairDatasetTrainHPlane):
    def __init__(self, filepath, models, args, negative_exploration, seed=15, num_q=-1):
        super().__init__(filepath, models, args, num_q=num_q, negative_exploration=negative_exploration, seed=seed)
        

def get_loss(qproj, cproj, sc, l1=1e-3, l2=1e-1, l3=1e-6):
    loss1 = torch.norm(cproj.abs() - 1, p=1, dim=-1).sum(-1)    # fence sitting, sum over batch
    loss1 = loss1.mean()    # mean over planes
    loss2 = cproj.sum(dim=1).abs().sum(dim=-1)                  # bit balance, sum over bits
    loss2 = loss2.mean()    # mean over planes
    
    dots = torch.einsum("wbd,wbd->wb", qproj, cproj)            # (nplanes, batch_size), verified
    ktop = torch.topk(sc, k=len(sc)).indices
    pos_dot = dots.transpose(0,1)[ktop[:25]].transpose(0,1)
    neg_dot = dots.transpose(0,1)[ktop[25:]].transpose(0,1)
    neg_minus_pos = neg_dot.unsqueeze(-2) - pos_dot.unsqueeze(-1)
    loss3 = F.relu(1 + neg_minus_pos).sum([1,2])              # collision minimizer, sum over pos, neg
    loss3 = loss3.mean()    # mean over planes

    loss = l1 * loss1 + l2 * loss2 + l3 * loss3
    return loss, loss1, loss2, loss3


@torch.no_grad()
def get_index_spread(W, global_corpus):
    index = 0.5 * (torch.einsum("nmd,bd->nbm", W, global_corpus.to(W.device)).sign() + 1)
    index_spread = [len(torch.unique(index[i], dim=0)) for i in range(len(index))]
    mu, std = np.mean(index_spread), np.std(index_spread)
    index_spread = f"{mu:.2f}±{std:.2f}"
    return index_spread, mu, std


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()

    experiment_id = cli_args.expt_id
    folder = "models" if not getattr(cli_args, "old", None) else "models_old"
    expt_root = f"{folder}/{experiment_id}/"
    hasher_expt_root = f"hashing/{experiment_id}/"
    
    DEBUG = getattr(cli_args, "debug", False)
    if not DEBUG:
        logger.add(f"{hasher_expt_root}/hyperplanes.log", level="INFO", format="{time:D-MM-YYYY HH:mm:ss} | {level} | {message}")
    else:
        logger.add(sys.stdout, level="INFO", format="{time:D-MM-YYYY HH:mm:ss} | {level} | {message}")
    
    DEVICE = f"cuda:{cli_args.device}" if torch.cuda.is_available() and cli_args.device != -1 else "cpu"

    import pickle
    with open(os.path.join(hasher_expt_root, "args.pkl"), "rb") as file:
        args = pickle.load(file)

    args = OmegaConf.create(vars(args))
    base_conf = OmegaConf.load(f"configs/hyperplane_base.yaml")
    hplane_args = OmegaConf.merge(base_conf, cli_args)
    print(f"Hyperplane args: {hplane_args}")

    args = argparse.Namespace(**OmegaConf.merge(args, hplane_args))

    seed_everything(args.seed)

    model, scoremodel, embed_model, preembed_model, aggregator = load_models(name="best", expt_root=expt_root, device=DEVICE, args=args)
    model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
    if aggregator is not None:
        aggregator.eval()
    hasher = torch.load(os.path.join(hasher_expt_root, "hasher_best.pt"), map_location=DEVICE)
    hasher.eval()
    print("Models loaded")

    for m in [model, scoremodel, embed_model, preembed_model, hasher]:
        for param in m.parameters():
            param.requires_grad = False

    dataset = args.dataset
    DATA_ROOT = f"final_data/{dataset}"
    TRAIN_FILE = f"{DATA_ROOT}/dataset_train.hdf5"
    VAL_FILE = f"{DATA_ROOT}/dataset_val.hdf5"
    TEST_FILE = f"{DATA_ROOT}/dataset_test.hdf5"

    image_embed_model = None
    name_replace = lambda x: x.replace(".hdf5", "_orig.hdf5")
    if "cifar" in dataset:
        image_embed_model_ckpt = "data/image_sequence/embedding_models/cifar_ae.pkl"
        image_embed_model = Autoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

        image_embed_model = image_embed_model.to(DEVICE)
        TRAIN_FILE = name_replace(TRAIN_FILE)
        VAL_FILE = name_replace(VAL_FILE)
        TEST_FILE = name_replace(TEST_FILE)

    if "lsun" in dataset:
        image_embed_model_ckpt = "data/image_sequence/embedding_models/lsun_ae.pkl"
        image_embed_model = LSUNAutoencoder()
        image_embed_model.load_state_dict(torch.load(image_embed_model_ckpt))
        image_embed_model.eval()

        for param in image_embed_model.parameters():
            param.requires_grad = False

        image_embed_model = image_embed_model.to(DEVICE)
        TRAIN_FILE = name_replace(TRAIN_FILE)
        VAL_FILE = name_replace(VAL_FILE)
        TEST_FILE = name_replace(TEST_FILE)

    W = nn.Parameter(torch.randn(args.nplanes, args.nbits, hasher[0].lrl[2].out_features, device=DEVICE), requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.9, patience=5)   # will track index spread

    models = [model, scoremodel, embed_model, preembed_model, hasher]

    st = perf_counter()
    train_dataset = PairDatasetTrainHPlane(TRAIN_FILE, models=models, args=args, num_q=args.num_q, negative_exploration=args.neg_expl)
    val_dataset = PairDatasetTestHPlane(VAL_FILE, models=models, args=args, negative_exploration=args.neg_expl, num_q=100)
    test_dataset = PairDatasetTestHPlane(TEST_FILE, models=models, args=args, negative_exploration=args.neg_expl, num_q=100)
    print(f"Pre-embedded corpus and queries in {perf_counter() - st:.3f}s")

    trainloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    global_corpus = torch.cat((train_dataset.c, val_dataset.c, test_dataset.c), axis=0)
    print(f"Global corpus shape: {global_corpus.shape}")

    l1, l2, l3 = getattr(args, "l1", 1e-3), getattr(args, "l2", 1e-1), getattr(args, "l3", 1e-3)
    hplanes_id = f"{args.nbits}_{datetime.now():%H%M}"
    logger.info(f"Hyperplane file: hyperplanes_{hplanes_id}.pkl, Loss Weights: {l1=}, {l2=}, {l3=}, Args: {args}")

    pbar = tqdm(range(1,args.nepochs+1,1), disable=False)
    bestmu = 0
    es = 0

    for epoch in pbar:
        inner_pbar = tqdm(trainloader, disable=False, leave=False)

        train_loss = []
        for i, (q, c, sc) in enumerate(inner_pbar):
            qpos, cpos, scpos = train_dataset.get_positive_samples(10)

            q = torch.cat((q, qpos), dim=0).to(DEVICE)
            c = torch.cat((c, cpos), dim=0).to(DEVICE)
            sc = torch.cat((sc, scpos), dim=0).float().to(DEVICE)

            qproj = torch.einsum("nmd,bd->nbm", W, q).tanh()   # (nplanes, batch_size, nbits)
            cproj = torch.einsum("nmd,bd->nbm", W, c).tanh()   # (nplanes, batch_size, nbits)

            loss = get_loss(qproj, cproj, sc, l1, l2, l3)[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if i % 100 == 0:
                index_spread, mu, std = get_index_spread(W, global_corpus)                
                scheduler.step(mu)
            
            inner_pbar.set_postfix_str(f"Loss: {loss.item():.4f}, Index Spread: {index_spread}")
        
        index_spread, mu, std = get_index_spread(W, global_corpus)

        if mu > bestmu:
            bestmu = mu
            es = 0
            torch.save(W, f"{hasher_expt_root}/hyperplanes_{hplanes_id}.pkl")
        else:
            es += 1
            if es == 100:
                break
        
        pbar.set_postfix_str(f"ES: {es:2d}, Index Spread: {mu:4f}, Best: {bestmu:.4f}")

    # saving planes in numpy format for using in lshash3
    W = torch.load(f"{hasher_expt_root}/hyperplanes_{hplanes_id}.pkl").cpu().detach().numpy()
    np.savez_compressed(f"{hasher_expt_root}/weights.npz", *W)