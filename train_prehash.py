import wandb

from main import *
from loguru import logger
from time import perf_counter
from opas.models.sortlrl import SortLRL
from opas.models.model_utils import load_models, get_image_embed_model

from opas.data import PairDatasetTrainHPlane, PairDatasetTestHPlane


def batch_fwd_q(image_embed_model, preembed_model, embed_model, q):
    q = embed_if_image_and_normalize(q, image_embed_model)
    q = embed_model(preembed_model(q))
    q = normalize(q)
    return q
    

@torch.no_grad()
def validation(loader, qhasher, chasher, criterion, batch_fwd_q):
    losses = []
    for q, c, l in loader:
        q = q.to(DEVICE)
        c = c.to(DEVICE)
        l = l.float().to(DEVICE)

        q = batch_fwd_q(q)
        q, c = qhasher(q), chasher(c)
        loss = criterion(q, c, l)
        losses.append(loss.item())
        
    loss = np.mean(losses)
    return loss


@torch.no_grad()
def validation_map(dataset, qhasher, chasher):
    device = next(qhasher.parameters()).device
    C = chasher(dataset.c.to(device))
    Q = qhasher(dataset.q.to(device))
    true_labels = dataset.lonehot.cpu()

    netscores = ((Q @ C.T) / (torch.norm(Q, dim=-1, keepdim=True) * torch.norm(C, dim=-1, keepdim=True).T)).sigmoid().cpu()
    # netscores = []
    # batch_size = 150
    # import ipdb; ipdb.set_trace()
    # for i in range(0, len(Q), batch_size):
    #     netscores.append(F.cosine_similarity(Q.unsqueeze(1), C.unsqueeze(0), dim=-1).sigmoid().to('cpu'))
    # import ipdb; ipdb.set_trace()
    # netscores = dataset.qcscores
    # netscores = torch.vstack(netscores)

    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    MRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    MAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    MAP /= (torch.arange(ranked_output.shape[1]) + 1)
    MAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    MAP = MAP.sum(dim=1).mean().item()

    return MAP, MRR


class SortNoLRL(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, indim))
        self.outdim = outdim

    def forward(self, x):
        # x: (batch_size, seq_len, indim)
        # output: (batch_size, outdim)
        proj = (x @ self.alpha.T).squeeze(-1)
        proj = torch.sort(proj, dim=-1)
        return F.pad(proj.values, (0, self.outdim - proj.values.shape[-1]), value=0)
    

class SortL(nn.Module):
    def __init__(self, indim, seq_len, outdim):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, indim))
        self.lin = nn.Linear(seq_len, outdim)

    def forward(self, x):
        # x: (batch_size, seq_len, indim)
        # output: (batch_size, outdim)
        proj = (x @ self.alpha.T).squeeze(-1)
        proj = torch.sort(proj, dim=-1)
        return self.lin(proj.values)
    

def save_dataset_to_file(dataset, filepath):
    q, c, l, lonehot, qcscores = dataset.q, dataset.c, dataset.l, dataset.lonehot, dataset.qcscores
    torch.save({"q": q, "c": c, "l": l, "lonehot": lonehot, "qcscores": qcscores}, filepath)


class LoadedDset(PairDatasetTest):
    def __init__(self, filepath):
        data = torch.load(filepath, map_location="cpu")
        self.q = data["q"]
        self.c = data["c"]
        self.l = data["l"]
        self.lonehot = data["lonehot"]
        self.qcscores = data["qcscores"]
    
    def __getitem__(self, idx):
        return self.q[idx], self.lonehot[idx], self.qcscores[idx]

    def __len__(self):
        return len(self.q)


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    base_conf = OmegaConf.load("configs/hash_base.yaml")

    experiment_id = cli_args.expt_id
    folder = "models" if not getattr(cli_args, "old", None) else "models_old"
    expt_root = f"{folder}/{experiment_id}/"

    DEVICE = f"cuda:{cli_args.device}" if torch.cuda.is_available() and cli_args.device != -1 else "cpu"
    
    import pickle
    with open(os.path.join(expt_root, "args.pkl"), "rb") as file:
        args = pickle.load(file)

    # overrides
    args = OmegaConf.create(vars(args))
    args = argparse.Namespace(**OmegaConf.merge(args, base_conf, cli_args))
    DEBUG = args.debug

    logger.remove(0)
    if not DEBUG:
        expt_id_root = f"hashing/{experiment_id}/"
        os.makedirs(expt_id_root, exist_ok=True)
        ls = [i for i in os.listdir(expt_id_root) if os.path.isdir(os.path.join(expt_id_root, i))]

        
        hasher_expt_root = os.path.join(expt_id_root, f"{len(ls)}")
        print("Logging to", hasher_expt_root)
        os.makedirs(hasher_expt_root, exist_ok=True)
        logger.add(f"{hasher_expt_root}/training.log", level="INFO", format="{time:D-MM-YYYY HH:mm:ss} | {level} | {message}")

        with open(f"{hasher_expt_root}/args.pkl", "wb") as file:
            pickle.dump(args, file)
        
        model_save_path = f"{hasher_expt_root}/hasher.pt"
        print(f"Running with args: {args}")
    else:
        logger.add(sys.stdout, level="INFO", format="{time:D-MM-YYYY HH:mm:ss} | {level} | {message}")
    
    logger.info("-"*100)
    logger.info("python " + " ".join(sys.argv))
    logger.info(f"Running with args: {args}")
    
    dataset = args.dataset
    if dataset == "lsun384":
        dataset = "lsun"
    DATA_ROOT = f"hashing"
    TRAIN_FILE = f"{DATA_ROOT}/{dataset}_train_dset.pt"
    VAL_FILE = f"{DATA_ROOT}/{dataset}_val_dset.pt"
    TEST_FILE = f"{DATA_ROOT}/{dataset}_test_dset.pt"

    st = perf_counter()
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(VAL_FILE) or not os.path.exists(TEST_FILE):
        model, scoremodel, embed_model, preembed_model, aggregator = load_models(name="best", expt_root=expt_root, device=DEVICE, args=args)
        model.eval(), scoremodel.eval(), embed_model.eval(), preembed_model.eval()
        if aggregator is not None:
            aggregator.eval()

        for m in [model, scoremodel, embed_model, preembed_model]:
            for param in m.parameters():
                param.requires_grad = False

        image_embed_model, TRAIN_FILE, VAL_FILE, TEST_FILE = get_image_embed_model(dataset, [TRAIN_FILE, VAL_FILE, TEST_FILE], device=DEVICE)
    
        hasher = [nn.Identity(), nn.Identity()]
        models = [model, scoremodel, embed_model, preembed_model, hasher]

        train_dataset = PairDatasetTestHPlane(TRAIN_FILE, models=models, args=args) #, num_q=args.num_q, negative_exploration=args.neg_expl)
        val_dataset = PairDatasetTestHPlane(VAL_FILE, models=models, args=args)
        test_dataset = PairDatasetTestHPlane(TEST_FILE, models=models, args=args)

        save_dataset_to_file(train_dataset, TRAIN_FILE)
        save_dataset_to_file(val_dataset, VAL_FILE)
        save_dataset_to_file(test_dataset, TEST_FILE)
    
    else:
        logger.info("Loading datasets from files")
        train_dataset = LoadedDset(TRAIN_FILE)
        val_dataset = LoadedDset(VAL_FILE)
        test_dataset = LoadedDset(TEST_FILE)
        
    print(f"Datasets loaded in {perf_counter() - st:.3f}s")

    trainloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)

    positive_samples = getattr(args, "pos_samps", 10)
    M = train_dataset.q[0].shape[0]
    N = train_dataset.c[0].shape[0]

    torch.cuda.empty_cache()

    hasher_type = getattr(args, "hasher_type", "SortLRL")
    hash_outdim = getattr(args, "hash_outdim", max(M,N))
    try: hash_outdim = args.outdim
    except: pass
    hash_latent = getattr(args, "hash_latent", args.xff)
    args.hasher_type, args.hash_outdim, args.hash_latent = hasher_type, hash_outdim, hash_latent
    if hasher_type == "SortLRL":        # outdim, latent used for inner LRL model
        qhasher = SortLRL(indim=args.xoutdim, seq_len=M, latent=hash_latent, outdim=hash_outdim).to(DEVICE)
        chasher = SortLRL(indim=args.xoutdim, seq_len=N, latent=hash_latent, outdim=hash_outdim).to(DEVICE)

    elif hasher_type == "SortNoLRL":    # outdim used for zero-padding
        qhasher = SortNoLRL(indim=args.xoutdim, outdim=hash_outdim).to(DEVICE)
        chasher = SortNoLRL(indim=args.xoutdim, outdim=hash_outdim).to(DEVICE)
    
    elif hasher_type == "SortL":        # outdim used for single linear layer
        qhasher = SortL(indim=args.xoutdim, seq_len=M, outdim=hash_outdim).to(DEVICE)
        chasher = SortL(indim=args.xoutdim, seq_len=N, outdim=hash_outdim).to(DEVICE)    

    elif hasher_type == "DeepSet":      # single model can be used for both q and c (agnostic to seq_len)
        qhasher = DeepSetModel(indim=args.xoutdim, latent=hash_latent, outdim=hash_outdim).to(DEVICE)
        if getattr(args, "share_hasher", False):
            chasher = DeepSetModel(indim=args.xoutdim, latent=hash_latent, outdim=hash_outdim).to(DEVICE)
        else:
            chasher = qhasher
    hasher = nn.ModuleList([qhasher, chasher])  # grouped so we can use a single optimizer

    use_amsgrad = getattr(args, "amsgrad", False)
    args.use_amsgrad = use_amsgrad
    optimizer = torch.optim.AdamW(hasher.parameters(), lr=args.lr, amsgrad=use_amsgrad)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.95 if args.scheduler else 1, patience=20, min_lr=5e-5)
    # criterion = lambda q, c, l: F.cosine_embedding_loss(q, c, l, margin=args.hash_margin)
    # criterion = lambda q, c, l: F.cross_entropy(0.5 * (F.cosine_similarity(q, c) + 1), l)
    # criterion = lambda q, c, l: nn.BCELoss()(0.5 * (F.cosine_similarity(q, c) + 1), l)

    def LOSS_ON_SILVER(q, c, l, loss='bce'):
        cs = F.cosine_similarity(q, c).sigmoid() 
        # cs is now in [0,1], original scores "l" are in [0,1]
        if loss == "bce":
            return F.binary_cross_entropy(cs, l)
        if loss == "mse":
            return F.mse_loss(cs, l)

    loss_type = args.loss_type
    criterion = partial(LOSS_ON_SILVER, loss=loss_type)
        
    sample_scores = args.sample_scores
    total_exploration = args.total_exploration 
    pbar = tqdm(range(1,args.nepochs+1,1), disable=False)
    best_val_map = 0
    es = 0
    
    if args.wandb:
        wandb.init(
            project = "hashing", 
            config = {
                "dataset": dataset,
                "loss_type": loss_type,
                "hasher_type": hasher_type,
                "hash_latent": hash_latent,
                "hash_outdim": hash_outdim,
                "lr": args.lr,
                "sample_scores": sample_scores,
                "total_exploration": total_exploration,
                "amsgrad": use_amsgrad
            }   
        )


    for epoch in pbar:
        inner_pbar = tqdm(trainloader, disable=False, leave=False)

        hasher.train()
        losses = []
        for n, (q, l, gtl) in enumerate(inner_pbar):
            q = q.to(DEVICE)
            gtl = gtl.to(DEVICE)
            c = train_dataset.c.to(DEVICE)
            # q: (batch_size, M, indim)
            # c: (len(train_dataset), N, indim)
            # gtl: (batch_size, len(train_dataset))

            q, c = qhasher(q), chasher(c)
            # q: (batch_size, outdim)
            # c: (len(train_dataset), outdim)

            scores = F.cosine_similarity(q.unsqueeze(1), c.unsqueeze(0), dim=-1).sigmoid()
            # scores: (batch_size, len(train_dataset))

            if sample_scores:
                idxs = []
                for i in range(len(l)):
                    pos = torch.where(l[i] == 1)[0]
                    neg = torch.where(l[i] == 0)[0]
                    neg = neg[torch.randperm(len(neg))[:total_exploration - len(pos)]]
                    idxs.append(torch.cat((pos, neg)))
                idxs = torch.stack(idxs).to(DEVICE)
            
                scores = torch.gather(scores, 1, idxs)
                gtl = torch.gather(gtl, 1, idxs)
                
            loss = F.mse_loss(scores, gtl, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            losses.append(loss.item())
            inner_pbar.set_postfix_str(f"Loss: {np.mean(losses[-50:]):.4f}")

            del scores

        torch.cuda.empty_cache()

        hasher.eval()
        val_map, val_mrr = validation_map(val_dataset, qhasher, chasher)
        # if epoch == 10:
        #     import ipdb; ipdb.set_trace()
        #     logger.info(validation_map(train_dataset, qhasher, chasher))
        #     raise
        scheduler.step(val_map)

        if val_map >= best_val_map + 1e-8:
            best_val_map = val_map
            es = 0
            if not DEBUG:
                torch.save(hasher, f"{hasher_expt_root}/hasher_best.pt")
            bestwts = hasher.state_dict()
        else:
            es += 1
            if es > 100:
                print(f"Early stopping at epoch {epoch}")
                break
        logstr = f"Loss: {np.mean(losses):.4f}, Val MAP: {val_map:.4f}, Val MRR: {val_mrr:.4f}, Best Val MAP: {best_val_map:.4f}"
        pbar.set_postfix_str(f"ES: {es:2d}, {logstr}")

        if args.wandb:
            log = {
                "loss": np.mean(losses),
                "val_map": val_map,
                "val_mrr": val_mrr,
                "best_val_map": best_val_map,
                "lr": optimizer.param_groups[0]['lr']
            }
            wandb.log(log)

        if not DEBUG:
            logger.info(logstr)

    if not DEBUG:
        # final args update
        with open(f"{hasher_expt_root}/args.pkl", "wb") as file:
            pickle.dump(args, file)

    hasher.load_state_dict(bestwts)
    hasher.eval()
    test_map, test_mrr = validation_map(test_dataset, qhasher, chasher)
    logstr = f"Test MAP: {test_map:.4f}, Test MRR: {test_mrr:.4f}"
    print(logstr)
    logger.info(logstr)    