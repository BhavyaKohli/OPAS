from main import *
from loguru import logger
from time import perf_counter
from opas.models.sortlrl import SortLRL


def get_image_embed_model(dataset, TRAIN_FILE, VAL_FILE, TEST_FILE):
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
    
    elif "lsun" in dataset:
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

    return image_embed_model, TRAIN_FILE, VAL_FILE, TEST_FILE


def load_models(name="best", expt_root=None, device="cpu", args=None):
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


def save_models(embed_model, hasher, expt_root):
    torch.save(embed_model.to('cpu'), f"{expt_root}/embed_model.pt")
    torch.save(hasher.to('cpu'), f"{expt_root}/hasher.pt")


@torch.no_grad()
def compute_metrics(dataset, embed_model, preembed_model, image_embed_model, qhasher, chasher):
    device = next(qhasher.parameters()).device
    C = embed_full_corpus(dataset, embed_model, preembed_model, image_embed_model, verbose=True)
    C = chasher(C)

    embed_input = lambda x: embed_model(preembed_model(embed_if_image_and_normalize(x, image_embed_model)))

    Q, true_labels = [], []
    loader = dataset.get_dataloader(batch_size=150, shuffle=False)
    for (q, l) in tqdm(loader, desc="Validation...", leave=False):
        q = q.to(device)
        q = embed_input(q)
        q = qhasher(q)                
        
        Q.append(q)  
        true_labels.append(l.to('cpu'))

    Q = torch.vstack(Q)
    true_labels = torch.vstack(true_labels)
    netscores = F.cosine_similarity(Q.unsqueeze(1), C.unsqueeze(0), dim=-1).to('cpu')

    ranking = netscores.argsort(dim=1, descending=True)
    ranked_output = torch.gather(true_labels, dim=1, index=ranking)

    MRR = (1 / (ranked_output.argmax(dim=1) + 1)).mean().item()

    MAP = (torch.cumsum(ranked_output, dim=1) * ranked_output).float()
    MAP /= (torch.arange(ranked_output.shape[1]) + 1)
    MAP /= torch.sum(ranked_output, dim=1, keepdim=True)
    MAP = MAP.sum(dim=1).mean().item()

    return MAP, MRR


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    if any([cli_args.expt_id is None, cli_args.device is None, cli_args.hash_latent is None]):
        print(cli_args)
        raise ValueError("Please provide a valid experiment ID (expt_id), device id (device), latent dimension for SortLRL (hash_latent)")

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
    args = argparse.Namespace(**OmegaConf.merge(args, cli_args, base_conf))

    DEBUG = getattr(args, "debug", False)

    os.makedirs("xhashing", exist_ok=True)
    ls = [int(l.split('_')[-1]) for l in os.listdir("xhashing") if experiment_id in l and os.path.isdir(f"xhashing/{l}")]
    if len(ls) == 0: ls = [-1]
    hasher_expt_root = f"xhashing/{experiment_id}_{max(ls)+1}"
    logger.remove(0)
    if not DEBUG:
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
    DATA_ROOT = f"final_data/{dataset}"
    TRAIN_FILE = f"{DATA_ROOT}/dataset_train.hdf5"
    VAL_FILE = f"{DATA_ROOT}/dataset_val.hdf5"
    TEST_FILE = f"{DATA_ROOT}/dataset_test.hdf5"

    image_embed_model, TRAIN_FILE, VAL_FILE, TEST_FILE = get_image_embed_model(dataset, TRAIN_FILE, VAL_FILE, TEST_FILE)
    
    neg_expl = getattr(args, "neg_expl", 800)
    args.neg_expl = neg_expl
    train_dataset = PairDatasetTrain(TRAIN_FILE, num_q=args.num_q, negative_exploration=neg_expl)
    val_dataset = PairDatasetTest(VAL_FILE)
    test_dataset = PairDatasetTest(TEST_FILE)

    trainloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)


    ######### HPARAMS #########
    positive_samples = 10
    M = train_dataset.q[0].shape[0]
    N = train_dataset.c[0].shape[0]

    nepochs = args.nepochs
    delta = args.delta
    ###########################

    tokenize_transform = lambda x: tokenize(x, args)[0]
    if isinstance(image_embed_model, Autoencoder) or isinstance(image_embed_model, LSUNAutoencoder):
        d_model = 384
    else:
        d_model = train_dataset.q[0].shape[-1]

    preembed_model = TransformInput(tokenize_transform).to(DEVICE)
    embed_model = nn.Sequential(
        PositionalEncoding(d_model=d_model, max_seq_length=500),
        nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=args.xnhead, batch_first=True, dim_feedforward=args.xff), args.xnumlayers),
        nn.Linear(d_model, args.xoutdim)
    ).to(DEVICE)
    embed_optimizer = torch.optim.AdamW(embed_model.parameters(), lr=args.xlr)
    scheduler = get_linear_schedule_with_warmup(embed_optimizer, args.xwarmupepochs * len(trainloader), nepochs * len(trainloader))

    embed_input = lambda x, embed_model: embed_model(preembed_model(embed_if_image_and_normalize(x, image_embed_model)))

    hasher_type = getattr(args, "hasher_type", "SortLRL")
    args.hasher_type = hasher_type
    if hasher_type == "SortLRL":
        hash_outdim = getattr(args, "hash_outdim", args.xoutdim)
        qhasher = SortLRL(indim=args.xoutdim, seq_len=M, latent=args.hash_latent, outdim=hash_outdim).to(DEVICE)
        chasher = SortLRL(indim=args.xoutdim, seq_len=N, latent=args.hash_latent, outdim=hash_outdim).to(DEVICE)
        args.hash_outdim = hash_outdim
    elif hasher_type == "DeepSet":
        qhasher = DeepSetModel(indim=args.xoutdim, latent=args.hash_latent, outdim=getattr(args, "hash_outdim", args.xoutdim)).to(DEVICE)
        if getattr(args, "share_hasher", False):
            chasher = DeepSetModel(indim=args.xoutdim, latent=args.hash_latent, outdim=getattr(args, "hash_outdim", args.xoutdim)).to(DEVICE)
        else:
            chasher = qhasher

    hashers = nn.ModuleList([qhasher, chasher])
    optimizer = torch.optim.Adam(hashers.parameters(), lr=args.lr, weight_decay=1e-4)
    
    pbar = tqdm(range(1,nepochs+1,1), disable=False)
    best_val_map = 0
    es = 0
    for epoch in pbar:
        inner_pbar = tqdm(trainloader, disable=False, leave=False)

        embed_model.train()
        hashers.train()
        losses = []
        for n, (q, c, l) in enumerate(inner_pbar):
            qpos, cpos, lpos = train_dataset.get_positive_samples(positive_samples)
            
            q = torch.cat((q, qpos), dim=0).to(DEVICE)
            c = torch.cat((c, cpos), dim=0).to(DEVICE)
            l = torch.cat((l, lpos), dim=0).float().to(DEVICE)
            # q: (batch_size, M, indim)
            # c: (batch_size, N, indim)
            # l: (batch_size)

            q = q + batch_get_white_noise(q, args.SNR)
            q, c = embed_input(q, embed_model), embed_input(c, embed_model)
            q, c = qhasher(q), chasher(c)
            # q: (batch_size, outdim)
            # c: (batch_size, outdim)
            
            netscore = F.cosine_similarity(q, c, dim=-1)

            pos_score = netscore[torch.where(l==1)]
            neg_score = netscore[torch.where(l==0)]

            neg_minus_pos = (neg_score.unsqueeze(0) - pos_score.unsqueeze(1)).reshape(-1)   # dim(pos_score) * dim(neg_score)
            loss = F.relu(delta + neg_minus_pos).mean() 
            
            optimizer.zero_grad()
            embed_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            embed_optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            inner_pbar.set_postfix_str(f"Loss: {np.mean(losses[-50:]):.4f}")

        hashers.eval()
        embed_model.eval()
        val_map, val_mrr = compute_metrics(val_dataset, embed_model, preembed_model, image_embed_model, qhasher, chasher)

        if val_map >= best_val_map + 1e-8:
            best_val_map = val_map
            es = 0
            if not DEBUG:
                torch.save(hashers, f"{hasher_expt_root}/hasher_best.pt")
            bestwts = hashers.state_dict()
            bestwts_embed = embed_model.state_dict()
        else:
            es += 1
            if es > 50:
                print(f"Early stopping at epoch {epoch}")
                break
        logstr = f"Epoch: {epoch}, Loss: {np.mean(losses):.4f}, Val MAP: {val_map:.4f}, Val MRR: {val_mrr:.4f}, Best Val MAP: {best_val_map:.4f}"
        pbar.set_postfix_str(f"ES: {es:2d}, {logstr}")

        if not DEBUG:
            logger.info(logstr)

    hashers.load_state_dict(bestwts)
    hashers.eval()
    embed_model.load_state_dict(bestwts_embed)
    embed_model.eval()
    
    test_map, test_mrr = compute_metrics(test_dataset, embed_model, preembed_model, image_embed_model, qhasher, chasher)
    logstr = f"Test MAP: {test_map:.4f}, Test MRR: {test_mrr:.4f}"
    print(logstr)
    logger.info(logstr)    