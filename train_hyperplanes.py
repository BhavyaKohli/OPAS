from train_prehash import *
from torch.utils.data import DataLoader, TensorDataset
from opas.data import DummyDataset, PairDatasetTrainHPlane, PairDatasetTestHPlane, PairDatasetTestHPlaneSampled
        

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
    hasher_expt_num = cli_args.hexpt_num
    hasher_expt_root = f"hashing/{experiment_id}_{hasher_expt_num}/"
    
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

    image_embed_model, TRAIN_FILE, VAL_FILE, TEST_FILE = get_image_embed_model(dataset, TRAIN_FILE, VAL_FILE, TEST_FILE)

    W = nn.Parameter(torch.randn(args.nplanes, args.nbits, hasher[0].lrl[2].out_features, device=DEVICE), requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.9, patience=5)   # will track index spread

    models = [model, scoremodel, embed_model, preembed_model, hasher]

    st = perf_counter()
    train_dataset = PairDatasetTrainHPlane(TRAIN_FILE, models=models, args=args, num_q=args.num_q, negative_exploration=args.neg_expl)
    val_dataset = PairDatasetTestHPlaneSampled(VAL_FILE, models=models, args=args, negative_exploration=args.neg_expl, num_q=300)
    test_dataset = PairDatasetTestHPlaneSampled(TEST_FILE, models=models, args=args, negative_exploration=args.neg_expl, num_q=100)
    print(f"Datasets loaded in {perf_counter() - st:.3f}s")

    trainloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    valloader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)
    global_corpus = torch.cat((train_dataset.c, val_dataset.c, test_dataset.c), axis=0)
    print(f"Global corpus shape: {global_corpus.shape}")

    l1, l2, l3 = getattr(args, "l1", 1e-3), getattr(args, "l2", 1e-1), getattr(args, "l3", 1e-3)
    hplanes_id = f"{args.nbits}_{datetime.now():%H%M}"
    logger.info(f"Hyperplane file: hyperplanes_{hplanes_id}.pkl, Loss Weights: {l1=}, {l2=}, {l3=}, Args: {args}")

    pbar = tqdm(range(1,args.nepochs+1,1), disable=False)
    bestmu = 0
    es = 0
    track_metric = getattr(args, "track_metric", "index_spread")

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

        if track_metric == "index_spread":
            if mu > bestmu:
                bestmu = mu
                es = 0
                if not DEBUG:
                    torch.save(W, f"{hasher_expt_root}/hyperplanes_{hplanes_id}.pkl")
            else:
                es += 1
                if es == 100:
                    break

        elif track_metric == "collision":
            inner_pbar = tqdm(valloader, leave=False, desc="Validation...")
            
            val_loss = []
            for i, (q, c, sc) in enumerate(inner_pbar):
                qpos, cpos, scpos = val_dataset.get_positive_samples(10)

                q = torch.cat((q, qpos), dim=0).to(DEVICE)
                c = torch.cat((c, cpos), dim=0).to(DEVICE)
                sc = torch.cat((sc, scpos), dim=0).float().to(DEVICE)

                qproj = torch.einsum("nmd,bd->nbm", W, q).tanh()   # (nplanes, batch_size, nbits)
                cproj = torch.einsum("nmd,bd->nbm", W, c).tanh()   # (nplanes, batch_size, nbits)

                loss = get_loss(qproj, cproj, sc, l1, l2, l3)[-1]
                val_loss.append(loss.item())            

            if -np.mean(val_loss) > bestmu:
                bestmu = np.mean(train_loss)
                es = 0
                if not DEBUG:
                    torch.save(W, f"{hasher_expt_root}/hyperplanes_{hplanes_id}.pkl")
            else:
                es += 1
                if es == 100:
                    break
            

        
        pbar.set_postfix_str(f"ES: {es:2d}, Index Spread: {mu:4f}, Best: {bestmu:.4f}")

    if not DEBUG:
        # saving planes in numpy format for using in lshash3
        W = torch.load(f"{hasher_expt_root}/hyperplanes_{hplanes_id}.pkl").cpu().detach().numpy()
        np.savez_compressed(f"{hasher_expt_root}/weights.npz", *W)