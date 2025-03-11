from train_prehash import *


def get_loss(qproj, cproj, l1=1e-3, l2=1e-1, l3=1e-6):
    loss1 = torch.norm(cproj.abs() - 1, p=1, dim=-1).sum(-1)    # fence sitting, sum over batch
    loss1 = loss1.mean()    # mean over planes
    loss2 = cproj.sum(dim=1).abs().sum(dim=-1)                  # bit balance, sum over bits
    loss2 = loss2.mean()    # mean over planes
            
    all_all_dot = torch.bmm(qproj, cproj.transpose(1,2))
    pos_dot = all_all_dot.transpose(0,2)[l==1].transpose(0,2)
    neg_dot = all_all_dot.transpose(0,2)[l==0].transpose(0,2)
    neg_minus_pos = neg_dot.unsqueeze(-2) - pos_dot.unsqueeze(-1)

    loss3 = F.relu(1 + neg_minus_pos).sum([1,2,3])              # collision minimizer, sum over pos, neg, batch
    loss3 = loss3.mean()    # mean over planes

    loss = l1 * loss1 + l2 * loss2 + l3 * loss3
    return loss, loss1, loss2, loss3


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()

    experiment_id = cli_args.expt_id
    folder = "models" if not getattr(cli_args, "old", None) else "models_old"
    expt_root = f"{folder}/{experiment_id}/"
    hasher_expt_root = f"hashing/{experiment_id}/"

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

    train_dataset = PairDatasetTrain(TRAIN_FILE, num_q=args.num_q, negative_exploration=args.neg_expl)
    val_dataset = PairDatasetTest(VAL_FILE) #, num_q=args.num_q, negative_exploration=args.neg_expl)

    class DummyDataset:
        def __init__(self, arr):
            self.c = arr

    st = perf_counter()
    with torch.no_grad():
        train_dataset.c = hasher[1](embed_full_corpus(train_dataset, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)).cpu().numpy()
        train_dataset.q = hasher[0](embed_full_corpus(DummyDataset(train_dataset.q), embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)).cpu().numpy()
        val_dataset.c = hasher[1](embed_full_corpus(val_dataset, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)).cpu().numpy()
        val_dataset.q = hasher[0](embed_full_corpus(DummyDataset(val_dataset.q), embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)).cpu().numpy()
    print(f"Pre-embedded corpus and queries in {perf_counter() - st:.3f}s")

    trainloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    valloader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    test_dataset = PairDatasetTest(TEST_FILE)
    with torch.no_grad():
        test_dataset.c = hasher[1](embed_full_corpus(test_dataset, embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)).cpu().numpy()
        test_dataset.q = hasher[0](embed_full_corpus(DummyDataset(test_dataset.q), embed_model, preembed_model, image_embed_model=image_embed_model, aggregator=aggregator)).cpu().numpy()

    global_corpus = torch.from_numpy(np.concatenate((train_dataset.c, val_dataset.c, test_dataset.c), axis=0))
    print(f"Global corpus shape: {global_corpus.shape}")

    pbar = tqdm(range(1,args.nepochs+1,1), disable=False)
    bestmu = 0
    es = 0
    l1, l2, l3 = getattr(args, "l1", 1e-3), getattr(args, "l2", 1e-1), getattr(args, "l3", 1e-6)
    hplanes_id = f"{args.nbits}_{datetime.now():%H%M}"

    for epoch in pbar:
        inner_pbar = tqdm(trainloader, disable=False, leave=False)

        train_loss = []
        for i, (q, c, l) in enumerate(inner_pbar):
            qpos, cpos, lpos = train_dataset.get_positive_samples(10)
            
            q = torch.cat((q, qpos.squeeze(-1)), dim=0).to(DEVICE)
            c = torch.cat((c, cpos.squeeze(-1)), dim=0).to(DEVICE)
            l = torch.cat((l, lpos), dim=0).float().to(DEVICE)

            qproj = torch.einsum("nmd,bd->nbm", W, q).tanh()   # (nplanes, batch_size, nbits)
            cproj = torch.einsum("nmd,bd->nbm", W, c).tanh()   # (nplanes, batch_size, nbits)

            loss = get_loss(qproj, cproj, l1, l2, l3)[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if i % 100 == 0:
                with torch.no_grad():
                    index = 0.5 * (torch.einsum("nmd,bd->nbm", W, global_corpus.to(DEVICE)).sign() + 1)
                    index_spread = [len(torch.unique(index[i], dim=0)) for i in range(len(index))]
                    mu, std = np.mean(index_spread), np.std(index_spread)
                    index_spread = f"{mu:.2f}±{std:.2f}"
                
                scheduler.step(mu)
            
            inner_pbar.set_postfix_str(f"Loss: {loss.item():.4f}, Index Spread: {index_spread}")
        
        with torch.no_grad():
            index = 0.5 * (torch.einsum("nmd,bd->nbm", W, global_corpus.to(DEVICE)).sign() + 1)
            index_spread = [len(torch.unique(index[i], dim=0)) for i in range(len(index))]
            mu, std = np.mean(index_spread), np.std(index_spread)
            index_spread = f"{mu:.2f}±{std:.2f}"
        
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
    W = torch.load(f"{hasher_expt_root}/hyperplanes.pkl").cpu().detach().numpy()
    np.savez_compressed(f"{hasher_expt_root}/weights.npz", *W)