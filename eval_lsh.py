from train_prehash import *
from train_hyperplanes import *
from sklearn.metrics import average_precision_score
from opas.data import get_all_pair_scores

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LSH(object):
    def __init__(self, hash_size, input_dim, num_hashtables, hyperplanes_file, queries, gt_sims, device='cpu'):
        self.m = hash_size
        self.L = num_hashtables
        self.d = input_dim

        self.device = device
        self.hyperplanes_file = hyperplanes_file

        self.sims = gt_sims
        self.q = queries

        self._init_tables()
        self._init_planes()

    def _init_planes(self):
        try:
            self.planes = torch.load(self.hyperplanes_file, map_location=self.device).detach()
            assert self.L <= self.planes.shape[0], f"Expected {self.L} planes, got {self.planes.shape[0]}"
            self.planes = self.planes[:self.L]
            assert self.planes.shape == (self.L, self.m, self.d), f"Expected {(self.L, self.m, self.d)}, got {self.planes.shape}"
        except:
            print("Initializing random planes...")
            self.planes = torch.randn(self.L, self.m, self.d, device=self.device)

    def _init_tables(self):
        self.tables = [{} for _ in range(self.L)]

    @staticmethod
    def _cvt_index_to_hashstr(i):
        return "".join(map(str, map(int, i.cpu().numpy())))

    def index(self, corpus):
        # corpus: (n, d)
        st = perf_counter()
        self.corpus = corpus
        
        hashcodes = self.hashall(corpus)
        for t in range(self.L):
            unique_codes = torch.unique(hashcodes[t], dim=0)
            for code in unique_codes:
                key = self._cvt_index_to_hashstr(code)
                lsh.tables[t][key] = torch.where(torch.all(hashcodes[t] == code, dim=1))[0].cpu().long()
        print(f"Indexed {len(corpus)} items in {perf_counter() - st:.2f}s")

    def hashall(self, tensor):
        # tensor: (n, d)
        tensor = torch.atleast_2d(tensor)
        hashcodes = 0.5 * (torch.einsum("nmd,bd->nbm", self.planes, tensor.to(self.device)).sign() + 1).cpu()
        return hashcodes.squeeze()

    def _query_single(self, query):
        # query: (1, d) or (d)
        hashcodes = self.hashall(query)
        result_idxs = []
        for t, code in enumerate(hashcodes):
            key = self._cvt_index_to_hashstr(code)
            matches = self.tables[t].get(key, torch.tensor([]))
            if len(matches) > 0:
                result_idxs.append(matches)
        result_idxs = torch.cat(result_idxs).unique()
        return self.corpus[result_idxs], result_idxs

    def _query_single_loose(self, query, kbits):
        # returns all items where first k bits of hash match
        if kbits == self.m:
            return self._query_single(query)
        hashcodes = self.hashall(query)
        result_idxs = []
        for t, code in enumerate(hashcodes):
            key = self._cvt_index_to_hashstr(code)
            random_k_idxs = np.random.permutation(self.m)[:kbits]
            matching_keys = [k for k in self.tables[t].keys() if k[random_k_idxs] == key[random_k_idxs]]
            for mk in matching_keys:
                matches = self.tables[t].get(mk, torch.tensor([]))
                if len(matches) > 0:
                    result_idxs.append(matches)
        result_idxs = torch.cat(result_idxs).unique()
        return self.corpus[result_idxs], result_idxs

    def query(self, qidx, k, distance_func="orig", kbits=None):
        # query: index of query in self.q
        query = self.q[qidx]
        if kbits is not None:
            matches, match_idxs = self._query_single_loose(query, kbits)
        else:
            matches, match_idxs = self._query_single(query)

        if distance_func == "orig":
            sims = self.sims[qidx][match_idxs]
        else:
            if distance_func == "cosine":
                f = LSH.CosineSimilarity
            elif distance_func == "euclidean" or distance_func == "mse":
                f = LSH.EuclideanDistance 
            sims = f(query[None].repeat_interleave(len(matches), dim=0), matches)
        
        topk = sims.topk(k=k if k else len(sims)).indices
        return matches[topk], match_idxs[topk], sims
    
    def query_batch(self, queries, k, distance_func="cosine", kbits=None):
        # TODO implement batch version
        # queries: (n, d)
        return [self.query(q, k, distance_func, kbits) for q in queries]
    
    def _query_single_loose_multi_kbits(self, query, kbits):
        # returns all items where first k bits of hash match, k bits is a list
        hashcodes = self.hashall(query)
        result_idxs = {k: [] for k in kbits}
        for t, code in enumerate(hashcodes):
            key = self._cvt_index_to_hashstr(code)
            for kcom in kbits:
                matching_keys = [k for k in self.tables[t].keys() if k[:kcom] == key[:kcom]]
                for mk in matching_keys:
                    matches = self.tables[t].get(mk, torch.tensor([]))
                    if len(matches) > 0:
                        result_idxs[kcom].append(matches)
        
        for kcom in kbits:
            result_idxs[kcom] = torch.cat(result_idxs[kcom]).unique()
        matches_out = {kcom: result_idxs[kcom] for kcom in kbits}
        
        return matches_out, result_idxs

    def query_multi_kbits(self, qidx, k, kbits, distance_func="orig"):
        query = self.q[qidx]
        matches, match_idxs = self._query_single_loose_multi_kbits(query, kbits)
        
        if distance_func == "orig":
            sims = {kcom: self.sims[qidx][match_idxs[kcom]] for kcom in kbits}
        else:
            if distance_func == "cosine":
                f = LSH.CosineSimilarity
            elif distance_func == "euclidean" or distance_func == "mse":
                f = LSH.EuclideanDistance 
            sims = {kcom: f(query[None].repeat_interleave(len(matches[kcom]), dim=0), matches[kcom]) for kcom in kbits}
        
        for kcom in kbits:
            topk = sims[kcom].topk(k=k if k else len(sims[kcom])).indices
            matches[kcom] = matches[kcom][topk]
            match_idxs[kcom] = match_idxs[kcom][topk]
        return matches, match_idxs, sims        

    @staticmethod
    def CosineSimilarity(x, y):
        return 0.5 * (F.cosine_similarity(x, y) + 1)

    @staticmethod
    def EuclideanDistance(x, y):
        return -(x - y).pow(2).sum(dim=-1).sqrt()
    

if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()

    try:
        skip_logging = cli_args.skip_logging
        def noop(*args, **kwargs):
            pass
        if skip_logging:
            print = noop    # disable printing
    except:
        pass

    expt_id = cli_args.expt_id
    hasher_expt_num = cli_args.hexpt_num
    hasher_expt_root = f"hashing/{expt_id}/{hasher_expt_num}/"
    expt_root = f"models/{expt_id}/"

    import pickle
    with open(f"{hasher_expt_root}/args.pkl", 'rb') as f:
        args = pickle.load(f)
    
    args = OmegaConf.merge(OmegaConf.create(vars(args)), cli_args)
    args = argparse.Namespace(**args)

    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    hasher = torch.load(f"{hasher_expt_root}/hasher_best.pt", map_location=DEVICE)
    hasher = hasher.eval()

    saved_gt_path = f"hashing/{expt_id}_embeds_gt.pkl"

    if os.path.exists(saved_gt_path) and getattr(args, "skip_gt", True):
        dataset = args.dataset
        savedict = torch.load(saved_gt_path)
        corpus_embedded = savedict["corpus"].cpu()
        labels = savedict["labels"].cpu()
        global_scores = savedict["scores"].cpu()
        test_query_embedded = savedict["queries"].cpu()
        print("Loaded ground truth")

    else:        
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

        image_embed_model, TRAIN_FILE, VAL_FILE, TEST_FILE = get_image_embed_model(dataset, [TRAIN_FILE, VAL_FILE, TEST_FILE], device=DEVICE)

        train_dataset = PairDatasetTest(TRAIN_FILE)
        val_dataset = PairDatasetTest(VAL_FILE)
        test_dataset = PairDatasetTest(TEST_FILE)   

        @torch.no_grad()
        def embed_full(tensor):
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor)
            
            return embed_full_corpus(DummyDataset(tensor), embed_model=embed_model, preembed_model=preembed_model, image_embed_model=image_embed_model, inner_batch_size=800, aggregator=aggregator, verbose=True)


        test_query_embedded = embed_full(test_dataset.q)
        corpus_embedded = [
            embed_full(test_dataset.c),
            embed_full(train_dataset.c),
            embed_full(val_dataset.c)
        ]
        labels = torch.hstack([
            torch.stack([
                F.one_hot(torch.from_numpy(l), num_classes=len(test_dataset.c)).sum(dim=0)
                for l in test_dataset.l
            ]),
            torch.zeros(len(test_dataset.q), len(train_dataset.c)),
            torch.zeros(len(test_dataset.q), len(val_dataset.c))
        ])
        global_scores = torch.hstack([
            get_all_pair_scores(test_query_embedded, corpus_embedded[0], models=[model, scoremodel], args=args),
            get_all_pair_scores(test_query_embedded, corpus_embedded[1], models=[model, scoremodel], args=args),
            get_all_pair_scores(test_query_embedded, corpus_embedded[2], models=[model, scoremodel], args=args)
        ])
        corpus_embedded = torch.vstack(corpus_embedded)
        print("Scores computed")

        savedict = {"corpus": corpus_embedded, "labels": labels, "scores": global_scores, "queries": test_query_embedded}
        torch.save(savedict, saved_gt_path)


    ####### LSH ########
    with torch.no_grad():
        corpus_embedded_out = []
        for i in range(0,len(corpus_embedded),200):
            corpus_embedded_out.append(hasher[1](corpus_embedded[i:i+200].to(DEVICE)).cpu())
        corpus_embedded = torch.vstack(corpus_embedded_out)
        test_query_embedded = hasher[0](test_query_embedded.to(DEVICE)).cpu()
    
    topk = None
    nbits = args.m
    kbits = getattr(args, "kbits", nbits)
    if type(kbits) == int:
        kbits = [kbits]
    L = args.L
    data_dim = corpus_embedded.shape[-1]
    hyperplanes_file = f'{hasher_expt_root}/{args.hplanes}.pkl'

    seed_everything(args.seed)
    lsh = LSH(
        hash_size=nbits,
        input_dim=data_dim,
        num_hashtables=L,
        hyperplanes_file=hyperplanes_file,
        queries=test_query_embedded,
        gt_sims=global_scores,
        device=DEVICE
    )
    lsh.index(corpus_embedded)

    randqueries = torch.randperm(len(lsh.q))[:100]
    num_matches, num_relevant = {k: [] for k in kbits}, {k: [] for k in kbits}
    ranked_output = {k: [] for k in kbits}
    MAP = {k: [] for k in kbits}
    for i in tqdm(range(len(lsh.q[randqueries])), desc="Evaluating...", disable=skip_logging):
        matches, match_idxs, sims = lsh.query_multi_kbits(i, topk, kbits, distance_func="orig")
        for k in kbits:
            sims[k] = sorted(sims[k], reverse=True)
            true_labels = labels[i]
            total_rel = true_labels.sum().item()

            true_labels = torch.cat([true_labels[match_idxs[k]], torch.zeros(len(sims[k]) - len(match_idxs[k]))])
            retrieved_rel = true_labels.sum().item()
            
            MAP[k].append(average_precision_score(true_labels, sims[k]) * retrieved_rel / total_rel)

            num_matches[k].append(len(matches[k]))
            num_relevant[k].append(true_labels.sum().item())
    MAP = {k: np.mean(MAP[k]) for k in kbits}
    # for k in kbits:
    #     print(f"{k}, MAP: {MAP[k]}, Mean matches: {np.mean(num_matches[k]):.2f}, Mean relevant: {np.mean(num_relevant[k]):.2f}")
    
    ls = [i for j in [(MAP[k], np.mean(num_matches[k]), np.mean(num_relevant[k])) for k in kbits] for i in j]
    os.makedirs(f"tmp/multi_{dataset}", exist_ok=True)
    np.save(f"tmp/multi_{dataset}/tmp_{args.save_expt_num}.npy", ls)
    print(ls)
    exit()

    with open(f"{hasher_expt_root}/lsh_perf.csv", "a+") as f:
        for k in kbits:
            f.write(f"{k}, {MAP[k]:.4f}, {np.mean(num_matches[k]):.2f}, {np.mean(num_relevant[k]):.2f}\n")

    # else:
    #     num_matches, num_relevant = [], []
    #     ranked_output = []
    #     MAP = []
    #     for i in tqdm(range(len(lsh.q)), desc="Evaluating..."):
    #         matches, match_idxs, sims = lsh.query(i, k, distance_func="orig", kbits=kbits)
    #         sims = sorted(sims, reverse=True)
    #         true_labels = labels[i]
    #         total_rel = true_labels.sum().item()

    #         true_labels = torch.cat([true_labels[match_idxs], torch.zeros(len(sims) - len(match_idxs))])
    #         retrieved_rel = true_labels.sum().item()
            
    #         MAP.append(average_precision_score(true_labels, sims) * retrieved_rel / total_rel)

    #         num_matches.append(len(matches))
    #         num_relevant.append(true_labels.sum().item())

    #     MAP = np.mean(MAP)

    #     with open(f"{hasher_expt_root}/lsh_perf.csv", "a+") as f:
    #         f.write(f"{kbits}, {MAP:.4f}, {np.mean(num_matches):.2f}, {np.mean(num_relevant):.2f}\n")
        
    #     print(f"MAP: {MAP:.4f},")
    #     print(f"Mean matches: {np.mean(num_matches):.2f}, Mean relevant: {np.mean(num_relevant):.2f}")