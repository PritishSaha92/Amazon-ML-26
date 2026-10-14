"""Sparse char-ngram TF-IDF retrieval on GPU, per country, full gallery (S2+S3) for every S1 query.
For each view: hashed char n-gram TF-IDF (l2-normalised) → cosine via torch sparse @ dense on GPU → top-k.
Output parquet: s1_id, t_id, view, score, rank.

Run: python src/retrieve_sparse.py --split train --views name,addr,full --k 30 --out data/cands_sparse_train.parquet
     add --limit 20000 to time on a subset of queries per country."""
import argparse, time, sys, os
import numpy as np, polars as pl, torch, scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train"); ap.add_argument("--views", default="name,addr,full")
ap.add_argument("--k", type=int, default=30); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--nfeat", type=int, default=2 ** 18); ap.add_argument("--ngram", default="3,4")
ap.add_argument("--reverse", action="store_true", help="queries = S2/S3 targets, gallery = S1; view names get an r prefix")
ap.add_argument("--views-file", default="", help="use precomputed text views (data/views_{split}.parquet) instead of norm.py")
a = ap.parse_args()
ng = tuple(int(x) for x in a.ngram.split(","))
dev = torch.device("cuda")

def wait_for_gpu(min_free_gb=14, poll=20):
    """The GPU is shared with other jobs: block until enough memory is free."""
    while True:
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 >= min_free_gb: return
        print(f"waiting for GPU: {free/2**30:.1f} GB free < {min_free_gb} GB", flush=True); time.sleep(poll)

rec = pl.read_parquet(f"data/{a.split}_records.parquet")
if a.views_file:  # license-safe views from text_views.py: Indic names already mapped to Latin via the learned dictionary
    v = pl.read_parquet(a.views_file, columns=["entity_id", "name_latin", "addr_words"])
    rec = rec.join(v, on="entity_id", how="left").with_columns(pl.col("name_latin").fill_null(""), pl.col("addr_words").fill_null(""))
    VIEWS = {"name": lambda n, ad: n, "addr": lambda n, ad: ad, "full": lambda n, ad: (n + " | " + ad).strip(" |")}
    NCOL, ACOL = "name_latin", "addr_words"
else:
    raise SystemExit("--views-file is required (license-safe text views from text_views.py); the old norm.py path is removed")
hv = HashingVectorizer(analyzer="char_wb", ngram_range=ng, n_features=a.nfeat, alternate_sign=False, norm=None, dtype=np.float32)

def vectorize(texts, n_jobs=12):
    chunks = np.array_split(np.arange(len(texts)), max(1, len(texts) // 50_000))
    mats = Parallel(n_jobs=n_jobs)(delayed(hv.transform)([texts[i] for i in c]) for c in chunks)
    return sp.vstack(mats).tocsr()

def tfidf_l2(X, idf):
    X = X.multiply(idf[None, :]).tocsr(); X.data = np.log1p(X.data)  # sublinear tf after idf is fine for hashing
    n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ X

def to_torch_csr(X):
    return torch.sparse_csr_tensor(torch.from_numpy(X.indptr).long(), torch.from_numpy(X.indices).long(),
                                   torch.from_numpy(X.data), size=X.shape, device=dev)

out = []
for country in rec["country"].unique().sort():
    q = rec.filter((pl.col("country") == country) & ((pl.col("source") == 1) if not a.reverse else (pl.col("source") != 1)))
    g = rec.filter((pl.col("country") == country) & ((pl.col("source") != 1) if not a.reverse else (pl.col("source") == 1)))
    if a.limit: q = q.head(a.limit)
    qid, gid = q["entity_id"].to_numpy(), g["entity_id"].to_numpy()
    for view in a.views.split(","):
        t0 = time.time(); f = VIEWS[view]
        gt_ = [f(n, ad) for n, ad in zip(g[NCOL], g[ACOL])]
        qt_ = [f(n, ad) for n, ad in zip(q[NCOL], q[ACOL])]
        G = vectorize(gt_); Q = vectorize(qt_)
        df = np.bincount(G.indices, minlength=a.nfeat); idf = np.log((1 + G.shape[0]) / (1 + df)) + 1
        G = tfidf_l2(G, idf.astype(np.float32)); Q = tfidf_l2(Q, idf.astype(np.float32))
        wait_for_gpu(); t1 = time.time(); Gt = to_torch_csr(G)  # (n_g, V)
        print(f"[{country}/{view}] vectorised g={G.shape[0]:,} q={Q.shape[0]:,} nnz_g={G.nnz:,} in {t1-t0:.0f}s", flush=True)
        S_all, I_all = [], []
        s = 0; chunk = a.chunk
        while s < Q.shape[0]:
            Qc = Q[s:s + chunk]
            try:
                # densify the query chunk ON the GPU (avoids a multi-GB CPU toarray + transfer)
                Qd = torch.sparse_csr_tensor(torch.from_numpy(Qc.indptr).long(), torch.from_numpy(Qc.indices).long(),
                                             torch.from_numpy(Qc.data), size=Qc.shape, device=dev).to_dense()  # (c, V)
                sim = torch.sparse.mm(Gt, Qd.T.contiguous())  # (n_g, c): CSR @ dense is the fast path (~1.4 TFLOP/s)
                simT = sim.T.contiguous(); del sim  # top-k along the LAST dim is fast; along dim 0 it is not
                sc, ix = torch.topk(simT, a.k, dim=1)  # (c, k)
                S_all.append(sc.cpu().numpy()); I_all.append(ix.cpu().numpy())
                del Qd, simT; s += chunk
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); chunk = max(32, chunk // 2)
                print(f"  OOM → chunk={chunk}", flush=True); wait_for_gpu(8)
        S, I = np.vstack(S_all), np.vstack(I_all)
        print(f"[{country}/{view}] search {Q.shape[0]:,} queries in {time.time()-t1:.0f}s", flush=True)
        n_q = len(qid)
        if a.reverse:
            out.append(pl.DataFrame({"s1_id": gid[I.ravel()], "t_id": np.repeat(qid, a.k), "view": "r" + view,
                                     "score": S.ravel().astype(np.float32), "rank": np.tile(np.arange(a.k, dtype=np.int16), n_q)}))
            del Gt; torch.cuda.empty_cache(); continue
        out.append(pl.DataFrame({"s1_id": np.repeat(qid, a.k), "t_id": gid[I.ravel()], "view": view,
                                 "score": S.ravel().astype(np.float32), "rank": np.tile(np.arange(a.k, dtype=np.int16), n_q)}))
        del Gt; torch.cuda.empty_cache()
res = pl.concat(out).filter(pl.col("score") > 0)
res.write_parquet(a.out); print("wrote", a.out, res.shape)
