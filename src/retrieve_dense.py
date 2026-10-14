"""Dense-embedding retrieval (multilingual E5 by default) on GPU, per country, full gallery (S2+S3) for every S1 query.
1) Embed raw text "name ; address" (scripts/accents kept) with sentence-transformers fp16, L2-normalised.
   Embeddings are cached in data/emb_{split}_{modeltag}_{text}.npy (+ _ids.npy); reruns only embed ids not yet cached.
2) Free the model, then per country: gallery fp16 on GPU, exact inner-product top-k for its S1 queries via torch
   matmul (faiss-gpu pip wheel has no Blackwell kernels), free the gallery.
Output parquet (same schema as retrieve_sparse.py): s1_id, t_id, view="dense", score, rank.

Run: python src/retrieve_dense.py --split train --k 30 --out data/cands_dense_train.parquet
     add --limit 3000 to time on a subset of queries per country (the full gallery is still embedded)."""
import argparse, gc, os, time
import numpy as np, polars as pl, torch

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train"); ap.add_argument("--model", default="intfloat/multilingual-e5-small")
ap.add_argument("--revision", default="614241f622f53c4eeff9890bdc4f31cfecc418b3")  # packaged copy: pinned model snapshot
ap.add_argument("--k", type=int, default=30); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0, help="S1 queries per country (0 = all)")
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--maxlen", type=int, default=64)
ap.add_argument("--text", default="full", choices=["full", "name"])
ap.add_argument("--reverse", action="store_true", help="queries = S2/S3 targets, gallery = S1 (view name rdense)")
ap.add_argument("--qbatch", type=int, default=1024, help="queries per search matmul")
ap.add_argument("--gblock", type=int, default=2_000_000, help="gallery rows per matmul block (qbatch x gblock fp16 scores)")
a = ap.parse_args()

tag = a.model.rstrip("/").split("/")[-1]
emb_path = f"data/emb_{a.split}_{tag}_{a.text}.npy"; ids_path = emb_path[:-4] + "_ids.npy"
prefix = "query: " if "e5" in a.model.lower() else ""  # E5: symmetric task -> "query: " on both sides

# ---- select records: full gallery of every country + (possibly limited) S1 queries ----
rec = pl.read_parquet(f"data/{a.split}_records.parquet")
countries = rec["country"].unique().sort().to_list()
parts = []
for c in countries:
    q = rec.filter((pl.col("country") == c) & (pl.col("source") == (1 if not a.reverse else 0)))  # placeholder, fixed below
    q = rec.filter((pl.col("country") == c) & ((pl.col("source") == 1) if not a.reverse else (pl.col("source") != 1)))
    g = rec.filter((pl.col("country") == c) & ((pl.col("source") != 1) if not a.reverse else (pl.col("source") == 1)))
    parts += [q.head(a.limit) if a.limit else q, g]
rec = pl.concat(parts)
del parts

def make_text(df):
    name, addr = df["business_name"].fill_null(""), df["business_address"].fill_null("")
    # .strip() matches sentence-transformers' own preprocessing (a trailing space would add a "▁" token)
    if a.text == "name": return [prefix + n.strip() for n in name]
    return [prefix + (n.strip() + " ; " + ad.strip() if ad.strip() else n.strip()) for n, ad in zip(name, addr)]

# ---- 1) embed (incremental cache) ----
if os.path.exists(emb_path):
    c_ids, c_emb = np.load(ids_path), np.load(emb_path, mmap_mode="r")
    print(f"cache: {len(c_ids):,} embeddings in {emb_path}", flush=True)
else:
    c_ids, c_emb = np.array([], dtype="U32"), np.zeros((0, 0), np.float16)
todo = rec.join(pl.DataFrame({"entity_id": c_ids.astype(object)}, schema={"entity_id": pl.Utf8}), on="entity_id", how="anti")

def tokenize(tok, texts, maxlen):
    """One fast-tokenizer call for a whole chunk (ST's per-batch tokenize is ~10x slower). Releases the GIL."""
    return tok(texts, truncation=True, max_length=maxlen)["input_ids"]

def forward(model, ids, batch):
    """Sort by token length, pad each batch only to its own max, fp16 forward (ST: mean pooling), L2-normalise."""
    pad = model.tokenizer.pad_token_id
    lens = np.fromiter(map(len, ids), dtype=np.int32, count=len(ids)); order = np.argsort(-lens, kind="stable")
    out = np.empty((len(ids), model.get_sentence_embedding_dimension()), np.float16)
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            b = order[s:s + batch]; L = lens[b[0]]
            x = np.full((len(b), L), pad, np.int64); m = np.zeros((len(b), L), np.int64)
            for j, i in enumerate(b): x[j, :lens[i]] = ids[i]; m[j, :lens[i]] = 1
            feats = {"input_ids": torch.from_numpy(x).cuda(), "attention_mask": torch.from_numpy(m).cuda()}
            e = torch.nn.functional.normalize(model(feats)["sentence_embedding"].float(), dim=-1)
            out[b] = e.half().cpu().numpy()
    return out

if len(todo):
    from concurrent.futures import ThreadPoolExecutor
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(a.model, device="cuda", revision=a.revision, model_kwargs={"torch_dtype": torch.float16}).eval()
    for c in countries:
        print(f"[{c}] to embed: {todo.filter(pl.col('country') == c).height:,}", flush=True)
    texts = make_text(todo); chunk = 500_000; starts = list(range(0, len(texts), chunk)); new = []
    # tokenization is CPU-bound: tokenize chunk i+1 in a background thread while the GPU embeds chunk i
    pool = ThreadPoolExecutor(1); t0 = time.time()
    fut = pool.submit(tokenize, model.tokenizer, texts[0:chunk], a.maxlen)
    for n_, s in enumerate(starts):
        t1 = time.time(); ids = fut.result(); t_wait = time.time() - t1
        if n_ + 1 < len(starts): fut = pool.submit(tokenize, model.tokenizer, texts[starts[n_ + 1]:starts[n_ + 1] + chunk], a.maxlen)
        new.append(forward(model, ids, a.batch))
        n = min(s + chunk, len(texts))
        print(f"  embedded {n:,}/{len(texts):,}  chunk {len(ids)/(time.time()-t1):,.0f} texts/s (tok wait {t_wait:.1f}s)  "
              f"overall {n/(time.time()-t0):,.0f} texts/s", flush=True)
    pool.shutdown(); new = np.vstack(new)
    print(f"embed: {len(texts):,} texts in {time.time()-t0:.0f}s = {len(texts)/(time.time()-t0):,.0f} texts/s", flush=True)
    del model, texts; gc.collect(); torch.cuda.empty_cache()  # free GPU before search
    c_ids = np.concatenate([c_ids, todo["entity_id"].to_numpy().astype("U32")])
    c_emb = np.vstack([np.asarray(c_emb), new]) if len(c_emb) else new
    np.save(emb_path, c_emb); np.save(ids_path, c_ids); print(f"cache: saved {len(c_ids):,} -> {emb_path}", flush=True)
    del new
pos = {k: i for i, k in enumerate(c_ids.tolist())}  # entity_id -> row in cache

def rows(ids): return np.fromiter((pos[x] for x in ids), dtype=np.int64, count=len(ids))

# ---- 2) exact search per country ----
# faiss-gpu (pip wheel 1.14) ships no sm_120 kernels -> "no kernel image" on the RTX 5090, so we do the same exact
# inner-product search as GpuIndexFlatIP(fp16) with torch: gallery resident on GPU in fp16, per query batch
# scores = Q @ G^T over gallery blocks, top-k per block, then merge blocks.
def search(G, Q, k, qbatch, gblock):
    S_all, I_all = [], []
    for s in range(0, len(Q), qbatch):
        q = torch.from_numpy(np.ascontiguousarray(Q[s:s + qbatch])).cuda()
        bs, bi = [], []
        for g0 in range(0, G.shape[0], gblock):
            sc, ix = torch.topk(q @ G[g0:g0 + gblock].T, min(k, G.shape[0] - g0), dim=1)  # fp16 matmul, fp32 accum
            bs.append(sc.float()); bi.append(ix + g0)
        sc, j = torch.topk(torch.cat(bs, 1), k, dim=1)
        S_all.append(sc.cpu().numpy()); I_all.append(torch.gather(torch.cat(bi, 1), 1, j).cpu().numpy())
    return np.vstack(S_all), np.vstack(I_all)

out = []
for c in countries:
    q = rec.filter((pl.col("country") == c) & ((pl.col("source") == 1) if not a.reverse else (pl.col("source") != 1)))
    g = rec.filter((pl.col("country") == c) & ((pl.col("source") != 1) if not a.reverse else (pl.col("source") == 1)))
    qid, gid = q["entity_id"].to_numpy(), g["entity_id"].to_numpy()
    t0 = time.time()
    G = torch.from_numpy(np.asarray(c_emb[rows(gid)])).cuda()  # (n_g, d) fp16, ~4.8 GB for 6.2M x 384
    torch.cuda.synchronize(); t1 = time.time()
    print(f"[{c}] gallery on GPU g={G.shape[0]:,} in {t1-t0:.0f}s", flush=True)
    S, I = search(G, np.asarray(c_emb[rows(qid)]), a.k, a.qbatch, a.gblock)
    print(f"[{c}] search {len(qid):,} queries in {time.time()-t1:.1f}s ({len(qid)/max(time.time()-t1,1e-9):,.0f} q/s)", flush=True)
    kk = min(a.k, len(gid))
    if a.reverse:  # rows are (target query, S1 hit); rank = rank of the S1 among this target's candidates
        out.append(pl.DataFrame({"s1_id": gid[I.ravel()], "t_id": np.repeat(qid, kk), "view": "rdense",
                                 "score": S.ravel().astype(np.float32), "rank": np.tile(np.arange(kk, dtype=np.int16), len(qid))}))
        del G; gc.collect(); torch.cuda.empty_cache(); continue
    out.append(pl.DataFrame({"s1_id": np.repeat(qid, a.k), "t_id": gid[I.ravel()], "view": "dense",
                             "score": S.ravel().astype(np.float32), "rank": np.tile(np.arange(a.k, dtype=np.int16), len(qid))}))
    del G; gc.collect(); torch.cuda.empty_cache()  # free this country's gallery before the next
res_df = pl.concat(out)
res_df.write_parquet(a.out); print("wrote", a.out, res_df.shape)
