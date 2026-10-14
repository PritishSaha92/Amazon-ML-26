"""Score pairs with a count-aware CE (frnlp). Text per side = "same-name refs: <bucket> | name | address", buckets from
--cnt (frnlp_common.count_table of the SAME records file the pairs live in). Output s1_id, t_id, p_ce, trunc (bool).
Fixture mode (--fixture): input has a, b, kind, split, s1_id, t_id (astra self-control); side ids per kind:
original (s1,t), refself (s1,s1), targetself (t,t); split val -> --cnt, test -> --cnt-test."""
import argparse, os, sys, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_common import attach_text, tokenize_pairs, collate
from frnlp_common import prefix

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="comma list; output p_<i> per model and p_ce = mean")
ap.add_argument("--pairs", required=True); ap.add_argument("--records", default=None)
ap.add_argument("--cnt", required=True); ap.add_argument("--cnt-test", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--filter", default=None)
ap.add_argument("--fixture", action="store_true")
ap.add_argument("--max-len", type=int, default=112); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--chunk", type=int, default=2_000_000)
a = ap.parse_args(); T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

lf = pl.scan_parquet(a.pairs)
if a.filter: lf = lf.sql(f"SELECT * FROM self WHERE {a.filter}")
P = lf.collect()
cnt = pl.read_parquet(a.cnt, columns=["entity_id", "bucket", "n_same"])
if a.fixture:
    ct = pl.read_parquet(a.cnt_test, columns=["entity_id", "bucket", "n_same"])
    P = P.with_columns(ida=pl.when(pl.col("kind") == "targetself").then(pl.col("t_id")).otherwise(pl.col("s1_id")),
                       idb=pl.when(pl.col("kind") == "refself").then(pl.col("s1_id")).otherwise(pl.col("t_id")))
    both = pl.concat([cnt.with_columns(split=pl.lit("val")), ct.with_columns(split=pl.lit("test"))])
    P = (P.with_row_index("_i")
         .join(both.rename({"entity_id": "ida", "bucket": "bs", "n_same": "ns"}), on=["ida", "split"], how="left")
         .join(both.rename({"entity_id": "idb", "bucket": "bt", "n_same": "nt"}), on=["idb", "split"], how="left").sort("_i").drop("_i"))
    assert P["bs"].null_count() == 0 and P["bt"].null_count() == 0
    P = P.with_columns(a_raw=pl.col("a"), b_raw=pl.col("b")).with_columns(a=prefix("bs") + pl.col("a"), b=prefix("bt") + pl.col("b"))
else:
    P = attach_text(P.select("s1_id", "t_id"), a.records)
    P = (P.with_row_index("_i").join(cnt.rename({"entity_id": "s1_id", "bucket": "bs", "n_same": "ns"}), on="s1_id", how="left")
         .join(cnt.rename({"entity_id": "t_id", "bucket": "bt", "n_same": "nt"}), on="t_id", how="left").sort("_i").drop("_i"))
    assert P["bs"].null_count() == 0 and P["bt"].null_count() == 0, "pair id missing from cnt table"
    P = P.with_columns(a=prefix("bs") + pl.col("a"), b=prefix("bt") + pl.col("b"))
log(f"{len(P):,} rows; example A: {P['a'][0]!r} B: {P['b'][0]!r}")
dev = torch.device("cuda"); models = a.models.split(",")
tok = AutoTokenizer.from_pretrained(models[0])
A_, B_ = P["a"].to_list(), P["b"].to_list()
ids = tokenize_pairs(tok, A_, B_, a.max_len)
full = tokenize_pairs(tok, A_, B_, 10_000)
trunc = np.array([len(f) > a.max_len for f in full]); del full
lens = np.fromiter((len(s) for s in ids), dtype=np.int32, count=len(ids)); order = np.argsort(lens, kind="stable")
log(f"tokenized; truncated@{a.max_len} {trunc.mean():.4%}")
cols = {}
for mi, m in enumerate(models):
    model = AutoModelForSequenceClassification.from_pretrained(m, dtype=torch.bfloat16).to(dev).eval()
    logit = np.empty(len(ids), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), a.batch):
            bi = order[i:i + a.batch]
            logit[bi] = model(**collate([ids[j] for j in bi], tok.pad_token_id, dev)).logits.squeeze(-1).float().cpu().numpy()
    cols[f"p_{mi}"] = (1 / (1 + np.exp(-logit))).astype(np.float32); log(f"scored {m}"); del model; torch.cuda.empty_cache()
out = P.with_columns(**{k: pl.Series(v) for k, v in cols.items()}, trunc=pl.Series(trunc))
out = out.with_columns(p_ce=pl.mean_horizontal([pl.col(k) for k in cols]).cast(pl.Float32))
keep = [c for c in out.columns if c not in ("a", "b")] if not a.fixture else out.columns
if not a.fixture: keep = ["s1_id", "t_id", "p_ce"] + list(cols) + ["trunc", "bs", "bt", "ns", "nt"]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); out.select(keep).write_parquet(a.out)
log(f"wrote {a.out} {len(out):,}")
