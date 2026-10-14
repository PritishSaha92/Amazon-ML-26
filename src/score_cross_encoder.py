"""Score candidate pairs with a trained cross-encoder (models/ce_<tag>/), optionally only inside an ambiguity band.

Input : --pairs parquet with s1_id, t_id [, p]. If --band lo,hi is given AND a column named by --p-col exists,
        only rows with lo <= p <= hi are scored; otherwise every row is scored.
        --records parquet (entity_id, business_name, business_address) supplies the text.
Output: --out parquet with s1_id, t_id, p_ce (sigmoid of the logit) for the scored rows only (band rows).
Inference: bf16 weights, length-sorted batches (dynamic padding), processed in chunks of --chunk rows so memory stays
bounded on 20M+ row inputs.

Example:
  HF_HOME=$HF_HOME nice -n 10 python src/score_cross_encoder.py --model models/ce_small_smoke \
      --pairs output/val_scores_v0.parquet --records data/train_records.parquet --band 0.02,0.98 \
      --out output/ce_val_small_smoke.parquet
"""
import argparse, os, sys, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(__file__)); from ce_common import attach_text, tokenize_pairs, collate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--pairs", required=True)
ap.add_argument("--records", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--band", default=None, help="lo,hi on column --p-col (inclusive)")
ap.add_argument("--p-col", default="p")
ap.add_argument("--filter", default=None, help="optional polars SQL-ish filter, e.g. \"split = 'val'\"")
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--max-len", type=int, default=96)
ap.add_argument("--chunk", type=int, default=2_000_000); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
T0 = time.time()
def log(m): print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

lf = pl.scan_parquet(a.pairs)
if a.filter: lf = lf.sql(f"SELECT * FROM self WHERE {a.filter}")
cols = lf.collect_schema().names()
n_all = lf.select(pl.len()).collect().item()
if a.band and a.p_col in cols:
    lo, hi = map(float, a.band.split(","))
    lf = lf.filter(pl.col(a.p_col).is_between(lo, hi))
elif a.band:
    log(f"no column {a.p_col!r}: scoring ALL rows")
P = lf.select("s1_id", "t_id").collect()
if a.limit: P = P.head(a.limit)
log(f"{len(P):,} of {n_all:,} rows to score ({len(P)/max(n_all,1):.1%})")

dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()

outs, t_gpu, t_prep = [], 0.0, 0.0
for c0 in range(0, len(P), a.chunk):
    t1 = time.time()
    C = attach_text(P.slice(c0, a.chunk), a.records)
    ids = tokenize_pairs(tok, C["a"].to_list(), C["b"].to_list(), a.max_len)
    lens = np.fromiter((len(s) for s in ids), dtype=np.int32, count=len(ids))
    order = np.argsort(lens, kind="stable")
    t2 = time.time(); t_prep += t2 - t1
    logit = np.empty(len(ids), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), a.batch):
            bi = order[i:i + a.batch]
            x = collate([ids[j] for j in bi], tok.pad_token_id, dev)
            logit[bi] = model(**x).logits.squeeze(-1).float().cpu().numpy()
    torch.cuda.synchronize(); t_gpu += time.time() - t2
    outs.append(C.select("s1_id", "t_id").with_columns(p_ce=pl.Series(1 / (1 + np.exp(-logit)), dtype=pl.Float32)))
    done = c0 + len(C)
    log(f"chunk {c0//a.chunk}: {done:,}/{len(P):,}; prep {t_prep:.0f}s, gpu {t_gpu:.0f}s "
        f"({done/max(t_gpu,1e-9):,.0f} pairs/s gpu-only, {done/(t_prep+t_gpu):,.0f} pairs/s end-to-end); "
        f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
out = pl.concat(outs) if outs else pl.DataFrame(schema={"s1_id": pl.String, "t_id": pl.String, "p_ce": pl.Float32})
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); out.write_parquet(a.out)
log(f"wrote {a.out} ({len(out):,} rows). THROUGHPUT: {len(P)/max(t_gpu,1e-9):,.0f} pairs/s gpu, "
    f"{len(P)/max(t_prep+t_gpu,1e-9):,.0f} pairs/s incl. text+tokenize")
