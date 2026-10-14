"""[train_ce_text.py] Variant of train_cross_encoder.py that trains on a prebuilt TEXT pair file
(src/build_ce_train_hard.py output; train split only), with --drop-pools / --pos-dup / --epochs. Otherwise identical.

Original docstring:
Train a pairwise cross-encoder reranker (1-logit sequence classifier) for S1 <-> target matching.

Training data: ONLY split=="train" rows of a build_pairs.py parquet (never val/audit).
  positives : all label==1 rows (capped at n_pairs * pos_frac, sampled uniformly if above the cap)
  negatives : label==0, mixed from difficulty pools (fractions via --neg-mix, sampled without replacement,
              pools are disjoint in priority order; any shortfall is filled from the remaining negatives):
      sib    sibling_conflict==1        (the target's true owner is another S1 with the same name)
      hard   t_margin >= -0.1 or best_rank <= 1   (this S1 is (near-)top competitor for the target / top-1 retrieved)
      medium best_rank <= 3
      random any other negative
Text per side = "name | address" from the records parquet; tokenizer(text=S1, text_pair=target, max_len).
Optimisation: AdamW (fused), linear warmup (--warmup frac) + linear decay, bf16 autocast over fp32 weights, 1 epoch,
length-bucketed batches (shuffle -> mega-chunks of 50 batches sorted by length -> shuffle batch order).
Output: models/ce_<tag>/ (HF save_pretrained model + tokenizer + train_meta.json).

Smoke:  HF_HOME=$HF_HOME nice -n 10 python src/train_cross_encoder.py --model intfloat/multilingual-e5-small \
            --tag small_smoke --n-pairs 200000
Full:   ... --model intfloat/multilingual-e5-base --tag base --n-pairs 1500000 --lr 3e-5
"""
import argparse, json, math, os, random, sys, time
# HF_HOME: taken from the environment if set, otherwise the HF default (~/.cache/huggingface)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(__file__)); from ce_common import tokenize_pairs, collate

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="parquet from build_ce_train_hard.py (s1_id,t_id,label,pool,a,b)")
ap.add_argument("--drop-pools", default="", help="comma list of pools to exclude, e.g. syn")
ap.add_argument("--pos-dup", type=int, default=0, help="extra copies of pool pos_hard")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--model", default="intfloat/multilingual-e5-base")
ap.add_argument("--tag", required=True)
ap.add_argument("--limit", type=int, default=0, help="smoke: stop after this many training pairs (after sampling)")
ap.add_argument("--lr", type=float, default=None, help="default 3e-5 (base) / 5e-5 (small)")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max-len", type=int, default=96)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--grad-ckpt", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads); pl.Config.set_tbl_rows(20)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
lr = a.lr if a.lr is not None else (5e-5 if "small" in a.model else 3e-5)
T0 = time.time()
def log(m): print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---------------- load prebuilt text pairs ----------------
D = pl.read_parquet(a.data, columns=["s1_id", "t_id", "label", "pool", "a", "b"])
if a.drop_pools: D = D.filter(~pl.col("pool").is_in(a.drop_pools.split(",")))
if a.pos_dup > 0:  # extra copies of hard positives
    D = pl.concat([D] + [D.filter(pl.col("pool") == "pos_hard")] * a.pos_dup)
D = D.sample(fraction=1.0, shuffle=True, seed=a.seed)
if a.limit: D = D.head(a.limit)
mix = {k: int(v) for k, v in D.group_by("pool").len().iter_rows()}
log(f"{len(D):,} training pairs (pos {int(D['label'].sum()):,}); pools {mix}")
tok = AutoTokenizer.from_pretrained(a.model)
ids = tokenize_pairs(tok, D["a"].to_list(), D["b"].to_list(), a.max_len)
y = D["label"].to_numpy().astype(np.float32)
lens = np.array([len(s) for s in ids])
log(f"tokenized: len mean {lens.mean():.1f}, p50 {np.median(lens):.0f}, p95 {np.percentile(lens, 95):.0f}, "
    f"truncated@{a.max_len} {np.mean(lens >= a.max_len):.3%}")

# ---------------- length-bucketed batches ----------------
order = np.random.permutation(len(ids)); B = a.batch; MEGA = B * 50
batches = []
for i in range(0, len(order), MEGA):
    chunk = order[i:i + MEGA]; chunk = chunk[np.argsort(lens[chunk], kind="stable")]
    batches += [chunk[j:j + B] for j in range(0, len(chunk), B)]
random.shuffle(batches)
for _ep in range(1, a.epochs):
    more = list(batches); random.shuffle(more); batches += more

# ---------------- model ----------------
dev = torch.device("cuda")
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
if a.grad_ckpt: model.gradient_checkpointing_enable()
no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
groups = [dict(params=[p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], weight_decay=a.wd),
          dict(params=[p for n, p in model.named_parameters() if any(k in n for k in no_decay)], weight_decay=0.0)]
opt = torch.optim.AdamW(groups, lr=lr, fused=True)
steps = len(batches); warm = max(1, int(a.warmup * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
lossf = torch.nn.BCEWithLogitsLoss()
log(f"model {a.model}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params; {steps} steps x {B}, lr {lr}, warmup {warm}")

model.train(); torch.cuda.reset_peak_memory_stats()
t_start = time.time(); seen = 0; run_loss = 0.0; run_n = 0
for step, bi in enumerate(batches):
    x = collate([ids[j] for j in bi], tok.pad_token_id, dev)
    yt = torch.from_numpy(y[bi]).to(dev, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**x).logits.squeeze(-1)
    loss = lossf(logits.float(), yt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    seen += len(bi)
    if step % 20 == 0:  # sparse .item() to avoid a sync every step
        run_loss += loss.item(); run_n += 1
    if (step + 1) % 200 == 0 or step + 1 == steps:
        el = time.time() - t_start
        log(f"step {step+1}/{steps}  loss~{run_loss/max(run_n,1):.4f}  {seen/el:,.0f} pairs/s  "
            f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB  ETA {(steps-step-1)*el/(step+1)/60:.1f} min")
        run_loss = run_n = 0
train_s = time.time() - t_start
log(f"TRAIN DONE: {seen:,} pairs in {train_s:.0f}s = {seen/train_s:,.0f} pairs/s")

out = f"models/ce_{a.tag}"; os.makedirs(out, exist_ok=True)
model.save_pretrained(out); tok.save_pretrained(out)
json.dump(dict(base_model=a.model, n_pairs=len(ids), n_pos=int(y.sum()), neg_mix=mix, lr=lr, batch=B, max_len=a.max_len,
               train_seconds=train_s, pairs_per_s=seen / train_s, data=a.data, drop_pools=a.drop_pools, pos_dup=a.pos_dup, epochs=a.epochs, split="train"),
          open(f"{out}/train_meta.json", "w"), indent=1)
log(f"saved {out}")
