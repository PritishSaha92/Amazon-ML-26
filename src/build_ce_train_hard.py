"""Build a cross-encoder training set (text pairs) with hard negatives + synthetic generator-recipe distractors.

ONLY split=="train" rows are used (pairs from --pairs, OOF stage-1 p from --oof joined on s1_id,t_id).
Pools (label, disjoint, priority order for negatives):
  pos       all label==1 train pairs
  pos_hard  duplicate copy of positives with OOF p < --hard-pos-p (stage-1 finds them hard)
  band      negatives with OOF p in [0.02, 0.98]  (all of them)
  shift     negatives with shift_sig == 3  (generator +d house-number signature)
  sib       sibling_conflict == 1
  hard      t_margin >= -0.1 or best_rank <= 1
  medium    best_rank <= 3
  random    anything else
  syn       synthetic distractors: take a train positive (S1, T), apply the generator recipe to T's text
            (80%) or to the S1 text itself (20%, with random upper-casing of the target side):
              - shift: one address number n -> n + d, d in {1,2,3,4,5,7,9,11,13,21} (zero-padding width kept)
              - insert: one word into the name, sampled from the per-country distractor-word table
                (data/tables/wordlr_train.json 'ins', LR > 2.5, count >= 150, weighted by count); 12% of
                insertions use the France F3 list (research/06) so the model learns the pattern, not the words
              - replace: swap one name word for a 'rep'-table word
              recipe mix: shift only 45%, shift+insert 30%, insert only 15%, replace (+shift 50%) 10%
              mild noise: 25% one char typo in the name (swap/drop), case follows the target text.
Output parquet: s1_id, t_id, label, pool, a, b (text "name | address" per side), split="train".
Usage: python src/build_ce_train_hard.py --out data/ce_train_hard.parquet --n-neg 1300000 --n-syn 300000
"""
import argparse, json, os, re, sys, time
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__)); from ce_common import attach_text

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/pairs_train_v3_feat.parquet")
ap.add_argument("--oof", default="data/oof_scores_v1.parquet")
ap.add_argument("--records", default="data/train_records.parquet")
ap.add_argument("--out", required=True)
ap.add_argument("--n-neg", type=int, default=1_300_000)
ap.add_argument("--neg-mix", default="band=1.0,shift=450000,sib=300000,hard=300000,medium=80000,random=-1",
                help="count per pool (1.0 = all of it; -1 = fill the remainder)")
ap.add_argument("--hard-pos-p", type=float, default=0.9)
ap.add_argument("--n-syn", type=int, default=300_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--s1-ids", default="", help="optional parquet with column s1_id: restrict to these train references (CE k-fold)")
a = ap.parse_args()
rng = np.random.default_rng(a.seed); T0 = time.time()
def log(m): print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

P = (pl.scan_parquet(a.pairs).filter(pl.col("split") == "train")
     .select("s1_id", "t_id", "country", "label", "best_rank", "t_margin", pl.col("sibling_conflict").fill_null(0), "shift_sig")
     .join(pl.scan_parquet(a.oof).filter(pl.col("split") == "train").select("s1_id", "t_id", pl.col("p").alias("oof_p")),
           on=["s1_id", "t_id"], how="left").collect())
if a.s1_ids:
    P = P.join(pl.read_parquet(a.s1_ids, columns=["s1_id"]).unique(), on="s1_id", how="semi")
    log(f"--s1-ids {a.s1_ids}: restricted to {P['s1_id'].n_unique():,} references")
log(f"train rows {len(P):,}; oof missing {P['oof_p'].null_count():,}")
pos = P.filter(pl.col("label") == 1)
hp = pos.filter(pl.col("oof_p") < a.hard_pos_p)
neg = P.filter(pl.col("label") == 0).with_columns(pool=
    pl.when(pl.col("oof_p").is_between(0.02, 0.98)).then(pl.lit("band"))
    .when(pl.col("shift_sig") == 3).then(pl.lit("shift"))
    .when(pl.col("sibling_conflict") == 1).then(pl.lit("sib"))
    .when((pl.col("t_margin") >= -0.1) | (pl.col("best_rank") <= 1)).then(pl.lit("hard"))
    .when(pl.col("best_rank") <= 3).then(pl.lit("medium")).otherwise(pl.lit("random")))
del P
log(f"pos {len(pos):,} (hard p<{a.hard_pos_p}: {len(hp):,}); neg pools available:\n{neg.group_by('pool').len().sort('pool')}")
parts, taken, fill = [], 0, None
for k, v in (kv.split("=") for kv in a.neg_mix.split(",")):
    v = float(v); g = neg.filter(pl.col("pool") == k)
    if v < 0: fill = k; continue
    m = len(g) if v == 1.0 else min(len(g), int(v))
    parts.append(g.sample(m, seed=a.seed)); taken += m
if fill:
    g = neg.filter(pl.col("pool") == fill); parts.append(g.sample(min(len(g), max(0, a.n_neg - taken)), seed=a.seed))
N = pl.concat(parts).select("s1_id", "t_id", "country", "label", "pool")
D = pl.concat([pos.select("s1_id", "t_id", "country", "label").with_columns(pool=pl.lit("pos")),
               hp.select("s1_id", "t_id", "country", "label").with_columns(pool=pl.lit("pos_hard")), N])
D = attach_text(D, a.records)
log(f"real pairs {len(D):,}")

# ---------------- synthetic distractors ----------------
D_ADD = np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 21])
FR = ["groupe", "participations", "france", "holding", "developpement", "distribution", "international",
      "club", "ecole", "comite", "amicale", "amis"]
wl = json.load(open("data/tables/wordlr_train.json"))
def table(c, kind):
    it = [(w, x[1]) for w, x in wl[c][kind].items() if x[0] > 2.5 and x[1] >= 150 and w.isalpha()]
    w = np.array([z[1] for z in it], dtype=float); return [z[0] for z in it], w / w.sum()
TAB = {(c, k): table(c, k) for c in ("India", "US") for k in ("ins", "rep")}
for c in ("India", "US"): log(f"{c}: ins words {len(TAB[(c,'ins')][0])}, rep words {len(TAB[(c,'rep')][0])}")
NUM = re.compile(r"\d+")

def case_like(w, ref):
    letters = [ch for ch in ref if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters): return w.upper()
    return w.capitalize()

def pick(c, kind):
    if kind == "ins" and rng.random() < 0.12: return FR[rng.integers(len(FR))]
    ws, p = TAB[(c if c in ("India", "US") else "US", kind)]; return ws[rng.choice(len(ws), p=p)]

def shift(addr):
    ms = list(NUM.finditer(addr))
    if not ms: return None
    m = ms[rng.integers(len(ms))]; s = m.group(); v = int(s) + int(D_ADD[rng.integers(len(D_ADD))])
    ns = str(v).zfill(len(s)) if s.startswith("0") and len(s) > 1 else str(v)
    return addr[:m.start()] + ns + addr[m.end():]

def typo(name):
    idx = [i for i, ch in enumerate(name) if ch.isalpha()]
    if len(idx) < 4: return name
    i = idx[rng.integers(1, len(idx) - 1)]
    if rng.random() < 0.5 and i + 1 < len(name): return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    return name[:i] + name[i + 1:]

def recipe(text, c):
    name, _, addr = text.partition(" | ")
    r = rng.random(); do_shift = r < 0.75; do_ins = 0.45 <= r < 0.90; do_rep = r >= 0.90
    if do_rep and rng.random() < 0.5: do_shift = True
    changed = False
    if do_shift:
        s = shift(addr)
        if s is not None: addr, changed = s, True
        elif not do_rep: do_ins = True
    toks = name.split()
    if do_rep and len(toks) >= 2:
        j = rng.integers(len(toks)); toks[j] = case_like(pick(c, "rep"), name); changed = True
    elif do_ins or (do_rep and len(toks) < 2):
        toks.insert(int(rng.integers(len(toks) + 1)), case_like(pick(c, "ins"), name)); changed = True
    name = " ".join(toks)
    if rng.random() < 0.25: name = typo(name)
    return (name + " | " + addr) if changed else None

if a.n_syn > 0:
    src = D.filter(pl.col("pool") == "pos").sample(min(a.n_syn * 12 // 10, int((D["pool"] == "pos").sum())), seed=a.seed + 7)
    rows = []
    for s1, t, c, ta, tb in zip(src["s1_id"], src["t_id"], src["country"], src["a"], src["b"]):
        from_ref = rng.random() < 0.2
        base = ta if from_ref else tb
        if not base.strip(" |"): continue
        nb = recipe(base, c)
        if nb is None: continue
        if from_ref and rng.random() < 0.5: nb = nb.upper()
        if nb == tb: continue
        rows.append((s1, f"SYN{len(rows)}:{t}", c, 0, "syn", ta, nb))
        if len(rows) >= a.n_syn: break
    S = pl.DataFrame(rows, schema=D.schema, orient="row")
    log(f"synthetic distractors {len(S):,}; examples:")
    for r in S.sample(8, seed=1).iter_rows(named=True): print(f"   {r['a']}  ==>  {r['b']}")
    D = pl.concat([D, S])
D = D.with_columns(split=pl.lit("train")).sample(fraction=1.0, shuffle=True, seed=a.seed)
log(f"final {len(D):,} pairs:\n{D.group_by('pool', 'label').len().sort('pool')}")
D.write_parquet(a.out); log(f"wrote {a.out}")
