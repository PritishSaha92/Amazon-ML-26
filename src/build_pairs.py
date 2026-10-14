"""Build the pair table for the matcher: one row per unique (s1_id, t_id) candidate, from one or more candidate files
(schema s1_id, t_id, view, score, rank; rank 0 = best within view for that s1).

Features
  retrieval : {view}_score (raw, 0.0 if not retrieved by view), {view}_rank (RANK_MISSING if not),
              {view}_nscore = 1 - rank/K_view (K_view = max rank + 1 in that view over all inputs; 0 if not retrieved),
              n_views, best_rank, max_score / sum_score = max / sum of {view}_nscore (rank-normalised, not raw scores)
  per-S1    : s1_n_cands, s1_max_score, s1_rank_of_pair (0 = best by max_score, ties share min rank), s1_gap_to_best
  per-target: t_n_owners (distinct s1 that retrieved t), t_rank_of_pair (0 = best owner by max_score), t_best_other_score
              (best max_score of a DIFFERENT s1 for t, 0 if none), t_margin = max_score - t_best_other_score
  per-view  : s1_gap_{v} (s1's max {v}_score - this {v}_score), t_rank_{v}, t_best_other_{v}, t_margin_{v}
              (same definitions as the merged ones, on raw {v}_score; non-retrieved pairs count with score 0)
  metadata  : country, t_source (2/3), split; train only: label, sibling_conflict (t's true owner is a different S1 with
              the same lowercased+stripped business_name as this s1; null for test)
All context features are computed on the FULL candidate set before any sampling. Processing is per country
(targets never cross countries, so per-target competition is exact within a country partition); ids are handled as
UInt32 internally (s1: numeric part; t: source*1e9 + numeric part) and restored to strings on output.

Sampling (--split train --sample-s1 N): keep all pairs of N random 'train'-split S1 (country-stratified, proportional to
the country's share of train-split S1 present in the candidates) + all pairs of every 'val' and 'audit' S1.
--extra-sibling-negatives: sampled S1 already keep all their candidates (incl. sibling targets); this flag additionally
keeps the sibling_conflict=1 rows (label 0 hard negatives) of NON-sampled train-split S1, flagged is_extra=1.

Candidate rule (--keep-rule "<polars expr>", default none = keep the full union): evaluated (with `pl` in scope) right
after the union and the per-view {v}_score/_rank/_nscore, n_views, best_rank, max_score, sum_score columns exist, and
BEFORE every context/competition feature, labels and sampling -> all features describe the reduced candidate set.
Helper columns for the rule only (not written): fwd_rank = min(full_rank, addr_rank, name_rank, dense_rank) (99 if none);
any of the six standard {v}_rank columns absent from the inputs is treated as 99. Example:
  --keep-rule "(pl.col('n_views') >= 2) | (pl.col('fwd_rank') <= 8) | (pl.col('rdense_rank') <= 4) | (pl.col('rfull_rank') <= 2)"

Run:
  POLARS_MAX_THREADS=12 nice -n 10 python src/build_pairs.py --cands data/cands_timing3.parquet --split train \
      --sample-s1 5000 --out data/pairs_timing3.parquet
"""
import argparse, os, time
import polars as pl
import pyarrow.parquet as pq

RANK_MISSING = 99
T_MUL = 1_000_000_000

ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True, help="comma-separated candidate parquet files")
ap.add_argument("--split", required=True, choices=["train", "test"])
ap.add_argument("--out", required=True)
ap.add_argument("--sample-s1", type=int, default=0, help="train only: number of 'train'-split S1 to keep (0 = keep all)")
ap.add_argument("--extra-sibling-negatives", action="store_true")
ap.add_argument("--seed", type=int, default=20260925)
ap.add_argument("--data", default="data")
ap.add_argument("--keep-s1-file", default="", help="train only: parquet with column s1_id; keep EXACTLY these S1 (no val/audit "
                "added, overrides --sample-s1). Default off (unchanged behaviour)")
ap.add_argument("--keep-rule", default="", help="polars expression string; keep only union pairs satisfying it (see docstring)")
a = ap.parse_args()
T0 = time.time()
def log(msg): print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

files = [f for f in a.cands.split(",") if f]
is_train = a.split == "train"
if a.sample_s1 and not is_train: raise SystemExit("--sample-s1 only valid with --split train")
if a.keep_s1_file and not is_train: raise SystemExit("--keep-s1-file only valid with --split train")
keep_rule = eval(a.keep_rule, {"pl": pl}) if a.keep_rule else None
if keep_rule is not None: assert isinstance(keep_rule, pl.Expr), "--keep-rule must evaluate to a polars expression"
STD_VIEWS = ["full", "addr", "name", "dense", "rdense", "rfull"]; FWD_VIEWS = ["full", "addr", "name", "dense"]

s1_int = pl.col("s1_id").str.slice(3).cast(pl.UInt32)
t_int = (pl.col("t_id").str.slice(1, 1).cast(pl.UInt32) * T_MUL + pl.col("t_id").str.slice(3).cast(pl.UInt32)).cast(pl.UInt32)

def scan_cands():
    return pl.concat([pl.scan_parquet(f).select(s=s1_int, t=t_int, view=pl.col("view").cast(pl.String),
                                                 score=pl.col("score").cast(pl.Float32), rank=pl.col("rank").cast(pl.Int16))
                      for f in files], how="vertical")

# ---- S1 table: country, name hash, split -------------------------------------------------------------------
recs = pl.scan_parquet(f"{a.data}/{a.split}_records.parquet").filter(pl.col("source") == 1).select(
    s=pl.col("entity_id").str.slice(3).cast(pl.UInt32), country=pl.col("country"),
    nh=pl.col("business_name").str.to_lowercase().str.strip_chars().hash())
if is_train:
    recs = recs.join(pl.scan_parquet(f"{a.data}/split.parquet").select(s=s1_int, split=pl.col("split")), on="s", how="left")
else:
    recs = recs.with_columns(split=pl.lit("test"))
vk = scan_cands().group_by("view").agg(K=(pl.col("rank").max().cast(pl.Int32) + 1)).collect(engine="streaming")
K = dict(zip(vk["view"], vk["K"])); views = sorted(K)  # per-view K = max rank + 1 (global over all input files)
present = scan_cands().select(pl.col("s").unique()).collect(engine="streaming")
s1tab = recs.join(present.lazy(), on="s", how="semi").collect()
log(f"views={views} K={K} S1 present={len(s1tab):,} (in records: {len(s1tab):,} / cand s1 {len(present):,})")
assert len(s1tab) == len(present), "some candidate s1_id not found in records for this split"

# ---- sampling decision (global, before per-country loop) ---------------------------------------------------
if is_train and a.keep_s1_file:
    keep_s = pl.read_parquet(a.keep_s1_file).select(s=s1_int).unique()
    keep_s = keep_s.join(s1tab.select("s"), on="s", how="semi")
    log(f"--keep-s1-file: keeping exactly {len(keep_s):,} S1 from {a.keep_s1_file}")
elif is_train and a.sample_s1:
    tr = s1tab.filter(pl.col("split") == "train")
    n_tr = len(tr); parts = []
    for i, (c, g) in enumerate(sorted(tr.group_by("country"), key=lambda x: x[0][0])):
        k = min(len(g), int(round(a.sample_s1 * len(g) / n_tr)))
        parts.append(g.sample(n=k, seed=a.seed + i).select("s"))
    keep_s = pl.concat(parts + [s1tab.filter(pl.col("split").is_in(["val", "audit"])).select("s")])
    log(f"sampling: {sum(len(p) for p in parts):,} train-split S1 of {n_tr:,} + {len(keep_s) - sum(len(p) for p in parts):,} val/audit S1")
else:
    keep_s = None

if is_train:
    tp = pl.scan_parquet(f"{a.data}/train_pairs.parquet").select(s=s1_int, t=t_int)
    owners = tp.select("t", owner="s").unique(subset="t").collect()  # each target has at most one true owner
    nh_all = pl.scan_parquet(f"{a.data}/train_records.parquet").filter(pl.col("source") == 1).select(
        owner=pl.col("entity_id").str.slice(3).cast(pl.UInt32),
        owner_nh=pl.col("business_name").str.to_lowercase().str.strip_chars().hash()).collect()
    owners = owners.join(nh_all, on="owner", how="left"); del nh_all

# ---- per-country build ----------------------------------------------------------------------------------------
stats = dict(pairs_union=0, pairs=0, out=0, per_view={v: 0 for v in views}, true=0, hit=0, lab1=0, lab0=0, split={}, sib=0, extra=0)
writer = None
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
tmp_out = a.out + ".tmp"
for country in sorted(s1tab["country"].unique().to_list()):
    tc = time.time()
    cs = s1tab.filter(pl.col("country") == country)
    agg = []
    for v in views:
        f = pl.col("view") == v
        agg += [pl.col("score").filter(f).max().fill_null(0.0).alias(f"{v}_score"),
                pl.col("rank").filter(f).min().fill_null(RANK_MISSING).cast(pl.Int16).alias(f"{v}_rank")]
    df = (scan_cands().join(cs.lazy().select("s"), on="s", how="semi")
          .group_by("s", "t").agg(*agg, n_views=pl.col("view").n_unique().cast(pl.Int8),
                                  best_rank=pl.col("rank").min().cast(pl.Int16))
          .collect(engine="streaming"))
    # rank-normalised per-view score: 1 - rank/K (0 if not retrieved); merged scores use nscore so no view's raw scale dominates
    df = df.with_columns([pl.when(pl.col(f"{v}_rank") == RANK_MISSING).then(0.0).otherwise(1.0 - pl.col(f"{v}_rank") / K[v])
                          .cast(pl.Float32).alias(f"{v}_nscore") for v in views])
    nsc = [f"{v}_nscore" for v in views]
    df = df.with_columns(max_score=pl.max_horizontal(nsc), sum_score=pl.sum_horizontal(nsc).cast(pl.Float32))
    n_union = len(df); stats["pairs_union"] += n_union
    if keep_rule is not None:  # candidate rule on the union, before any context feature
        miss = [pl.lit(RANK_MISSING, pl.Int16).alias(f"{v}_rank") for v in STD_VIEWS if v not in views]
        fwd = [pl.col(f"{v}_rank") for v in FWD_VIEWS if v in views]
        hx = df.with_columns(*miss).with_columns(
            fwd_rank=(pl.min_horizontal(fwd) if fwd else pl.lit(RANK_MISSING)).cast(pl.Int16))
        df = df.filter(hx.select(keep_rule.fill_null(False).alias("_k"))["_k"]); del hx
    # per-S1 context
    df = df.with_columns(
        s1_n_cands=pl.len().over("s").cast(pl.Int32),
        s1_max_score=pl.col("max_score").max().over("s"),
        s1_rank_of_pair=(pl.col("max_score").rank("min", descending=True).over("s") - 1).cast(pl.Int32),
    ).with_columns(s1_gap_to_best=pl.col("s1_max_score") - pl.col("max_score"))
    # per-target competition
    tstat = df.group_by("t").agg(t_n_owners=pl.len().cast(pl.Int32), t_top1=pl.col("max_score").max(),
                                 t_top2=pl.col("max_score").top_k(2).min().filter(pl.len() > 1).first())
    df = df.join(tstat, on="t", how="left").with_columns(
        t_rank_of_pair=(pl.col("max_score").rank("min", descending=True).over("t") - 1).cast(pl.Int32),
        t_best_other_score=pl.when(pl.col("t_n_owners") == 1).then(0.0)
                             .when(pl.col("max_score") >= pl.col("t_top1")).then(pl.col("t_top2"))
                             .otherwise(pl.col("t_top1")).cast(pl.Float32),
    ).with_columns(t_margin=pl.col("max_score") - pl.col("t_best_other_score")).drop("t_top1", "t_top2")
    del tstat
    # per-view context on each view's own raw score scale
    for v in views:
        sc = f"{v}_score"
        tv = df.group_by("t").agg(_n=pl.len(), _t1=pl.col(sc).max(), _t2=pl.col(sc).top_k(2).min())
        df = df.join(tv, on="t", how="left").with_columns(
            (pl.col(sc).max().over("s") - pl.col(sc)).alias(f"s1_gap_{v}"),
            (pl.col(sc).rank("min", descending=True).over("t") - 1).cast(pl.Int32).alias(f"t_rank_{v}"),
            pl.when(pl.col("_n") == 1).then(0.0).when(pl.col(sc) >= pl.col("_t1")).then(pl.col("_t2"))
              .otherwise(pl.col("_t1")).cast(pl.Float32).alias(f"t_best_other_{v}"),
        ).with_columns((pl.col(sc) - pl.col(f"t_best_other_{v}")).alias(f"t_margin_{v}")).drop("_n", "_t1", "_t2")
        del tv
    n_all = len(df); stats["pairs"] += n_all
    for v in views: stats["per_view"][v] += int((df[f"{v}_rank"] != RANK_MISSING).sum())
    # labels + recall on the full candidate set
    if is_train:
        tpc = tp.join(cs.lazy().select("s"), on="s", how="semi").collect()
        df = df.join(tpc.with_columns(label=pl.lit(1, pl.Int8)), on=["s", "t"], how="left").with_columns(pl.col("label").fill_null(0))
        stats["true"] += len(tpc); stats["hit"] += int(df["label"].sum()); del tpc
        df = (df.join(cs.select("s", "nh"), on="s", how="left").join(owners, on="t", how="left")
                .with_columns(sibling_conflict=((pl.col("owner").is_not_null()) & (pl.col("owner") != pl.col("s"))
                                                & (pl.col("owner_nh") == pl.col("nh"))).cast(pl.Int8))
                .drop("owner", "owner_nh", "nh"))
    else:
        df = df.with_columns(sibling_conflict=pl.lit(None, pl.Int8))
    # sampling
    df = df.with_columns(is_extra=pl.lit(0, pl.Int8))
    if keep_s is not None:
        in_keep = pl.col("s").is_in(keep_s["s"].implode())
        if a.extra_sibling_negatives:
            df = df.filter(in_keep | (pl.col("sibling_conflict") == 1)).with_columns(
                is_extra=(~pl.col("s").is_in(keep_s["s"].implode())).cast(pl.Int8))
        else:
            df = df.filter(in_keep)
    df = df.join(cs.select("s", "split"), on="s", how="left").with_columns(
        country=pl.lit(country), t_source=(pl.col("t") // T_MUL).cast(pl.Int8),
        s1_id=pl.lit("S1-") + pl.col("s").cast(pl.String),
        t_id=pl.lit("S") + (pl.col("t") // T_MUL).cast(pl.String) + pl.lit("-") + (pl.col("t") % T_MUL).cast(pl.String),
    ).sort("s", "max_score", descending=[False, True])
    cols = ["s1_id", "t_id", "country", "split", "t_source"] + [c for v in views for c in (f"{v}_score", f"{v}_rank", f"{v}_nscore")] + \
           ["n_views", "best_rank", "max_score", "sum_score", "s1_n_cands", "s1_max_score", "s1_rank_of_pair", "s1_gap_to_best",
            "t_n_owners", "t_rank_of_pair", "t_best_other_score", "t_margin"] + \
           [c for v in views for c in (f"s1_gap_{v}", f"t_rank_{v}", f"t_best_other_{v}", f"t_margin_{v}")] + (["label"] if is_train else []) + \
           ["sibling_conflict", "is_extra"]
    df = df.select(cols)
    stats["out"] += len(df); stats["extra"] += int(df["is_extra"].sum())
    for sp, n in df.group_by("split").len().iter_rows(): stats["split"][sp] = stats["split"].get(sp, 0) + n
    if is_train:
        stats["lab1"] += int(df["label"].sum()); stats["lab0"] += int((df["label"] == 0).sum()); stats["sib"] += int(df["sibling_conflict"].sum())
    tbl = df.to_arrow()
    if writer is None: writer = pq.ParquetWriter(tmp_out, tbl.schema, compression="zstd")
    writer.write_table(tbl, row_group_size=1_000_000)
    kr = f" (union {n_union:,} = {n_union / len(cs):.1f}/S1 -> kept {n_all:,} = {n_all / len(cs):.1f}/S1)" if keep_rule is not None else ""
    log(f"{country}: {len(cs):,} S1, {n_all:,} unique pairs{kr}, {len(df):,} written ({time.time() - tc:.1f}s)")
    del df, tbl
writer.close(); os.replace(tmp_out, a.out)

# ---- summary ------------------------------------------------------------------------------------------------------
print(f"\n=== summary ({a.split}) files={files}")
if keep_rule is not None:
    print(f"keep-rule: {a.keep_rule}")
    print(f"union before rule = {stats['pairs_union']:,}  ({stats['pairs_union'] / len(s1tab):.1f}/S1)  ->  after rule = "
          f"{stats['pairs']:,}  ({stats['pairs'] / len(s1tab):.1f}/S1)  kept {stats['pairs'] / max(stats['pairs_union'], 1):.2%}")
print(f"unique pairs (full candidate set) = {stats['pairs']:,}  ({stats['pairs'] / len(s1tab):.1f}/S1)")
print("pairs per view: " + "  ".join(f"{v}={n:,}" for v, n in stats["per_view"].items()))
if is_train:
    print(f"candidate recall vs train_pairs (S1 present, all ranks) = {stats['hit']:,}/{stats['true']:,} = {stats['hit'] / max(stats['true'], 1):.4f}")
print(f"rows written = {stats['out']:,}  per split: " + "  ".join(f"{k}={v:,}" for k, v in sorted(stats["split"].items())))
if is_train:
    print(f"label balance (written): pos={stats['lab1']:,} neg={stats['lab0']:,} pos_rate={stats['lab1'] / max(stats['out'], 1):.4f}"
          f"  sibling_conflict rows={stats['sib']:,}  is_extra rows={stats['extra']:,}")
print(f"out -> {a.out}  total {time.time() - T0:.1f}s")
