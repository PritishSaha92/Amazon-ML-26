"""rmiss context patcher: adds channel pairs to an EXISTING build_pairs output without rebuilding everything.
Exact re-implementation of src/build_pairs.py's per-pair base columns (for the new pairs) and its per-S1 / per-target context
(for every row whose S1 received a new pair, and every row on a target that received a new pair). Other rows are unchanged
by construction (their S1's row set and their target's owner set are unchanged).
  new pair in the 6-view union (keep-rule drop): base columns from its union view rows (score max / rank min per view)
  pure route pair: one 7th-view row 'rname' rank 98 -> all 6 standard views missing, n_views 1, best_rank 98, nscore 1-98/99
API: base_cols(new_union_rows, pure_pairs, K) ; patch(rows_old, rows_new, R1, T1)"""
import polars as pl
STD = ["full", "addr", "name", "dense", "rdense", "rfull"]; RANK_MISSING = 99
KEYS = ["s1_id", "t_id"]

def base_cols(union_rows: pl.DataFrame, pure: pl.DataFrame, K: dict) -> pl.DataFrame:
    """union_rows: s1_id,t_id,view,score,rank rows of the in-union new pairs; pure: s1_id,t_id of route-only pairs.
    K: per-view K incl. 'rname' (99). Returns s1_id,t_id + per-view score/rank/nscore (+rname) + n_views, best_rank, max_score, sum_score."""
    views = sorted(K)
    rn = pure.select(KEYS).with_columns(view=pl.lit("rname"), score=pl.lit(0.0, pl.Float32), rank=pl.lit(98, pl.Int16))
    x = pl.concat([union_rows.select(KEYS + [pl.col("view").cast(pl.String), pl.col("score").cast(pl.Float32), pl.col("rank").cast(pl.Int16)]), rn])
    agg = []
    for v in views:
        f = pl.col("view") == v
        agg += [pl.col("score").filter(f).max().fill_null(0.0).alias(f"{v}_score"), pl.col("rank").filter(f).min().fill_null(RANK_MISSING).cast(pl.Int16).alias(f"{v}_rank")]
    df = x.group_by(KEYS).agg(*agg, n_views=pl.col("view").n_unique().cast(pl.Int8), best_rank=pl.col("rank").min().cast(pl.Int16))
    df = df.with_columns([pl.when(pl.col(f"{v}_rank") == RANK_MISSING).then(0.0).otherwise(1.0 - pl.col(f"{v}_rank") / K[v]).cast(pl.Float32).alias(f"{v}_nscore") for v in views])
    nsc = [f"{v}_nscore" for v in views]
    return df.with_columns(max_score=pl.max_horizontal(nsc), sum_score=pl.sum_horizontal(nsc).cast(pl.Float32))

def s1_context(df: pl.DataFrame, views) -> pl.DataFrame:
    """df = ALL rows of the affected S1s (old + new). Recomputes the per-S1 columns exactly as build_pairs."""
    df = df.with_columns(s1_n_cands=pl.len().over("s1_id").cast(pl.Int32), s1_max_score=pl.col("max_score").max().over("s1_id"),
                         s1_rank_of_pair=(pl.col("max_score").rank("min", descending=True).over("s1_id") - 1).cast(pl.Int32)
                         ).with_columns(s1_gap_to_best=pl.col("s1_max_score") - pl.col("max_score"))
    return df.with_columns([(pl.col(f"{v}_score").max().over("s1_id") - pl.col(f"{v}_score")).alias(f"s1_gap_{v}") for v in views])

def t_context(df: pl.DataFrame, views) -> pl.DataFrame:
    """df = ALL rows (all owners, old + new) of the affected targets. Recomputes the per-target columns exactly as build_pairs."""
    ts = df.group_by("t_id").agg(t_n_owners=pl.len().cast(pl.Int32), t_top1=pl.col("max_score").max(),
                                 t_top2=pl.col("max_score").top_k(2).min().filter(pl.len() > 1).first())
    df = df.drop([c for c in ("t_n_owners",) if c in df.columns]).join(ts, on="t_id", how="left").with_columns(
        t_rank_of_pair=(pl.col("max_score").rank("min", descending=True).over("t_id") - 1).cast(pl.Int32),
        t_best_other_score=pl.when(pl.col("t_n_owners") == 1).then(0.0).when(pl.col("max_score") >= pl.col("t_top1")).then(pl.col("t_top2"))
                           .otherwise(pl.col("t_top1")).cast(pl.Float32)).with_columns(t_margin=pl.col("max_score") - pl.col("t_best_other_score")).drop("t_top1", "t_top2")
    for v in views:
        sc = f"{v}_score"
        tv = df.group_by("t_id").agg(_n=pl.len(), _t1=pl.col(sc).max(), _t2=pl.col(sc).top_k(2).min())
        df = df.join(tv, on="t_id", how="left").with_columns(
            (pl.col(sc).rank("min", descending=True).over("t_id") - 1).cast(pl.Int32).alias(f"t_rank_{v}"),
            pl.when(pl.col("_n") == 1).then(0.0).when(pl.col(sc) >= pl.col("_t1")).then(pl.col("_t2")).otherwise(pl.col("_t1")).cast(pl.Float32).alias(f"t_best_other_{v}"),
        ).with_columns((pl.col(sc) - pl.col(f"t_best_other_{v}")).alias(f"t_margin_{v}")).drop("_n", "_t1", "_t2")
    return df

S1C = lambda views: ["s1_n_cands", "s1_max_score", "s1_rank_of_pair", "s1_gap_to_best"] + [f"s1_gap_{v}" for v in views]
TC = lambda views: ["t_n_owners", "t_rank_of_pair", "t_best_other_score", "t_margin"] + [c for v in views for c in (f"t_rank_{v}", f"t_best_other_{v}", f"t_margin_{v}")]

def patch(r1_rows: pl.DataFrame, t1_rows: pl.DataFrame, new: pl.DataFrame, views=STD) -> pl.DataFrame:
    """r1_rows: all OLD rows of S1s that get a new pair; t1_rows: all OLD rows on targets that get a new pair (any S1);
    new: base columns of the new pairs. Returns patched rows = new rows + every old row in r1_rows ∪ t1_rows, full context.
    Per-S1 columns: recomputed for S1s in new (over r1_rows + new). Per-target columns: recomputed for targets in new
    (over t1_rows + new); for old rows on other targets they keep their stored values."""
    base = [c for c in new.columns]
    allrows = pl.concat([r1_rows, t1_rows], how="diagonal_relaxed").unique(subset=KEYS, keep="first")
    allrows = pl.concat([allrows.with_columns(pl.lit(False).alias("_new")), new.with_columns(pl.lit(True).alias("_new"))], how="diagonal_relaxed")
    R1 = new.select("s1_id").unique(); T1 = new.select("t_id").unique()
    a = allrows.filter(pl.col("s1_id").is_in(R1["s1_id"].implode()))
    a = s1_context(a, views)
    b = allrows.filter(pl.col("t_id").is_in(T1["t_id"].implode()))
    b = t_context(b, views)
    # merge: start from allrows; overwrite S1 columns from a, target columns from b
    out = allrows.drop(S1C(views) + TC(views), strict=False)
    s1v = allrows.select(KEYS + S1C(views)).join(a.select(KEYS + S1C(views)), on=KEYS, how="left", suffix="_n")
    tv_ = allrows.select(KEYS + TC(views)).join(b.select(KEYS + TC(views)), on=KEYS, how="left", suffix="_n")
    s1v = s1v.select(KEYS + [pl.coalesce(f"{c}_n", c).alias(c) for c in S1C(views)])
    tv_ = tv_.select(KEYS + [pl.coalesce(f"{c}_n", c).alias(c) for c in TC(views)])
    out = out.join(s1v, on=KEYS, how="left").join(tv_, on=KEYS, how="left")
    assert out.filter(pl.col("_new"))["t_n_owners"].null_count() == 0 and out.filter(pl.col("_new"))["s1_n_cands"].null_count() == 0
    return out
