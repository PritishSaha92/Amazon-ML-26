"""Per-target competition features from stage-1 p over a reference universe (any score parquet with t_id, p; all rows compete).
For a band row (s1, t, p) over the other refs r != s1 with t among their candidates:
  c_best max p_r (0 if none) | c_margin p - c_best | c_n50/#p_r>=.5 | c_n10 #p_r>=.1 | c_rank 1+#p_r>p | c_ncand #r | c_sum sum p_r"""
import time
import polars as pl, pyarrow.parquet as pq
K = ["s1_id", "t_id"]; F = ["c_best", "c_margin", "c_n50", "c_n10", "c_rank", "c_ncand", "c_sum"]
H = lambda c: pl.col(c).hash(seed=43)
EPS = 1e-7

def target_aggs(universe, batch=10_000_000, log=print):
    t0 = time.time(); parts = []; hi = []
    for rb in pq.ParquetFile(universe).iter_batches(batch_size=batch, columns=["t_id", "p"]):
        x = pl.from_arrow(rb).select(H("t_id").alias("th"), pl.col("p").cast(pl.Float64))
        parts.append(x.group_by("th").agg(pl.col("p").top_k(2).alias("top"), (pl.col("p") >= 0.5).sum().alias("n50"),
                                         (pl.col("p") >= 0.1).sum().alias("n10"), pl.len().alias("n"), pl.col("p").sum().alias("s")))
        hi.append(x.filter(pl.col("p") > 0.005))   # for ranks of band rows (band p > 0.01) only higher-p rows matter
    g = pl.concat(parts); del parts
    g = g.group_by("th").agg(pl.col("top").list.explode().top_k(2).alias("top"), pl.col("n50").sum(), pl.col("n10").sum(), pl.col("n").sum(), pl.col("s").sum())
    g = g.with_columns(pl.col("top").list.sort(descending=True)).with_columns(
        pl.col("top").list.get(0, null_on_oob=True).alias("top1"), pl.col("top").list.get(1, null_on_oob=True).fill_null(0.0).alias("top2")).drop("top")
    log(f"  universe {universe}: {len(g):,} targets ({time.time()-t0:.0f}s)")
    return g, pl.concat(hi).rename({"p": "pr"})

def comp_features(band, g, hi, self_in_universe=True):
    """band: K + p (p > 0.005, rows of the universe). Returns band K + F in band order."""
    b = band.select(K + [pl.col("p").cast(pl.Float64)]).with_columns(H("t_id").alias("th"), pl.int_range(pl.len()).alias("row"))
    b = b.join(g, on="th", how="left", maintain_order="left")
    if not self_in_universe:   # rivals-only universe: nothing to subtract; targets absent from it have no rivals
        b = b.with_columns(pl.col("top1").fill_null(0.0), pl.col("top2").fill_null(0.0), pl.col(["n50", "n10", "n"]).fill_null(0), pl.col("s").fill_null(0.0))
        b = b.with_columns(c_best=pl.col("top1"), c_n50=pl.col("n50"), c_n10=pl.col("n10"), c_ncand=pl.col("n"), c_sum=pl.col("s")).with_columns(c_margin=pl.col("p") - pl.col("c_best"))
    else:
      miss = b["n"].null_count(); assert miss == 0, f"{miss} band rows whose target is not in the universe"
      b = b.with_columns(self_top=(pl.col("p") >= pl.col("top1") - EPS))
      b = b.with_columns(c_best=pl.when(pl.col("self_top")).then(pl.col("top2")).otherwise(pl.col("top1")),
                       c_n50=pl.col("n50") - (pl.col("p") >= 0.5).cast(pl.UInt32), c_n10=pl.col("n10") - (pl.col("p") >= 0.1).cast(pl.UInt32),
                       c_ncand=pl.col("n") - 1, c_sum=pl.col("s") - pl.col("p")).with_columns(c_margin=pl.col("p") - pl.col("c_best"))
    k = (b.select("th", "p", "row").join(hi.join(b.select("th").unique(), on="th", how="semi"), on="th", how="inner")
         .filter(pl.col("pr") > pl.col("p") + EPS).group_by("row").agg(pl.len().alias("k")))
    b = b.join(k, on="row", how="left", maintain_order="left").with_columns(c_rank=(1 + pl.col("k").fill_null(0)).cast(pl.UInt32))
    return b.select(K + F)
