"""R65 fralign_: shared helpers — stage-1 cells, per-cell quantile map onto a reference CE distribution, KS, contested counts.
Cell = t_empty (target business_address blank) x p1 bin on the CE band (edges as research/scripts_57r/r57_cealign.py);
secondary cell adds s1_empty (reference address blank)."""
import numpy as np, polars as pl

EDGES = [0.1, 0.3, 0.5, 0.7, 0.9]
K = ["s1_id", "t_id"]


def addr_empty(records: pl.DataFrame, key: str) -> pl.DataFrame:
    return records.select(pl.col("entity_id").alias(key), (pl.col("business_address").fill_null("").str.strip_chars() == "").alias(
        "t_empty" if key == "t_id" else "s_empty"))


def with_cells(d: pl.DataFrame, secondary: bool = False) -> pl.DataFrame:
    """d needs p1, t_empty (and s_empty if secondary). Adds p1b (string bin) and cell (string key)."""
    d = d.with_columns(pl.col("p1").cut(EDGES).cast(pl.String).alias("p1b"))
    parts = [pl.when(pl.col("t_empty")).then(pl.lit("E")).otherwise(pl.lit("N")), pl.col("p1b")]
    if secondary: parts.append(pl.when(pl.col("s_empty")).then(pl.lit("|sE")).otherwise(pl.lit("|sN")))
    return d.with_columns(pl.concat_str(parts).alias("cell"))


def qmap(src: pl.DataFrame, ref: pl.DataFrame, col: str = "p_ce") -> pl.DataFrame:
    """Per-cell quantile map. src, ref: DataFrames with columns cell, <col>. Returns src with <col>_q = F_ref^-1(u),
    u = (average rank within the src cell - 0.5) / n_src_cell; F_ref^-1 = linear-interpolated quantile of the ref cell.
    Order inside a cell is preserved; src cells absent from ref are an error."""
    refd = {c: np.sort(g[col].to_numpy().astype(np.float64)) for (c,), g in ref.group_by("cell")}
    src = src.with_row_index("_r")
    outs = []
    for (c,), g in src.group_by("cell"):
        assert c in refd and len(refd[c]) >= 50, f"reference cell {c} missing or tiny ({len(refd.get(c, []))})"
        v = g[col].to_numpy().astype(np.float64)
        u = (pl.Series(v).rank("average").to_numpy() - 0.5) / len(v)
        r = refd[c]; q = np.interp(u * (len(r) - 1), np.arange(len(r)), r)   # == np.quantile(r, u, method='linear')
        outs.append(g.select("_r").with_columns(pl.Series(f"{col}_q", np.clip(q, 0.0, 1.0))))
    m = pl.concat(outs)
    return src.join(m, on="_r", how="left").sort("_r").drop("_r")


def ks(a, b):
    if len(a) == 0 or len(b) == 0: return np.nan
    g = np.linspace(0, 1, 201)
    A = np.searchsorted(np.sort(a), g, side="right") / len(a); B = np.searchsorted(np.sort(b), g, side="right") / len(b)
    return float(np.abs(A - B).max())


def weighted_ks(fr: pl.DataFrame, ref: pl.DataFrame, col: str = "p_ce"):
    """fr/ref with cell + col; weighted (by France rows) mean of per-cell KS, and the per-cell dict."""
    refd = {c: g[col].to_numpy() for (c,), g in ref.group_by("cell")}
    per = {}; tot = w = 0
    for (c,), g in fr.group_by("cell"):
        k = ks(g[col].to_numpy(), refd.get(c, np.array([]))); per[c] = (g.height, k)
        if not np.isnan(k): tot += k * g.height; w += g.height
    return tot / max(w, 1), dict(sorted(per.items()))


def contested_per_1k(sc: pl.DataFrame, t_empty: pl.DataFrame, n_s1: int, col: str = "p", thr: float = 0.2):
    """R57 definition: targets with >= 2 S1 at score >= thr, split by target address empty, per 1k S1 of the slice."""
    c = sc.filter(pl.col(col) >= thr).group_by("t_id").agg(pl.len().alias("n")).filter(pl.col("n") >= 2).join(t_empty, on="t_id", how="left")
    e = int(c["t_empty"].fill_null(False).sum()); return 1000 * e / n_s1, 1000 * (c.height - e) / n_s1


def ref_train_oof():
    """Reference = the CE semantics stage 2 was TRAINED on: production OOF CE (data/ce2_oof_v12.parquet) on TRAIN-split pairs
    (split == 'train' in data/oof_scores_v12avg.parquet; val/audit rows excluded, no labels read), cells from the train OOF
    stage-1 p and raw stripped target-address emptiness. Returns cell, p_ce."""
    ce = pl.read_parquet("data/ce2_oof_v12.parquet")
    p1 = (pl.scan_parquet("data/oof_scores_v12avg.parquet").select("s1_id", "t_id", "split", pl.col("p").alias("p1"))
          .filter((pl.col("split") == "train") & pl.col("p1").is_between(0.02, 0.98)).join(ce.lazy(), on=K).collect())
    TR = pl.read_parquet("data/train_records.parquet", columns=["entity_id", "business_address", "source"]).filter(pl.col("source") != 1)
    return with_cells(p1.join(addr_empty(TR, "t_id"), on="t_id", how="left").with_columns(pl.col("t_empty").fill_null(False))).select("cell", "p_ce")
