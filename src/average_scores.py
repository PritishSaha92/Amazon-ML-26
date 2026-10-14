"""Average two per-pair score files (same pair set) — e.g. OOF scores of two stage-1 models, or their test scores.
Output = file a with p := wa*p_a + (1-wa)*p_b (and p_s2 likewise when both files carry it); all other columns come from a.
Rows are matched on (s1_id, t_id); the two files must contain exactly the same pairs (asserted).
Run: python src/average_scores.py --a data/oof_scores_v5.parquet --b data/oof_scores_v5b.parquet --out data/oof_scores_v6.parquet [--wa 0.5]
"""
import argparse, time
import numpy as np, polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--wa", type=float, default=0.5, help="weight of file a")
a = ap.parse_args(); t0 = time.time()
K = ["s1_id", "t_id"]
sa, sb = pl.read_parquet_schema(a.a), pl.read_parquet_schema(a.b)
cols = [c for c in ("p", "p_s2") if c in sa and c in sb]; assert "p" in cols, "both files need column p"
A = pl.read_parquet(a.a)
B = pl.read_parquet(a.b, columns=K + cols).rename({c: f"{c}__b" for c in cols})
assert len(A) == len(B), f"row counts differ: {len(A):,} vs {len(B):,}"
if A.select(K).equals(B.select(K)):  # fast path: same pairs in the same order (e.g. two predict.py / oof_scores.py runs on one file)
    print("rows aligned: identical (s1_id, t_id) sequence → column-wise average, no join")
    out = A.hstack(B.drop(K))
else:
    assert A.select(K).is_duplicated().sum() == 0 and B.select(K).is_duplicated().sum() == 0, "duplicate (s1_id, t_id) keys"
    out = A.join(B.with_columns(pl.lit(True).alias("__hit")), on=K, how="left", maintain_order="left")
    miss = out["__hit"].null_count(); out = out.drop("__hit")
    assert out.height == len(A) and miss == 0, f"{miss} pairs of a missing from b"
del B
for c in cols:
    corr = np.corrcoef(out[c].cast(pl.Float64).to_numpy(), out[f"{c}__b"].cast(pl.Float64).to_numpy())[0, 1]
    print(f"{c}: mean a {out[c].mean():.5f}  b {out[f'{c}__b'].mean():.5f}  corr {corr:.5f}")
    out = out.with_columns((a.wa * pl.col(c).cast(pl.Float64) + (1 - a.wa) * pl.col(f"{c}__b").cast(pl.Float64)).cast(A.schema[c]).alias(c)).drop(f"{c}__b")
out.write_parquet(a.out)
print(f"wrote {a.out}: {len(out):,} rows, averaged {cols} (wa={a.wa}) in {time.time()-t0:.0f}s")
