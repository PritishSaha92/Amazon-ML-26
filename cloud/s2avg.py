"""3-seed stage-2 average (protocol: average the PAIR probabilities of seeds 0,1,2 before any decoding).
val/audit: python cloud/s2avg.py --tag T
   reads output/{val,audit}_scores_s2_{T,T_seed1,T_seed2}.parquet (column p_s2) -> output/{val,audit}_scores_s2_T_s3avg.parquet (p_s2 averaged;
   v12_eval.py decodes p_s2).
test:      python cloud/s2avg.py --test --tag T
   reads output/test_scores_{T,T_seed1,T_seed2}_s2.parquet (stage2_apply output, column p = stage-2 prob where applied) ->
   output/test_scores_T_s3avg_s2.parquet with the AVERAGE IN COLUMN `p` and NO `p_s2` column (predict.py / chain_rules_on.sh decode `p`).
Row sets of the three files must be identical."""
import argparse, polars as pl
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--splits", default="val,audit")
ap.add_argument("--test", action="store_true"); a = ap.parse_args()
K = ["s1_id", "t_id"]
def avg(fs, col, out):
    base = pl.read_parquet(fs[0]); assert col in base.columns, (fs[0], base.columns); res = base
    for i, f in enumerate(fs[1:], 1):
        o = pl.read_parquet(f, columns=K + [col]).rename({col: f"x{i}"})
        assert len(o) == len(base) and o.select(K).is_duplicated().sum() == 0, f
        res = res.join(o, on=K, how="left"); assert res[f"x{i}"].null_count() == 0, f"{f}: row sets differ"
    res = res.with_columns(((pl.col(col) + pl.col("x1") + pl.col("x2")) / 3).cast(base[col].dtype).alias(col)).drop("x1", "x2")
    if a.test:
        res = res.drop([c for c in ("p_s2",) if c in res.columns]); assert "p_s2" not in res.columns and "p" in res.columns
    res.write_parquet(out); print(f"wrote {out} rows {len(res):,} (mean {col} {res[col].mean():.6f})")
if a.test:
    avg([f"output/test_scores_{a.tag}{x}_s2.parquet" for x in ("", "_seed1", "_seed2")], "p", f"output/test_scores_{a.tag}_s3avg_s2.parquet")
else:
    for s in a.splits.split(","):
        avg([f"output/{s}_scores_s2_{a.tag}{x}.parquet" for x in ("", "_seed1", "_seed2")], "p_s2", f"output/{s}_scores_s2_{a.tag}_s3avg.parquet")
