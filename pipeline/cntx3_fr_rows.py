"""Count-aware cross-encoder, three-seed mean, French rows only (the input of research/scripts_62/guard.py CNTX3).
  $BER_SCRATCH/frnlp/frnlpm_test_all.parquet (research/scripts_70/frnlp_seedmean.py 0 1 2) -> $BER_SCRATCH/fr12d/CNTX3_fr_mean.parquet
Columns s1_id, t_id, p_ce; row order of the input.
Run from $BER_ROOT:  $BER_PY pipeline/cntx3_fr_rows.py"""
import os, polars as pl
S = os.environ["BER_SCRATCH"]; K = ["s1_id", "t_id"]
x = pl.read_parquet(f"{S}/frnlp/frnlpm_test_all.parquet").filter(pl.col("country") == "France").select(K + ["p_ce"])
os.makedirs(f"{S}/fr12d", exist_ok=True); x.write_parquet(f"{S}/fr12d/CNTX3_fr_mean.parquet"); print("CNTX3_fr_mean", x.height)
