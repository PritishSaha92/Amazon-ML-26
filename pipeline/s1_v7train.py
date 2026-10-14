"""Training-split references of the 400,000-reference sample (input of the cloud stage-1 chain's 'complement' step, which
builds pairs for all OTHER training references).
  data/pairs_train_v7_feat.parquet + data/split.parquet -> data/v12/s1_v7train.parquet (column s1_id, sorted)
Run from $BER_ROOT:  $BER_PY pipeline/s1_v7train.py"""
import os, polars as pl
sp = pl.read_parquet("data/split.parquet").filter(pl.col("split") == "train").select("s1_id")
s = pl.scan_parquet("data/pairs_train_v7_feat.parquet").select("s1_id").unique().collect().join(sp, on="s1_id", how="semi").sort("s1_id")
os.makedirs("data/v12", exist_ok=True); s.write_parquet("data/v12/s1_v7train.parquet"); print("train references in the sample", s.height)
