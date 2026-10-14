"""Two-seed French-form cross-encoder (X2) on the French test band: float32 mean of the two seed means
(each seed mean = mean of its fold-A and fold-B models, written by research/scripts_62/score_arm.sh as <NAME>_fr_mean.parquet).
  $BER_SCRATCH/fr12d/FR12_fr_mean.parquet (seed 0) + FR12S1_fr_mean.parquet (seed 1) -> $BER_SCRATCH/fr12d/FR12X2_fr_mean.parquet
Keys and row order are those of the seed-0 file (= data/v12/fr12_frtest_pairs.parquet).
Run from $BER_ROOT:  $BER_PY pipeline/fr12x2_fr_mean.py [seed0_name] [seed1_name]"""
import os, sys, polars as pl
K = ["s1_id", "t_id"]; W = os.path.join(os.environ["BER_SCRATCH"], "fr12d")
n0, n1 = (sys.argv[1:3] + ["FR12", "FR12S1"][len(sys.argv[1:3]):])[:2]
a = pl.read_parquet(f"{W}/{n0}_fr_mean.parquet", columns=K + ["p_ce"]).rename({"p_ce": "a"})
b = pl.read_parquet(f"{W}/{n1}_fr_mean.parquet", columns=K + ["p_ce"]).rename({"p_ce": "b"})
x = a.join(b, on=K, how="inner", maintain_order="left"); assert x.height == a.height == b.height, "key mismatch"
out = x.select(K + [((pl.col("a") + pl.col("b")) / 2).cast(pl.Float32).alias("p_ce")])
out.write_parquet(f"{W}/FR12X2_fr_mean.parquet"); print("FR12X2_fr_mean", out.height, float(out["p_ce"].mean()))
