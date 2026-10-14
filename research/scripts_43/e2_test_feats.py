"""TEST-side E2 features: one producer (full-fit A/B average = output/test_scores_v12avg.parquet p) over ALL test candidates per target.
Writes $BER_SCRATCH/data/e2_feats_test_v12.parquet (s1_id, t_id, c_best, c_margin, c_n50, c_rank as float32) for every test band row (p > 0.01)
and prints the val-vs-test density comparison."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys
sys.path.insert(0, (_BER_R + "/research/scripts_43"))
from common43 import *
from comp43 import target_aggs, comp_features
FE = ["c_best", "c_margin", "c_n50", "c_rank"]
g, hi = target_aggs("output/test_scores_v12avg.parquet", log=log)
b = pl.scan_parquet("output/test_scores_v12avg.parquet").select(K + ["country", "p"]).filter(pl.col("p") > S.P_MIN).collect()
c = comp_features(b.select(K + ["p"]), g, hi)
assert c.height == b.height and c.select(K).equals(b.select(K)) and c.select(K).is_duplicated().sum() == 0 and c["c_best"].null_count() == 0
c.select(K + [pl.col(f).cast(pl.Float32) for f in FE]).write_parquet((_BER_S + "/data/e2_feats_test_v12.parquet"))
log(f"wrote test E2 features: {c.height:,} band rows (p > {S.P_MIN}) of {pl.scan_parquet('output/test_scores_v12avg.parquet').select(pl.len()).collect().item():,} test pairs")
st = lambda d: d.agg(pl.len(), pl.col("c_ncand").mean(), (pl.col("c_best") >= .5).mean().alias("cbest>=.5"), (pl.col("c_rank") > 1).mean().alias("not_top"),
                     pl.col("c_n50").mean(), pl.col("c_margin").mean(), pl.col("c_best").mean())
print(st(c.with_columns(b["country"]).group_by("country")).sort("country"))
if not _ber_os.path.exists("data/r43_comp_v12_pc.parquet"): sys.exit(0)  # packaged copy: train-side comparison optional
v = pl.read_parquet("data/r43_comp_v12_pc.parquet").with_columns(pl.read_parquet("data/s2_band_v12avg.parquet", columns=["split", "country"]))
print(st(v.group_by("split", "country")).sort("split", "country"))
# stable-schema copy of the train/val/audit PC features
v.select(K + [pl.col(f).cast(pl.Float32) for f in FE]).write_parquet((_BER_S + "/data/e2_feats_v12_pc.parquet")); log(("wrote " + _BER_S + "/data/e2_feats_v12_pc.parquet"))
