"""Key sets of the cross-encoder test band, split by country, in the row order of output/ce2_test_v12avg.parquet
(the production CE test file: 1,060,103 pairs with stage-1 probability in [0.02, 0.98]).
  French pairs      -> data/v12/fr12_frtest_pairs.parquet                    (261,127 keys; scored by research/scripts_62/score_arm.sh)
  US / India pairs  -> $BER_SCRATCH/cntall/fr12all/ui_pairs.parquet          (798,976 keys; scored by research/scripts_cntall/score_fr12all.sh)
Run from $BER_ROOT:  $BER_PY pipeline/ce_band_keys.py"""
import os, polars as pl
K = ["s1_id", "t_id"]; S = os.environ["BER_SCRATCH"]
ce = pl.read_parquet("output/ce2_test_v12avg.parquet", columns=K)
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
j = ce.join(cty, on="s1_id", how="left", maintain_order="left")
assert j.height == ce.height and j["country"].null_count() == 0
os.makedirs("data/v12", exist_ok=True); os.makedirs(f"{S}/cntall/fr12all", exist_ok=True)
fr = j.filter(pl.col("country") == "France").select(K); fr.write_parquet("data/v12/fr12_frtest_pairs.parquet")
ui = j.filter(pl.col("country") != "France").select(K); ui.write_parquet(f"{S}/cntall/fr12all/ui_pairs.parquet")
print("France band", fr.height, "US/India band", ui.height)
