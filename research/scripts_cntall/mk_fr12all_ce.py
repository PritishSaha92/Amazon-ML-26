"""cntall: all-country fr12 2-seed CE file for the test band. US/India rows = mean of the 4 local scores (fr12_ce{A,B}, fr12s1_ce{A,B};
= mean of the two seed means, the FR12X2 definition), France rows = $BER_SCRATCH/fr12d/FR12X2_fr_mean.parquet VERBATIM (so the French
part is identical to submissions/v12ce2_E2_frFR12x2). Output keys/order = output/ce2_test_v12avg.parquet.
  python mk_fr12all_ce.py -> $BER_SCRATCH/cntall/fr12all/fr12x2_test_all.parquet"""
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import numpy as np, polars as pl
K = ["s1_id", "t_id"]; W = (_BER_S + "/cntall/fr12all")
base = pl.read_parquet("output/ce2_test_v12avg.parquet"); assert base.select(K).is_duplicated().sum() == 0, "duplicate keys"  # packaged copy: dynamic check
assert base.height == 1060103 or _ber_os.environ.get("BER_STRICT_MD5", "0") != "1", base.height  # packaged copy: reference-run count strict only with BER_STRICT_MD5=1
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
b = base.with_row_index("_r").join(cty, on="s1_id", how="left"); assert b["country"].null_count() == 0
ui = pl.read_parquet(f"{W}/ui_pairs.parquet")
for m in ("fr12_ceA", "fr12_ceB", "fr12s1_ceA", "fr12s1_ceB"):
    s = pl.read_parquet(f"{W}/ui_{m}.parquet", columns=K + ["p_ce"]); assert s.height == ui.height and s.join(ui, on=K, how="anti").height == 0
    ui = ui.join(s.rename({"p_ce": m}), on=K, how="left")
ui = ui.with_columns((((pl.col("fr12_ceA").cast(pl.Float64) + pl.col("fr12_ceB")) / 2 + (pl.col("fr12s1_ceA").cast(pl.Float64) + pl.col("fr12s1_ceB")) / 2) / 2).alias("p_new"))
fr = pl.read_parquet((_BER_S + "/fr12d/FR12X2_fr_mean.parquet"), columns=K + ["p_ce"]).rename({"p_ce": "p_fr"})
j = b.join(ui.select(K + ["p_new"]), on=K, how="left").join(fr, on=K, how="left").sort("_r")
isfr = j["country"] == "France"
assert int(j.filter(isfr)["p_fr"].null_count()) == 0 and int(j.filter(~isfr)["p_new"].null_count()) == 0, "coverage"
assert j.filter(isfr).height == fr.height and fr.select(K).is_duplicated().sum() == 0, "France coverage"  # packaged copy: dynamic check
assert fr.height == 261127 or _ber_os.environ.get("BER_STRICT_MD5", "0") != "1", fr.height  # packaged copy: reference-run count strict only with BER_STRICT_MD5=1
out = j.with_columns(p_ce=pl.when(isfr).then(pl.col("p_fr").cast(pl.Float64)).otherwise(pl.col("p_new")).cast(base["p_ce"].dtype))
assert out.select(K).equals(base.select(K))
_v = out["p_ce"].to_numpy(); assert np.isfinite(_v).all() and (_v >= 0).all() and (_v <= 1).all(), "p_ce not finite in [0, 1]"  # packaged copy: dynamic check
out.select(base.columns).write_parquet(f"{W}/fr12x2_test_all.parquet")
fr_eq = out.filter(isfr).select(K + ["p_ce"]).join(fr, on=K)
print("France rows identical to FR12X2_fr_mean:", bool((fr_eq["p_ce"] == fr_eq["p_fr"]).all()), fr_eq.height)
d = out.with_columns(o=base["p_ce"])
print(d.group_by("country").agg(pl.len(), pl.col("o").mean().alias("prod_mean"), pl.col("p_ce").mean().alias("fr12x2_mean"),
      pl.corr("o", "p_ce", method="spearman").alias("spearman"), ((pl.col("o") < .5) & (pl.col("p_ce") >= .5)).sum().alias("up05"),
      ((pl.col("o") >= .5) & (pl.col("p_ce") < .5)).sum().alias("down05")).sort("country"))
