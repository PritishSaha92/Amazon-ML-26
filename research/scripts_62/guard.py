"""fr12d guarded variant G (pre-registered in LOG.md): candidate CE for French band pairs, but for empty-target-address pairs the score
may not exceed production: min(cand, prod). Same rule on val.  python guard.py <NAME>  -> <NAME>G_fr_mean.parquet, <NAME>G_val_mean.parquet"""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, os, polars as pl
os.chdir(_BER_R); sys.path.insert(0, "research/scripts_65"); import fralign_common as F
W = (_BER_S + "/fr12d"); K = F.K; N = sys.argv[1]
for kind, recs, prodf in (("fr", "data/test_records.parquet", "output/ce2_test_v12avg.parquet"), ("val", "data/train_records.parquet", "data/ce2_oof_v12.parquet")):
    if kind == "val" and not os.path.exists(f"{W}/{N}_{kind}_mean.parquet"): continue  # packaged copy: validation files optional
    c = pl.read_parquet(f"{W}/{N}_{kind}_mean.parquet", columns=K + ["p_ce"])
    te = F.addr_empty(pl.read_parquet(recs, columns=["entity_id", "business_address"]), "t_id")
    pr = pl.read_parquet(prodf).unique(subset=K).rename({"p_ce": "prod"})
    d = c.join(te, on="t_id", how="left").join(pr, on=K, how="left"); assert d["t_empty"].null_count() == 0 and d["prod"].null_count() == 0 and len(d) == len(c)
    d = d.with_columns(p_ce=pl.when(pl.col("t_empty")).then(pl.min_horizontal("p_ce", "prod")).otherwise(pl.col("p_ce")).cast(pl.Float32))
    d.select(K + ["p_ce"]).write_parquet(f"{W}/{N}G_{kind}_mean.parquet")
    print(kind, len(d), "t_empty", int(d["t_empty"].sum()), "capped (cand > prod on t_empty)", int((d["t_empty"] & (c["p_ce"] > d["prod"])).sum()))
