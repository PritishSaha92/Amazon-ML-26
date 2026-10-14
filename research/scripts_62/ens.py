"""fr12d frENS (pre-registered in LOG.md): 0.5 * fr12 2-seed mean + 0.5 * guarded count-aware 3-seed mean. Writes ENS_fr_mean, ENS_val_mean, ENS_oof."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, polars as pl
os.chdir(_BER_R); sys.path.insert(0, "research/scripts_65"); import fralign_common as F
W = (_BER_S + "/fr12d"); K = F.K
def mix(a, b, out):
    x = pl.read_parquet(a, columns=K + ["p_ce"]).rename({"p_ce": "a"}).join(pl.read_parquet(b, columns=K + ["p_ce"]).rename({"p_ce": "b"}), on=K)
    x.select(K + [((pl.col("a") + pl.col("b")) / 2).cast(pl.Float32).alias("p_ce")]).write_parquet(out); print(out, len(x))
mix(f"{W}/FR12X2_fr_mean.parquet", f"{W}/CNTX3G_fr_mean.parquet", f"{W}/ENS_fr_mean.parquet")
if not os.path.exists(f"{W}/FR12X2_val_mean.parquet"): sys.exit(0)  # packaged copy: the validation / OOF blends below are optional
mix(f"{W}/FR12X2_val_mean.parquet", f"{W}/CNTX3G_val_mean.parquet", f"{W}/ENS_val_mean.parquet")
# OOF: guard the count-aware OOF against production OOF on empty-target pairs (train records), then mix
c = pl.read_parquet((_BER_S + "/frnlp/FRNLPm_oof.parquet"), columns=K + ["p_ce"])
te = F.addr_empty(pl.read_parquet("data/train_records.parquet", columns=["entity_id", "business_address"]), "t_id")
pr = pl.read_parquet("data/ce2_oof_v12.parquet").unique(subset=K).rename({"p_ce": "prod"})
g = c.join(te, on="t_id", how="left").join(pr, on=K, how="left"); assert g["t_empty"].null_count() == 0 and g["prod"].null_count() == 0 and len(g) == len(c)
g = g.with_columns(p_ce=pl.when(pl.col("t_empty")).then(pl.min_horizontal("p_ce", "prod")).otherwise(pl.col("p_ce")).cast(pl.Float32))
g.select(K + ["p_ce"]).write_parquet(f"{W}/CNTX3G_oof.parquet"); print("CNTX3G_oof", len(g))
mix(f"{W}/FR12X2_oof.parquet", f"{W}/CNTX3G_oof.parquet", f"{W}/ENS_oof.parquet")
