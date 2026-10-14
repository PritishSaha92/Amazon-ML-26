"""Build count-aware training sets from ce_train_hard_v12f2{F}_frmix.parquet (node).
Counts come from the ORIGINAL record (s1_id / t_id are original train ids; transplanting changes surface text only).
Usage: python frnlp_prep_train.py  (writes cnt tables + frnlp_train_{A,B}.parquet)"""
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import json, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frnlp_common import count_table, prefix

D = (_BER_S + "/data/v12/cefold2"); O = (_BER_S + "/frnlp"); T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
ct = count_table("data/train_records.parquet", f"{O}/namecore_train.parquet"); ct.write_parquet(f"{O}/cnt_train.parquet")
log(f"cnt_train {len(ct):,}; bucket x country x source:\n{ct.group_by('country','source','bucket').len().sort('country','source','bucket')}")
ce = count_table("data/test_records.parquet", f"{O}/namecore_test.parquet"); ce.write_parquet(f"{O}/cnt_test.parquet")
log(f"cnt_test {len(ce):,}:\n{ce.group_by('country','source','bucket').len().sort('country','source','bucket')}")
m = ct.select("entity_id", "bucket")
stats = {}
for F in "AB":
    X = pl.read_parquet(f"{D}/ce_train_hard_v12f2{F}_frmix.parquet")
    n0 = len(X)
    X = (X.join(m.rename({"entity_id": "s1_id", "bucket": "bs"}), on="s1_id", how="left")
          .join(m.rename({"entity_id": "t_id", "bucket": "bt"}), on="t_id", how="left"))
    assert len(X) == n0 and X["bs"].null_count() == 0 and X["bt"].null_count() == 0, "count join failed"
    X = X.with_columns(a=prefix("bs") + pl.col("a"), b=prefix("bt") + pl.col("b"))
    X.write_parquet(f"{O}/frnlp_train_{F}.parquet")
    stats[F] = dict(rows=n0, bs=dict(X.group_by("bs").len().iter_rows()), bt=dict(X.group_by("bt").len().iter_rows()),
                    bs_by_label=X.group_by("label", "bs").len().sort("label", "bs").rows())
    log(f"fold {F}: {n0:,} rows written; examples:\n" + "\n".join(f"  {r['label']} {r['pool']} A: {r['a']}\n        B: {r['b']}" for r in X.sample(6, seed=0).iter_rows(named=True)))
    del X
json.dump(stats, open(f"{O}/frnlp_train_stats.json", "w"), indent=1, default=str)
log("DONE")
