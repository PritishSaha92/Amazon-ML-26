#!/bin/bash
# cntall: score the US + India test CE band (798,976 pairs = output/ce2_test_v12avg.parquet rows of US/India refs) with the four full fr12
# models (fr12_ce{A,B}, fr12s1_ce{A,B}) using the PRODUCTION scorer exactly as fr12d's research/scripts_62/score_arm.sh
# (src/score_cross_encoder.py, max-len 96, default batch, --threads 3). Step 0 = scorer check on 20k French band pairs vs the existing
# fr12 French scores. Run as ONE heavy job: tools/heavy.sh -m 24G bash research/scripts_cntall/score_fr12all.sh
set -euo pipefail; cd ${BER_ROOT:?}
P=${BER_PY:?}; W=${BER_SCRATCH:?}/cntall/fr12all; MD=${BER_SCRATCH:?}/models
export HF_HOME=${BER_SCRATCH:?}/hf POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3
say() { echo "[$(date '+%F %T')] $*"; }
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
say "check: 20k French pairs with fr12_ceA"
$P src/score_cross_encoder.py --model $MD/fr12_ceA --pairs $W/fr_check20k.parquet --records data/test_records.parquet --out $W/check_fr12A.parquet --threads 3 --max-len 96 > $W/check.log 2>&1
$P - $W <<'PY'
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, polars as pl
W = sys.argv[1]; K = ["s1_id", "t_id"]
c = pl.read_parquet(f"{W}/check_fr12A.parquet", columns=K + ["p_ce"])
for ref in ((_BER_S + "/fr12/frtest_A.parquet"), (_BER_S + "/fr12d/FR12_fr_A.parquet")):
    try: r = pl.read_parquet(ref, columns=K + ["p_ce"]).rename({"p_ce": "r"})
    except Exception as e: print(ref, "unavailable", e); continue
    j = c.join(r, on=K); d = (j["p_ce"].cast(pl.Float64) - j["r"]).abs()
    print(f"vs {ref}: n {j.height}  mean|d| {d.mean():.6f}  p99 {d.quantile(.99):.5f}  max {d.max():.5f}  flips0.5 {int(((j['p_ce'] >= .5) != (j['r'] >= .5)).sum())}")
    if "fr12/frtest_A" in ref: assert d.mean() < 0.002, "scorer check failed"
PY
for M in fr12_ceA fr12_ceB fr12s1_ceA fr12s1_ceB; do
  say "score US/India with $M"
  $P src/score_cross_encoder.py --model $MD/$M --pairs $W/ui_pairs.parquet --records data/test_records.parquet --out $W/ui_$M.parquet --threads 3 --max-len 96 > $W/score_$M.log 2>&1
  tail -2 $W/score_$M.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
done
say "SCORED"
