#!/bin/bash
# fr12d: score one CE arm (A,B checkpoints -> mean) on the French test CE band (261,127) and the VAL band (32,376) with the production
# scorer src/score_cross_encoder.py (max-len 96).  Usage: bash research/scripts_62/score_arm.sh <NAME> <MODEL_A> <MODEL_B>
# Out: ${BER_SCRATCH:?}/fr12d/<NAME>_fr_mean.parquet, <NAME>_val_mean.parquet (s1_id,t_id,p_ce[,p_ce_A,p_ce_B])
set -euo pipefail; cd ${BER_ROOT:?}
P=${BER_PY:?}; W=${BER_SCRATCH:?}/fr12d; N=$1; MA=$2; MB=$3
export HF_HOME=${BER_SCRATCH:?}/hf POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3
for F in A B; do eval M=\$M$F
  nice -n 10 $P src/score_cross_encoder.py --model $M --pairs data/v12/fr12_frtest_pairs.parquet --records data/test_records.parquet --out $W/${N}_fr_$F.parquet --threads 3 --max-len 96
  [ ! -f $W/val_band_pairs.parquet ] || nice -n 10 $P src/score_cross_encoder.py --model $M --pairs $W/val_band_pairs.parquet --records data/train_records.parquet --out $W/${N}_val_$F.parquet --threads 3 --max-len 96
done
$P - $W $N <<'PY'
import os, sys, polars as pl
W, N = sys.argv[1:3]; K = ["s1_id", "t_id"]
for k in (("fr", "val") if os.path.exists(f"{W}/{N}_val_A.parquet") else ("fr",)):  # packaged copy: validation band optional
    a = pl.read_parquet(f"{W}/{N}_{k}_A.parquet", columns=K + ["p_ce"]).rename({"p_ce": "p_ce_A"})
    b = pl.read_parquet(f"{W}/{N}_{k}_B.parquet", columns=K + ["p_ce"]).rename({"p_ce": "p_ce_B"})
    j = a.join(b, on=K); assert len(j) == len(a) == len(b)
    j = j.with_columns(p_ce=((pl.col("p_ce_A") + pl.col("p_ce_B")) / 2).cast(pl.Float32)); j.write_parquet(f"{W}/{N}_{k}_mean.parquet")
    print(k, len(j), float(j["p_ce"].mean()))
PY
echo "SCORED $N"
