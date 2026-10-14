#!/bin/bash
# Node 3: full test CE band (all countries; keys = output/ce2_test_v12avg.parquet) with a count-aware CE pair.
# Usage: bash cloud/n3_frnlp_testall.sh MODEL_A MODEL_B OUTNAME GPU_A GPU_B
cd ${BER_ROOT:?} || exit 1
P=$HOME/v12env/bin/python; O=${BER_SCRATCH:?}/frnlp; MA=$1; MB=$2; N=$3; GA=$4; GB=$5
export HF_HOME=$HOME/hf POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8
say() { echo "[$(date '+%F %T')] [frnlp testall $N] $*"; }
while [ ! -f $MA/model.safetensors ] || [ ! -f $MB/model.safetensors ]; do sleep 30; done
S="research/scripts_70/frnlp_score.py --max-len 112 --pairs output/ce2_test_v12avg.parquet --records data/test_records.parquet --cnt $O/cnt_test.parquet"
CUDA_VISIBLE_DEVICES=$GA $P $S --models $MA --out $O/${N}_raw_A.parquet > logs/n3/${N}_A.out 2>&1 || say "FAILED A" &
CUDA_VISIBLE_DEVICES=$GB $P $S --models $MB --out $O/${N}_raw_B.parquet > logs/n3/${N}_B.out 2>&1 || say "FAILED B" &
wait
$P - $N <<'PY'
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, polars as pl
N = sys.argv[1]; O = (_BER_S + "/frnlp"); K = ["s1_id", "t_id"]
keys = pl.read_parquet("output/ce2_test_v12avg.parquet", columns=K)
a = pl.read_parquet(f"{O}/{N}_raw_A.parquet").select(K + [pl.col("p_ce").alias("p_0"), "trunc", "bs", "bt", "ns", "nt"])
b = pl.read_parquet(f"{O}/{N}_raw_B.parquet").select(K + [pl.col("p_ce").alias("p_1")])
c = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
j = keys.join(a, on=K, how="left").join(b, on=K, how="left").join(c, on="s1_id", how="left")
assert len(j) == len(keys) and j["p_0"].null_count() == 0 and j["p_1"].null_count() == 0 and j["country"].null_count() == 0, "coverage"
j = j.with_columns(p_ce=((pl.col("p_0") + pl.col("p_1")) / 2).cast(pl.Float32))
j.select(K + ["p_ce", "p_0", "p_1", "country", "trunc", "bs", "bt", "ns", "nt"]).write_parquet(f"{O}/{N}.parquet")
print("rows", len(j), "keys", len(keys), "unique keys", keys.unique().height); print(j.group_by("country").agg(pl.len(), pl.col("p_ce").mean(), pl.col("trunc").mean()))
PY
md5sum $O/$N.parquet; touch logs/n3/.done_$N; say DONE
