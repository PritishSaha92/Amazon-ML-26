#!/bin/bash
# Node 3 lean scoring for an extra seed: OOF keys (oof_arm routing) + full test band, all 4 jobs in parallel on GPU_A/GPU_B.
# Usage: bash cloud/n3_frnlp_seedscore.sh SEED GPU_A GPU_B  -> FRNLP<S>_oof.parquet, frnlp<S>_test_all.parquet
cd ${BER_ROOT:?} || exit 1
S=$1; GA=$2; GB=$3; P=$HOME/v12env/bin/python; O=${BER_SCRATCH:?}/frnlp; MA=models/frnlp_cnt${S}_ceA; MB=models/frnlp_cnt${S}_ceB; N=FRNLP$S
export HF_HOME=$HOME/hf POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8
say() { echo "[$(date '+%F %T')] [frnlp seedscore $S] $*"; }
while [ ! -f $MA/model.safetensors ] || [ ! -f $MB/model.safetensors ]; do sleep 20; done
say START
bash cloud/n3_frnlp_testall.sh $MA $MB frnlp${S}_test_all $GA $GB > logs/n3/frnlp${S}_testall.out 2>&1 &
X="research/scripts_70/frnlp_score.py --max-len 112 --pairs $O/../fr12d/oof_keys.parquet --records data/train_records.parquet --cnt $O/cnt_train.parquet"
CUDA_VISIBLE_DEVICES=$GA $P $X --models $MA --out $O/${N}_oofraw_A.parquet > logs/n3/${N}_s_oofA.out 2>&1 || say "FAILED oofA" &
CUDA_VISIBLE_DEVICES=$GB $P $X --models $MB --out $O/${N}_oofraw_B.parquet > logs/n3/${N}_s_oofB.out 2>&1 || say "FAILED oofB" &
wait
$P - $N <<'PY'
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import polars as pl, sys; N = sys.argv[1]
O = (_BER_S + "/frnlp"); K = ["s1_id", "t_id"]
a = pl.read_parquet(f"{O}/{N}_oofraw_A.parquet").select(K + [pl.col("p_ce").alias("a")])
b = pl.read_parquet(f"{O}/{N}_oofraw_B.parquet").select(K + [pl.col("p_ce").alias("b")])
fo = pl.read_parquet("data/v12/cefold2/folds.parquet"); j = a.join(b, on=K).join(fo, on="s1_id", how="left"); assert len(j) == len(a) == len(b)
j = j.with_columns(p_ce=pl.when(pl.col("fold") == "A").then(pl.col("b")).when(pl.col("fold") == "B").then(pl.col("a")).otherwise((pl.col("a") + pl.col("b")) / 2).cast(pl.Float32))
j.select(K + ["p_ce"]).write_parquet(f"{O}/{N}_oof.parquet"); print(f"wrote {N}_oof", len(j))
PY
md5sum $O/${N}_oof.parquet $O/frnlp${S}_test_all.parquet; touch logs/n3/.done_frnlp_seedscore$S; say DONE
