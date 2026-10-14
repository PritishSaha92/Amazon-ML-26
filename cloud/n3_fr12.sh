#!/bin/bash
# Node 3: fr12 training = research/scripts_55/run_train.sh recipe on the node (paths adapted, single GPU per run).
# Usage: bash cloud/n3_fr12.sh FOLD GPU SEED   (SEED 0 -> tag fr12_F -> models/fr12_ceF ; SEED 1 -> tag fr12s1_F -> models/fr12s1_ceF)
# Fold B: if the _frmix set is missing, the first caller (seed 0) rebuilds it with transplant_v12.py (NPROC=100); others wait.
cd ${BER_ROOT:?} || exit 1
F=$1; G=$2; S=${3:-0}; P=$HOME/v12env/bin/python; D=${BER_SCRATCH:?}/data/v12/cefold2
export HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=true POLARS_MAX_THREADS=16 OMP_NUM_THREADS=8 RAYON_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=$G
E5=$HOME/hf/hub/models--intfloat--multilingual-e5-base/snapshots/d128750597153bb5987e10b1c3493a34e5a4502a
[ -f $E5/model.safetensors ] || { echo "missing $E5: run cloud/env_setup.sh (pinned model download)"; exit 1; }  # packaged copy
[ $S = 0 ] && T=fr12_$F && OUT=models/fr12_ce$F || { T=fr12s1_$F; OUT=models/fr12s1_ce$F; }
say() { echo "[$(date '+%F %T')] [$T gpu$G] $*"; }
if [ $F = B ] && [ $S = 0 ] && [ ! -f $D/fr12_transplant_checks_B.json ]; then
  while [ ! -f $D/ce_train_hard_v12f2B.parquet ]; do sleep 10; done
  echo 3861628113df0f967029ec40de2c727e $D/ce_train_hard_v12f2B.parquet | md5sum -c || { [ "${BER_STRICT_MD5:-0}" = 1 ] && { say "B input md5 mismatch"; exit 1; }; say "WARNING: B input md5 differs from the reference"; }  # packaged copy
  say "START transplant B (NPROC 100)"
  NPROC=100 POLARS_MAX_THREADS=64 $P research/scripts_55/transplant_v12.py B > logs/fr12/transplant_B.log 2>&1 || { say "transplant B FAILED"; exit 1; }
  say "DONE transplant B"; python3 -c "import json;d=json.load(open('$D/fr12_transplant_checks_B.json'));print({k:d[k] for k in ('rows','frmix_rows','exposure_after_pos_dup1','newly_identical_negatives','newly_identical_positives','sha256_frmix','seconds')})"
fi
while [ ! -f $D/fr12_transplant_checks_$F.json ] || [ ! -f $D/ce_train_hard_v12f2${F}_frmix.parquet ]; do sleep 20; done
if [ $F = A ]; then echo 23d8e7b036053c1774940b56fb21c6fc $D/ce_train_hard_v12f2A_frmix.parquet | md5sum -c || { [ "${BER_STRICT_MD5:-0}" = 1 ] && { say "A frmix md5 mismatch"; exit 1; }; say "WARNING: A frmix md5 differs from the reference"; }; fi  # packaged copy
say "START train $T seed $S"
$P research/scripts_44/train_ce_text_swap.py --data $D/ce_train_hard_v12f2${F}_frmix.parquet --drop-pools syn --pos-dup 1 \
   --tag $T --batch 96 --threads 8 --lr 3e-5 --max-len 96 --epochs 1 --seed $S --model $E5 || { say FAILED; exit 1; }
[ -f models/ce_$T/model.safetensors ] || { say "no model"; exit 1; }
mv models/ce_$T $OUT && md5sum $OUT/model.safetensors && sha256sum $OUT/model.safetensors
touch logs/n3/.done_$T; say "DONE train $T"
