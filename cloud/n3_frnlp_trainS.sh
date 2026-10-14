#!/bin/bash
# Node 3 frnlp count-aware CE: prep (once) then train fold F on GPU G = fr12h recipe with count-token text, max-len 112.
# Usage: bash cloud/n3_frnlp_trainS.sh F G SEED (no prep) -> models/frnlp_cnt<SEED>_ceF
cd ${BER_ROOT:?} || exit 1
F=$1; G=$2; SD=$3; P=$HOME/v12env/bin/python; O=${BER_SCRATCH:?}/frnlp; T=frnlp_cnt${SD}_$F; OUT=models/frnlp_cnt${SD}_ce$F
export HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=true POLARS_MAX_THREADS=16 OMP_NUM_THREADS=8 RAYON_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=$G
E5=$HOME/hf/hub/models--intfloat--multilingual-e5-base/snapshots/d128750597153bb5987e10b1c3493a34e5a4502a
[ -f $E5/model.safetensors ] || { echo "missing $E5: run cloud/env_setup.sh (pinned model download)"; exit 1; }  # packaged copy
say() { echo "[$(date '+%F %T')] [$T gpu$G] $*"; }
while [ ! -f $O/.prep_done ]; do sleep 10; done
say "START train $T"
$P research/scripts_55/train_ce_text_swap_ptok.py --data $O/frnlp_train_$F.parquet --drop-pools syn --pos-dup 1 \
   --tag $T --batch 96 --threads 8 --lr 3e-5 --max-len 112 --epochs 1 --seed $SD --limit 10350000 --tok-workers 24 --tok-check 50000 --model $E5 || { say FAILED; exit 1; }
[ -f models/ce_$T/model.safetensors ] || { say "no model"; exit 1; }
mv models/ce_$T $OUT && md5sum $OUT/*
touch logs/n3/.done_$T; say "DONE train $T"
