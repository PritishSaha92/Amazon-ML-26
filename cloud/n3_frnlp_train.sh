#!/bin/bash
# Node 3 frnlp count-aware CE: prep (once) then train fold F on GPU G = fr12h recipe with count-token text, max-len 112.
# Usage: bash cloud/n3_frnlp_train.sh F G      -> models/frnlp_cnt_ceF
cd ${BER_ROOT:?} || exit 1
F=$1; G=$2; P=$HOME/v12env/bin/python; O=${BER_SCRATCH:?}/frnlp; T=frnlp_cnt_$F; OUT=models/frnlp_cnt_ce$F
export HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=true POLARS_MAX_THREADS=16 OMP_NUM_THREADS=8 RAYON_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=$G
E5=$HOME/hf/hub/models--intfloat--multilingual-e5-base/snapshots/d128750597153bb5987e10b1c3493a34e5a4502a
[ -f $E5/model.safetensors ] || { echo "missing $E5: run cloud/env_setup.sh (pinned model download)"; exit 1; }  # packaged copy
say() { echo "[$(date '+%F %T')] [$T gpu$G] $*"; }
if [ $F = A ]; then
  say "START namecore"; NPROC=100 $P research/scripts_70/frnlp_namecore.py > logs/n3/frnlp_namecore.out 2>&1 || { say "namecore FAILED"; exit 1; }
  say "START prep"; $P research/scripts_70/frnlp_prep_train.py > logs/n3/frnlp_prep.out 2>&1 || { say "prep FAILED"; exit 1; }
  touch $O/.prep_done; say "DONE prep"
fi
while [ ! -f $O/.prep_done ]; do sleep 10; done
say "START train $T"
$P research/scripts_55/train_ce_text_swap_ptok.py --data $O/frnlp_train_$F.parquet --drop-pools syn --pos-dup 1 \
   --tag $T --batch 96 --threads 8 --lr 3e-5 --max-len 112 --epochs 1 --seed 0 --limit 10350000 --tok-workers 24 --tok-check 50000 --model $E5 || { say FAILED; exit 1; }
[ -f models/ce_$T/model.safetensors ] || { say "no model"; exit 1; }
mv models/ce_$T $OUT && md5sum $OUT/*
touch logs/n3/.done_$T; say "DONE train $T"
