#!/bin/bash
# 3-seed clean control: after the chain's stage 2 (seed 0 = v12ce), fit v12ce_seed1/_seed2 with the IDENTICAL recipe, average p_s2 -> v12ce_s3avg,
# and read it on val AND audit (clean control; pre-registered scale gate). Logs in logs/v12/.
cd "$(dirname "$0")/.." || exit 1; P=$HOME/v12env/bin/python; L=logs/v12
say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
until [ -f $L/.done_stage2 ]; do sleep 60; done
S2="src/stage2.py --pairs data/pairs_train_v12_feat.parquet --scores data/oof_scores_v12avg.parquet --model models/lgbm_v12avg --band-cache data/s2_band_v12avg.parquet --ce data/ce_oof_v12.parquet"
pids=""; for SD in 1 2; do say "START v12ce_seed$SD"; ( /usr/bin/time -v $P $S2 --tag v12ce_seed$SD --seed $SD --threads ${S2TH:-40} > $L/stage2_v12ce_seed$SD.log 2>&1 ) & pids="$pids $!"; done
for p in $pids; do wait $p || { say "FAILED seed fit"; exit 1; }; done
$P cloud/s2avg.py --tag v12ce > $L/s2avg_v12ce.log 2>&1 || { say "FAILED s2avg"; exit 1; }
$P cloud/v12_eval.py --tag v12ce_s3avg > $L/gate_v12ce_s3avg.log 2>&1; cat $L/gate_v12ce_s3avg.log
md5sum models/stage2_v12ce.txt models/stage2_v12ce_seed1.txt models/stage2_v12ce_seed2.txt
date > $L/.done_clean_s3; say "clean 3-seed done"
