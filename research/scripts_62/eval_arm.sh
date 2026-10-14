#!/bin/bash
# fr12d: full three-way evaluation of one France CE arm + entry via the CERTIFIED builder (scripts/build_france_entry.sh).
#   bash research/scripts_62/eval_arm.sh <NAME> <MODEL_A> <MODEL_B> [NODE_SCORE_DIR]
#   NAME alnum (e.g. FR12H, FR12, ST12). If NODE_SCORE_DIR has France/val score files they are used; otherwise the 5090 scores.
# Every heavy step goes through tools/heavy.sh (memory cap), 3 threads. Steps, all logged to logs/fr12d/eval_<NAME>.log:
#  1 scores (A,B -> mean)  2 val (i) raw swap  3 val (ii) native-val-fitted per-cell quantile map  4 OOF for (iii) (GPU; refit on node)
#  5 pick the best variant; build entry iff its val delta >= -0.0001  6 label-free diagnostics (diag / numcheck / agree)
set -uo pipefail; cd ${BER_ROOT:?}
P=${BER_PY:?}; W=${BER_SCRATCH:?}/fr12d; H=tools/heavy.sh
export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 RAYON_NUM_THREADS=3
N=$1; MA=$2; MB=$3; ND=${4:-}; TG=fr12d_$(echo $N | tr A-Z a-z)
say() { echo "[$(date '+%F %T')] $*"; }
say "eval $N: models $MA $MB node-dir '${ND}'"
md5sum $MA/model.safetensors $MB/model.safetensors 2>/dev/null
# 1 scores
if [ ! -f $W/${N}_fr_mean.parquet ]; then
  $H -m 16G bash research/scripts_62/score_arm.sh $N $MA $MB > logs/fr12d/score_$N.log 2>&1 || { say "scoring FAILED"; tail -5 logs/fr12d/score_$N.log; exit 1; }
fi
tail -3 logs/fr12d/score_$N.log
# 2 val raw
$H -m 16G $P research/scripts_62/val_apply.py --ce-val $W/${N}_val_mean.parquet --tag ${TG} --threads 3 > logs/fr12d/${TG}_val_apply.log 2>&1
$H -m 12G $P research/scripts_62/val_contrast_se.py --tag ${TG}_s3avg | tee logs/fr12d/${TG}_val_contrast.log
# 3 val qmap
$H -m 28G $P research/scripts_62/qmap_val.py --name $N --val $W/${N}_val_mean.parquet --fr $W/${N}_fr_mean.parquet | tee logs/fr12d/${N}_qmap.log
$H -m 16G $P research/scripts_62/val_apply.py --ce-val $W/${N}q_val.parquet --tag ${TG}q --threads 3 > logs/fr12d/${TG}q_val_apply.log 2>&1
$H -m 12G $P research/scripts_62/val_contrast_se.py --tag ${TG}q_s3avg | tee logs/fr12d/${TG}q_val_contrast.log
# 4 OOF (for the node refit, variant iii)
[ -f $W/${N}_oof.parquet ] || $H -m 16G bash research/scripts_62/oof_arm.sh $N $MA $MB > logs/fr12d/oof_$N.log 2>&1
tail -2 logs/fr12d/oof_$N.log
# 5 pick + build
read DR DQ < <($P - ${TG} <<'PY'
import re, sys
t = sys.argv[1]; g = lambda f: float(re.search(r"delta ([+-][0-9.]+)", open(f).readline()).group(1))
print(g(f"logs/fr12d/{t}_val_contrast.log"), g(f"logs/fr12d/{t}q_val_contrast.log"))
PY
)
say "val deltas: raw $DR  qmap $DQ"
BEST=$($P -c "r,q=$DR,$DQ; b=max((r,'raw'),(q,'q')); print(b[1] if b[0]>=-0.0001 else 'none')")
say "best supported variant: $BEST"
if [ "$BEST" = raw ]; then EN=$N; FCE=$W/${N}_fr_mean.parquet; elif [ "$BEST" = q ]; then EN=${N}Q; FCE=$W/${N}q_fr.parquet; else EN=""; fi
if [ -n "$EN" ]; then
  say "building entry v12ce2_E2_fr$EN via certified builder"
  bash research/scripts_62/build_entry.sh $EN $FCE > logs/fr12d/build_$EN.log 2>&1; tail -4 logs/fr12d/build_$EN.log
  $H -m 16G $P research/scripts_62/diag.py --name $N --fce $W/${N}_fr_mean.parquet --tag ${EN}ce2_E2 --entry submissions/v12ce2_E2_fr$EN > logs/fr12d/diag_$N.txt 2>&1
else
  say "no variant meets the -0.0001 retention bar: no entry; CE-level diagnostics only"
  $H -m 16G $P research/scripts_62/diag.py --name $N --fce $W/${N}_fr_mean.parquet > logs/fr12d/diag_$N.txt 2>&1
fi
grep -E "==|CE cross|stage-2 cross|FINAL" logs/fr12d/diag_$N.txt
[ -f $W/${N}_final_diff.parquet ] && $H -m 12G $P research/scripts_62/numcheck.py $N ST0 frce 2>/dev/null
$H -m 12G $P research/scripts_62/agree.py $N ST0 frce
# G guarded variant (pre-registered 11:20 for fr12h; same rule for every arm)
if [ "$MA" != none ]; then
  $H -m 12G $P research/scripts_62/guard.py $N
  $H -m 16G $P research/scripts_62/val_apply.py --ce-val $W/${N}G_val_mean.parquet --tag ${TG}g --threads 3 > logs/fr12d/${TG}g_val_apply.log 2>&1
  $H -m 12G $P research/scripts_62/val_contrast_se.py --tag ${TG}g_s3avg | tee logs/fr12d/${TG}g_val_contrast.log
  DG=$($P -c "import re;print(re.search(r'delta ([+-][0-9.]+)',open('logs/fr12d/${TG}g_val_contrast.log').readline()).group(1))")
  say "G val delta $DG"
  $P -c "import sys; sys.exit(0 if $DG >= -0.0001 else 1)" && { bash research/scripts_62/build_entry.sh ${N}G $W/${N}G_fr_mean.parquet > logs/fr12d/build_${N}G.log 2>&1; tail -2 logs/fr12d/build_${N}G.log; }
fi
# (iii) refit variant: wait up to 90 min for the node's models/stage2_fr12d_s2${N}{,_seed1,_seed2}
for k in $(seq 1 18); do [ -f models/stage2_fr12d_s2${N}_seed2.txt ] && [ -f models/stage2_fr12d_s2${N}.txt ] && break; sleep 300; done
if [ -f models/stage2_fr12d_s2${N}_seed2.txt ]; then
  $H -m 16G $P research/scripts_62/val_apply.py --ce-val $W/${N}_val_mean.parquet --tag ${TG}r --model models/stage2_fr12d_s2${N} --threads 3 > logs/fr12d/${TG}r_val_apply.log 2>&1
  $H -m 12G $P research/scripts_62/val_contrast_se.py --tag ${TG}r_s3avg | tee logs/fr12d/${TG}r_val_contrast.log
  DRF=$($P -c "import re;print(re.search(r'delta ([+-][0-9.]+)',open('logs/fr12d/${TG}r_val_contrast.log').readline()).group(1))")
  say "refit val delta $DRF"
  if $P -c "import sys; sys.exit(0 if $DRF >= -0.0001 else 1)"; then
    cp research/scripts_62/run_arm.sh $W/run_arm_frozen_${N}R.sh
    S2MODEL=models/stage2_fr12d_s2${N} $H -m 40G bash $W/run_arm_frozen_${N}R.sh ${TG}r - $W/${N}_fr_mean.parquet ${N}R > logs/fr12d/${TG}r.out 2>&1; tail -4 logs/fr12d/${TG}r.out
  fi
else say "no refit models for $N after 90 min"; fi
say "EVAL_DONE $N (entry: ${EN:-none})"
