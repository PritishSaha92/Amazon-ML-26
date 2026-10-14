#!/bin/bash
# fr12d: resume scripts/build_france_entry.sh AFTER its step (4a) (used when cloud/s2avg.py OOMs and the average was produced by
# research/scripts_62/s2avg_lean.py). Executes the certified builder's own code from "(4b)" to the end VERBATIM (extracted at runtime).
#   bash research/scripts_62/builder_resume.sh <NAME> <FRANCE_CE_FILE>
set -euo pipefail; cd ${BER_ROOT:?}
B=scripts/build_france_entry.sh; [ "$(md5sum < $B | cut -d' ' -f1)" = 4269a5740577e6d98ecc29d15ffb3bda ] || { echo "builder changed"; exit 1; }  # packaged copy: md5 of the packaged scripts/build_france_entry.sh
NAME=$1; FCE=$2; P=${BER_PY:?}; TH=3; export TH POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 RAYON_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
TAG=${NAME}ce2_E2; FIN=submissions/v12ce2_E2_fr; ENTRY=submissions/v12ce2_E2_fr${NAME}; L=logs/france_entries; T0=$(date +%s)
say() { echo "[$(date '+%F %T')] $*"; }; die() { say "FAILED: $*"; exit 1; }
[ -f output/test_scores_${TAG}_s3avg_s2.parquet ] || die "no s3avg"
[ -e $ENTRY/matching_results.tsv ] && die "$ENTRY exists"
SHIFT=$(tail -1 $L/${NAME}_ce.log)
source <(sed -n '/^say "(4b) chain_rules_on \$TAG"/,$p' $B)
