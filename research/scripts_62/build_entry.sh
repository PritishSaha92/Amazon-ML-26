#!/bin/bash
# fr12d: certified builder with OOM fallback. Runs scripts/build_france_entry.sh under heavy.sh; if it dies after writing the three seed
# stage-2 files (cloud/s2avg.py OOM at the cap), averages them with s2avg_lean.py and resumes the builder's own tail (builder_resume.sh).
#   bash research/scripts_62/build_entry.sh <NAME> <FRANCE_CE_FILE>     (log: logs/fr12d/build_<NAME>.log)
cd ${BER_ROOT:?}; N=$1; F=$2; T=${N}ce2_E2; P=${BER_PY:?}; export POLARS_MAX_THREADS=3
tools/heavy.sh -m 40G bash scripts/build_france_entry.sh $N $F; rc=$?
[ $rc -eq 0 ] && { echo "BUILD_OK $N"; exit 0; }
if [ ! -f output/test_scores_${T}_s3avg_s2.parquet ] && ls output/test_scores_${T}_s2.parquet output/test_scores_${T}_seed1_s2.parquet output/test_scores_${T}_seed2_s2.parquet >/dev/null 2>&1 \
   && grep -q "slot .* done rc=0: .*stage2_apply\|wrote output/test_scores_${T}_seed2_s2" logs/stage2_apply_${T}_seed2.log logs/heavy_slots/heavy.log 2>/dev/null; then
  echo "builder rc=$rc after the seed files -> lean average + resume"
  tools/heavy.sh -m 24G $P research/scripts_62/s2avg_lean.py --tag $T || { echo BUILD_FAILED; exit 1; }
  tools/heavy.sh -m 40G bash research/scripts_62/builder_resume.sh $N $F && { echo "BUILD_OK $N (resumed)"; exit 0; }
fi
echo "BUILD_FAILED $N rc=$rc"; exit 1
