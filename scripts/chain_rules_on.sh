#!/bin/bash
# Apply the two confirmed France rules (acronym raise, category-swap drop) on top of ANY stage-2 test scores parquet,
# then predict + freeze a submission.  Usage: scripts/chain_rules_on.sh <tag> <stage2_test_scores.parquet>
# Result: submissions/<tag>_fr/ (matching_results.tsv, candidate_pairs.tsv hard-linked to v7_s2ce's, md5.txt)
cd ${BER_ROOT:?} || exit 1
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8
P=${BER_PY:?}; TAG=$1; SRC=$2; [ -n "$TAG" ] && [ -f "$SRC" ] || { echo "usage: $0 <tag> <scores.parquet>"; exit 1; }
say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
run() { local log=$1; shift; say "START $* > $log"; nice -n 10 "$@" > "$log" 2>&1 || { say "FAILED ($?): $* (see $log)"; exit 1; }; say "ok"; }
grep -q '"ownership": true' models/lgbm_v7avg.json || { say "lgbm_v7avg.json ownership is not true"; exit 1; }
run logs/acr_${TAG}.log $P src/apply_acronym_rule.py --scores "$SRC" --pairs data/pairs_test_v7_feat.parquet --views data/views_test.parquet --out output/test_scores_${TAG}_acr.parquet
run logs/catswap_${TAG}.log $P src/apply_catswap_drop.py --scores output/test_scores_${TAG}_acr.parquet --pairs data/pairs_test_v7_feat.parquet --views data/views_test.parquet --table data/tables/wordlr_test.json --out output/test_scores_${TAG}_fr.parquet
run logs/predict_${TAG}_fr.log $P src/predict.py --pairs data/pairs_test_v7_feat.parquet --model models/lgbm_v7avg --scores-from output/test_scores_${TAG}_fr.parquet --t 0.50 --alpha 0.7 --tag ${TAG}_fr
grep -q "PASS" logs/predict_${TAG}_fr.log || { say "validator did not PASS"; exit 1; }
# packaged copy: in a fresh tree the first run creates the base candidate export (v7_s2ce; v10w_fr is the same file)
[ -f submissions/v7_s2ce/candidate_pairs.tsv ] || { mkdir -p submissions/v7_s2ce submissions/v10w_fr; cp output/candidate_pairs_${TAG}_fr.tsv submissions/v7_s2ce/candidate_pairs.tsv; ln -f submissions/v7_s2ce/candidate_pairs.tsv submissions/v10w_fr/candidate_pairs.tsv; say "created base candidate file submissions/v7_s2ce"; }
cmp -s output/candidate_pairs_${TAG}_fr.tsv submissions/v7_s2ce/candidate_pairs.tsv || { say "candidate file differs from v7_s2ce"; exit 1; }
ln -f submissions/v7_s2ce/candidate_pairs.tsv output/candidate_pairs_${TAG}_fr.tsv
mkdir -p submissions/${TAG}_fr; cp output/matching_results_${TAG}_fr.tsv submissions/${TAG}_fr/matching_results.tsv
ln -f submissions/v7_s2ce/candidate_pairs.tsv submissions/${TAG}_fr/candidate_pairs.tsv
(cd submissions/${TAG}_fr && md5sum matching_results.tsv candidate_pairs.tsv > md5.txt && cat md5.txt)
grep -E "France|India|US " logs/predict_${TAG}_fr.log | head -3
say "DONE rules on $TAG -> submissions/${TAG}_fr"
