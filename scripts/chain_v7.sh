#!/bin/bash
# v7 = the full v6 pipeline rebuilt on a REDUCED candidate set: the candidate rule below is applied to the union of the
# 6 retrieval routes inside build_pairs.py (before any context/competition feature, labels, sampling), in train AND test,
# so candidate_pairs.tsv (~55-60/S1 on test) is exactly the set every model runs over.
#   stage 1: lgbm_v7 (defaults, 1500 rounds) + lgbm_v7b (255 leaves, lr 0.03, ff 0.7, min-leaf 50, seed 7, 3000 rounds)
#   → OOF of both (sequential; ~81 GB RSS each at full size) → averaged → stage 2 (v7avg)
#   → val/audit blend check with the averaged CE seeds (ce_v4_hard + ce_v4_hard_s2) at w0.5 t0.58 alpha 0.7
#   → test: build_pairs + features → predict v7/v7b → average → stage2_apply → CE bands (2 seeds) → average → blend
#   → submissions/v7_s2_ce4avg
cd ${BER_ROOT:?} || exit 1
export HF_HOME=${BER_SCRATCH:?}/hf POLARS_MAX_THREADS=10
P=${BER_PY:?}
RULE="(pl.col('n_views') >= 2) | (pl.col('fwd_rank') <= 8) | (pl.col('rdense_rank') <= 4) | (pl.col('rfull_rank') <= 2)"
HP="--leaves 255 --lr 0.03 --ff 0.7 --bf 0.8 --min-leaf 50 --l2 1.0 --seed 7"
CEA=models/ce_v4_hard; CEB=models/ce_v4_hard_s2
say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
run() { local log=$1; shift; say "START $* > $log"; nice -n 10 "$@" > "$log" 2>&1 || { say "FAILED ($?): $* (see $log) — aborting v7"; exit 1; }; say "ok"; }
say "v7 chain start; rule: $RULE"

# ---- train side -------------------------------------------------------------------------------------------------
run logs/build_pairs_train_v7.log $P src/build_pairs.py --cands data/cands_sparsev_train.parquet,data/cands_dense_train.parquet,data/cands_rdense_train.parquet,data/cands_rsparse_train.parquet \
    --split train --sample-s1 400000 --keep-rule "$RULE" --out data/pairs_train_v7.parquet
grep -E "keep-rule|union before|unique pairs|recall|rows written|label balance" logs/build_pairs_train_v7.log
run logs/pair_features_train_v7.log $P src/pair_features.py --pairs data/pairs_train_v7.parquet --split train --out data/pairs_train_v7_feat.parquet --workers 10
run logs/train_matcher_v7.log $P src/train_matcher.py --pairs data/pairs_train_v7_feat.parquet --tag v7 --rounds 1500 --threads 8
grep -E "^trained|best val macro|ALL" logs/train_matcher_v7.log
run logs/train_matcher_v7b.log $P src/train_matcher.py --pairs data/pairs_train_v7_feat.parquet --tag v7b $HP --rounds 3000 --threads 8
grep -E "^trained|best val macro|ALL" logs/train_matcher_v7b.log
say "(v5 val 0.9873 / audit 0.9877; v5b 0.9875 / 0.9876)"
run logs/oof_v7.log  $P src/oof_scores.py --pairs data/pairs_train_v7_feat.parquet --tag v7  --folds 5 --threads 8
run logs/oof_v7b.log $P src/oof_scores.py --pairs data/pairs_train_v7_feat.parquet --tag v7b --folds 5 --threads 8 $HP
run logs/average_oof_v7avg.log $P src/average_scores.py --a data/oof_scores_v7.parquet --b data/oof_scores_v7b.parquet --out data/oof_scores_v7avg.parquet
# stage2.py reads only "countries" from models/lgbm_<tag>.json; predict.py --scores-from loads models/lgbm_<tag>.{json,txt} (ownership policy)
cp models/lgbm_v7.json models/lgbm_v7avg.json && ln -sf lgbm_v7.txt models/lgbm_v7avg.txt || { say "model json/symlink failed"; exit 1; }
run logs/stage2_v7avg.log $P src/stage2.py --tag v7avg --pairs data/pairs_train_v7_feat.parquet
grep -E "STAGE-1 baseline|STAGE-2|SUMMARY" logs/stage2_v7avg.log; say "(v6 stage 2: val 0.9886 / audit 0.9890)"

# ---- val/audit blend check (s2adj; labels from the v7 stage-1 score files: same pairs) -----------------------------
run logs/build_audit_scores_v7.log $P src/eval_ce_blend.py --split audit --scores output/audit_scores_v7.parquet --build --lgbm models/lgbm_v7 \
    --feat data/pairs_train_v7_feat.parquet --alpha 0.0 --t 0.75
run logs/s2adj_v7avg.log $P - <<'PY'
import polars as pl
for s in ("val", "audit"):
    s2 = pl.read_parquet(f"output/{s}_scores_s2_v7avg.parquet").select("s1_id", "t_id", pl.col("p_s2").cast(pl.Float64).alias("p"))
    lab = pl.read_parquet(f"output/{s}_scores_v7.parquet").select("s1_id", "t_id", "label")
    out = s2.join(lab, on=["s1_id", "t_id"], how="left"); assert out["label"].null_count() == 0
    out.select("s1_id", "t_id", "label", "p").write_parquet(f"output/{s}_scores_s2adj_v7avg.parquet"); print(s, len(out))
PY
for S in val audit; do
  run logs/ce4_score_${S}_s2adj_v7avg.log $P src/score_cross_encoder.py --model $CEA --pairs output/${S}_scores_s2adj_v7avg.parquet --records data/train_records.parquet --band 0.02,0.98 --out output/ce4_${S}_s2adj_v7avg.parquet --threads 8
  run logs/ce4s2_score_${S}_s2adj_v7avg.log $P src/score_cross_encoder.py --model $CEB --pairs output/${S}_scores_s2adj_v7avg.parquet --records data/train_records.parquet --band 0.02,0.98 --out output/ce4s2_${S}_s2adj_v7avg.parquet --threads 8
  run logs/ce4avg_${S}_s2adj_v7avg.log $P src/average_ce.py --a output/ce4_${S}_s2adj_v7avg.parquet --b output/ce4s2_${S}_s2adj_v7avg.parquet --out output/ce4avg_${S}_s2adj_v7avg.parquet
done
run logs/ce4avg_blend_val_s2adj_v7avg.log $P src/eval_ce_blend.py --split val --scores output/val_scores_s2adj_v7avg.parquet --ce output/ce4avg_val_s2adj_v7avg.parquet --alpha 0.7
run logs/ce4avg_blend_audit_s2adj_v7avg.log $P src/eval_ce_blend.py --split audit --scores output/audit_scores_s2adj_v7avg.parquet --ce output/ce4avg_audit_s2adj_v7avg.parquet --alpha 0.7 --w 0.5 --t 0.58
say "blend check done (logs/ce4avg_blend_{val,audit}_s2adj_v7avg.log; v6_s2_ce4avg: val 0.9899 / audit 0.9900)"

# ---- test side ----------------------------------------------------------------------------------------------------
run logs/build_pairs_test_v7.log $P src/build_pairs.py --cands data/cands_sparsev_test.parquet,data/cands_dense_test.parquet,data/cands_rdense_test.parquet,data/cands_rsparse_test.parquet \
    --split test --keep-rule "$RULE" --out data/pairs_test_v7.parquet
grep -E "keep-rule|union before|unique pairs|rows written" logs/build_pairs_test_v7.log
run logs/pair_features_test_v7.log $P src/pair_features.py --pairs data/pairs_test_v7.parquet --split test --out data/pairs_test_v7_feat.parquet --workers 10
run logs/predict_v7.log  $P src/predict.py --pairs data/pairs_test_v7_feat.parquet --model models/lgbm_v7  --tag v7
run logs/predict_v7b.log $P src/predict.py --pairs data/pairs_test_v7_feat.parquet --model models/lgbm_v7b --tag v7b
run logs/average_test_v7avg.log $P src/average_scores.py --a output/test_scores_v7.parquet --b output/test_scores_v7b.parquet --out output/test_scores_v7avg.parquet
run logs/stage2_apply_v7avg.log $P src/stage2_apply.py --scores output/test_scores_v7avg.parquet --pairs data/pairs_test_v7_feat.parquet --split test --model models/stage2_v7avg --out output/test_scores_v7avg_s2.parquet
run logs/ce4_score_test_s2_v7avg.log   $P src/score_cross_encoder.py --model $CEA --pairs output/test_scores_v7avg_s2.parquet --records data/test_records.parquet --band 0.02,0.98 --out output/ce4_test_s2_v7avg.parquet --threads 8
run logs/ce4s2_score_test_s2_v7avg.log $P src/score_cross_encoder.py --model $CEB --pairs output/test_scores_v7avg_s2.parquet --records data/test_records.parquet --band 0.02,0.98 --out output/ce4s2_test_s2_v7avg.parquet --threads 8
run logs/ce4avg_test_s2_v7avg.log $P src/average_ce.py --a output/ce4_test_s2_v7avg.parquet --b output/ce4s2_test_s2_v7avg.parquet --out output/ce4avg_test_s2_v7avg.parquet
run logs/predict_v7_s2_ce4avg.log $P src/predict.py --pairs data/pairs_test_v7_feat.parquet --model models/lgbm_v7avg --scores-from output/test_scores_v7avg_s2.parquet \
    --blend-ce output/ce4avg_test_s2_v7avg.parquet --w 0.5 --t 0.58 --alpha 0.7 --tag v7_s2_ce4avg
tail -15 logs/predict_v7_s2_ce4avg.log
mkdir -p submissions/v7_s2_ce4avg && cp output/matching_results_v7_s2_ce4avg.tsv submissions/v7_s2_ce4avg/matching_results.tsv \
  && cp output/candidate_pairs_v7_s2_ce4avg.tsv submissions/v7_s2_ce4avg/candidate_pairs.tsv \
  && md5sum submissions/v7_s2_ce4avg/matching_results.tsv submissions/v7_s2_ce4avg/candidate_pairs.tsv > submissions/v7_s2_ce4avg/md5.txt || { say "freeze failed"; exit 1; }
cat submissions/v7_s2_ce4avg/md5.txt
echo "DONE v7_s2_ce4avg" >> logs/predict_v7_s2_ce4avg.log; say "DONE v7_s2_ce4avg"
