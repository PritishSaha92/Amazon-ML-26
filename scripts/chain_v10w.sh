#!/bin/bash
# v10w = champion pipeline (v7_s2ce_acr_nocat) with the India/US TEST word-LR columns computed from the TRAIN table
# (research/18_audit_features.md finding 1). France rows keep wordlr_test.json values. Frozen models; no retrain.
# Usage: scripts/chain_v10w.sh <stage>   stages: feat | s1 | ce | s2 | rules | report | controls   (run in order)
# As run 2026-09-26: feat 17:25-17:37 (setsid nohup, logs/chain_v10w_feat.out); s1 17:37-18:10 (copy ${BER_SCRATCH:?}/v10w/run_s1.sh, logs/chain_v10w_s1.out);
# ce+s2+rules+report 18:10-18:17 via ${BER_SCRATCH:?}/v10w/run_rest_all.sh (loops these 4 stages; logs/chain_v10w_rest.out); controls separately.
# Then: python logs/probe_diff.py v10w_fr v7_s2ce_acr_nocat
cd ${BER_ROOT:?} || exit 1
export HF_HOME=${BER_SCRATCH:?}/hf POLARS_MAX_THREADS=10 OMP_NUM_THREADS=10
P=${BER_PY:?}; W=${BER_SCRATCH:?}/v10w; mkdir -p $W
say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
run() { local log=$1; shift; say "START $* > $log"; local t0=$(date +%s); nice -n 10 "$@" > "$log" 2>&1 || { say "FAILED ($?): $* (see $log)"; exit 1; }; say "ok ($(( $(date +%s) - t0 ))s)"; }
case "$1" in
feat)
  # (a) reproduction check: recompute India/US word-LR columns with the TEST table loaded from json -> must equal stored columns
  # (b) the fix: same with the TRAIN table. Run concurrently (5 polars threads each).
  POLARS_MAX_THREADS=5 run logs/pf_v10w_check.log /usr/bin/time -v $P src/pair_features.py --split test --wordlr-only --wordlr-countries India,US \
      --wordlr-table-in data/tables/wordlr_test.json --pairs data/pairs_test_v7_feat.parquet --out $W/pairs_test_v7_testtab_check.parquet &
  POLARS_MAX_THREADS=5 run logs/pf_v10w.log /usr/bin/time -v $P src/pair_features.py --split test --wordlr-only --wordlr-countries India,US \
      --wordlr-table-in data/tables/wordlr_train.json --pairs data/pairs_test_v7_feat.parquet --out data/pairs_test_v10w_feat.parquet &
  wait
  [ -f $W/pairs_test_v7_testtab_check.parquet ] && [ -f data/pairs_test_v10w_feat.parquet ] || { say "feature step failed"; exit 1; }
  run logs/v10w_feat_compare.log $P logs/v10w_feat_compare.py
  ;;
s1)
  # stage 1 exactly as chain_v7.sh (predict.py streams the features file with the frozen models; only the scores file is used)
  run logs/predict_v10wA.log /usr/bin/time -v $P src/predict.py --pairs data/pairs_test_v10w_feat.parquet --model models/lgbm_v7  --tag v10wA
  run logs/predict_v10wB.log /usr/bin/time -v $P src/predict.py --pairs data/pairs_test_v10w_feat.parquet --model models/lgbm_v7b --tag v10wB
  rm -f output/matching_results_v10wA.tsv output/candidate_pairs_v10wA.tsv output/matching_results_v10wB.tsv output/candidate_pairs_v10wB.tsv
  run logs/average_test_v10w.log $P src/average_scores.py --a output/test_scores_v10wA.parquet --b output/test_scores_v10wB.parquet --out output/test_scores_v10w_avg.parquet
  run logs/v10w_s1_compare.log $P - <<'PY'
import polars as pl, numpy as np
for a, b in (("output/test_scores_v10wA.parquet", "output/test_scores_v7.parquet"), ("output/test_scores_v10wB.parquet", "output/test_scores_v7b.parquet"),
             ("output/test_scores_v10w_avg.parquet", "output/test_scores_v7avg.parquet")):
    x, y = pl.read_parquet(a), pl.read_parquet(b)
    assert x.select("s1_id", "t_id", "country").equals(y.select("s1_id", "t_id", "country")), "row order/ids differ"
    fr = x["country"] == "France"
    print(f"{a} vs {b}: France p identical {x.filter(fr)['p'].equals(y.filter(fr)['p'])}")
    for c in ("India", "US"):
        m = (x["country"] == c).to_numpy(); d = x["p"].to_numpy()[m].astype(np.float64) - y["p"].to_numpy()[m]
        po, pn = y["p"].to_numpy()[m], x["p"].to_numpy()[m]
        print(f"  {c}: {m.sum():,} pairs, mean p {po.mean():.5f} -> {pn.mean():.5f}, mean|dp| {np.abs(d).mean():.2e}, changed {(d != 0).mean():.2%}, "
              f"cross 0.5 up {((po < .5) & (pn >= .5)).sum():,} down {((po >= .5) & (pn < .5)).sum():,}; CE band [.02,.98] {((po >= .02) & (po <= .98)).sum():,} -> {((pn >= .02) & (pn <= .98)).sum():,}")
PY
  ;;
ce)
  # CE band (0.02-0.98 of the averaged stage-1 p) shifts: reuse cef_test_v7avg scores for band rows that have one; score the rest with
  # both fold CEs (ce_v7fA / ce_v7fB, as chain_v7ce_folds.sh) and average; cef_test_v10w = exactly the new band.
  run logs/v10w_ce_missing.log $P - <<'PY'
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import polars as pl
s = pl.read_parquet("output/test_scores_v10w_avg.parquet").filter(pl.col("p").is_between(0.02, 0.98)).select("s1_id", "t_id", "country")
old = pl.read_parquet("output/cef_test_v7avg.parquet"); ob = pl.read_parquet("output/test_scores_v7avg.parquet").filter(pl.col("p").is_between(0.02, 0.98))
assert len(old) == len(ob)
miss = s.join(old, on=["s1_id", "t_id"], how="anti"); left = old.join(s, on=["s1_id", "t_id"], how="anti")
print(f"new CE band {len(s):,} (old {len(old):,}); reused {len(s)-len(miss):,}; lacking CE {len(miss):,} {miss.group_by('country').len().sort('country').rows()}; left band {len(left):,}")
miss.select("s1_id", "t_id").write_parquet((_BER_S + "/v10w/ce_missing_test.parquet"))
PY
  cat logs/v10w_ce_missing.log
  for F in A B; do
    run logs/cef_score_test_v10w_$F.log $P src/score_cross_encoder.py --model models/ce_v7f$F --pairs $W/ce_missing_test.parquet --records data/test_records.parquet --out $W/pce_test_missing_$F.parquet --threads 8
  done
  run logs/cef_avg_test_v10w_missing.log $P src/average_ce.py --a $W/pce_test_missing_A.parquet --b $W/pce_test_missing_B.parquet --out $W/pce_test_missing_avg.parquet
  run logs/v10w_ce_merge.log $P - <<'PY'
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import polars as pl
s = pl.read_parquet("output/test_scores_v10w_avg.parquet").filter(pl.col("p").is_between(0.02, 0.98)).select("s1_id", "t_id")
old = pl.read_parquet("output/cef_test_v7avg.parquet"); new = pl.read_parquet((_BER_S + "/v10w/pce_test_missing_avg.parquet"))
out = pl.concat([old.join(s, on=["s1_id", "t_id"], how="semi"), new.select(old.columns).cast(old.schema)])
assert out.select("s1_id", "t_id").is_duplicated().sum() == 0 and len(out) == len(s) and len(out.join(s, on=["s1_id", "t_id"], how="semi")) == len(s)
out.write_parquet("output/cef_test_v10w.parquet"); print(f"wrote output/cef_test_v10w.parquet {len(out):,} rows (reused {len(out)-len(new):,}, new {len(new):,})")
PY
  cat logs/v10w_ce_merge.log
  ;;
s2)
  run logs/stage2_apply_v10w.log /usr/bin/time -v $P src/stage2_apply.py --scores output/test_scores_v10w_avg.parquet --pairs data/pairs_test_v10w_feat.parquet \
      --split test --model models/stage2_v7ce --ce output/cef_test_v10w.parquet --out output/test_scores_v10w_s2.parquet
  run logs/v10w_s2_compare.log $P - <<'PY'
import polars as pl, numpy as np
x, y = pl.read_parquet("output/test_scores_v10w_s2.parquet"), pl.read_parquet("output/test_scores_v7ce_s2.parquet")
assert x.select("s1_id", "t_id", "country").equals(y.select("s1_id", "t_id", "country"))
fr = x["country"] == "France"; print("France stage-2 p identical to test_scores_v7ce_s2:", x.filter(fr).equals(y.filter(fr)))
for c in ("India", "US"):
    m = (x["country"] == c).to_numpy(); po, pn = y["p"].to_numpy()[m], x["p"].to_numpy()[m]
    print(f"{c}: mean p {po.mean():.5f} -> {pn.mean():.5f}; p>=0.5 {(po >= .5).sum():,} -> {(pn >= .5).sum():,}; crossing up {((po < .5) & (pn >= .5)).sum():,} down {((po >= .5) & (pn < .5)).sum():,}")
PY
  ;;
rules)
  # France rules read only country/shift_sig/hn_exact/t_addr_empty (acronym) and country/hn_exact/shift_sig + views + France pool of
  # wordlr_test.json (catswap); predict --scores-from reads only s1_id/t_id from --pairs -> the v7 features file is correct here.
  scripts/chain_rules_on.sh v10w output/test_scores_v10w_s2.parquet || exit 1
  ;;
report)
  run logs/v10w_report.log $P logs/v10w_report.py
  cat logs/v10w_report.log
  ;;
controls)
  # reviewer controls for research/18 (val/audit matched-CE reverse experiment; table vocabulary / support-cutoff split)
  W2=$W/ctl; mkdir -p $W2
  run logs/v10w_ctl_prep.log $P research/scripts_18/c1_matched_ce_control.py prep
  for F in A B; do run logs/v10w_ctl_ce_$F.log $P src/score_cross_encoder.py --model models/ce_v7f$F --pairs $W2/ce_missing.parquet --records data/train_records.parquet --out $W2/ce_missing_$F.parquet --threads 8; done
  run logs/v10w_ctl_ce_avg.log $P src/average_ce.py --a $W2/ce_missing_A.parquet --b $W2/ce_missing_B.parquet --out $W2/ce_missing_avg.parquet
  run logs/v10w_ctl_eval.log $P research/scripts_18/c1_matched_ce_control.py eval
  POLARS_MAX_THREADS=4 run logs/v10w_ctl_vocab.log $P research/scripts_18/c2_table_vocab.py   # needs output/test_scores_v10w_avg.parquet
  ;;
esac
