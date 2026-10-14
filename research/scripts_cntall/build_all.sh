#!/bin/bash
# cntall: ALL-COUNTRY CE swap test chain (adapted from research/scripts_62/run_arm.sh, which fr12d certified byte-identical).
#   bash research/scripts_cntall/build_all.sh <TAG> <TEST_CE|prod> [S2MODEL]
#   TAG: cntall_* (controls) or v12cnt*  -> submissions/<TAG>_fr
#   TEST_CE: parquet s1_id,t_id,p_ce covering EXACTLY the 1,060,103 keys of output/ce2_test_v12avg.parquet (all countries);
#            'prod' = feed the production CE file unchanged (parity control).
#   S2MODEL: default models/stage2_v12ce2_E2 (raw swap); models/stage2_fr12d_s2FRNLP for the refit.
# Every heavy step goes through tools/heavy.sh (one at a time, <=3 threads). Never touches production files.
set -euo pipefail; cd ${BER_ROOT:?}
P=${BER_PY:?}; export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 RAYON_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
TAG=${1:?tag}; TCE=${2:?test ce or prod}; S2M=${3:-models/stage2_v12ce2_E2}; XFT=${4:-data/e2_feats_test_v12.parquet}   # 4th arg: test extra-feats (stacked arms: ${BER_SCRATCH:?}/cntall/stk_<ARM>_test.parquet)
[[ $TAG =~ ^(cntall_|v12cnt|v12stk|v12fr)[A-Za-z0-9_]+$ ]] || { echo "bad tag"; exit 1; }
H=tools/heavy.sh; W=${BER_SCRATCH:?}/cntall; mkdir -p $W logs/cntall; FIN=submissions/v12ce2_E2_fr
say() { echo "[$(date '+%F %T')] $*"; }
[ "$(md5sum < $FIN/matching_results.tsv | cut -d' ' -f1)" = fb0f90f3d79f625c2e6adb3bd1d39b06 ] || { [ "${BER_STRICT_MD5:-0}" = 1 ] && { say "finalist md5 changed"; exit 1; }; say "WARNING: $FIN differs from the reference md5 fb0f90f3 (expected after retraining)"; }  # packaged copy: strict only with BER_STRICT_MD5=1
[ -e submissions/${TAG}_fr ] && { say "submissions/${TAG}_fr exists; refusing"; exit 1; }
for X in "" _seed1 _seed2; do [ -f $S2M$X.txt ] && [ -f $S2M$X.json ] || { say "missing $S2M$X"; exit 1; }; md5sum $S2M$X.txt; done
if [ "$TCE" = "prod" ]; then NEWCE=output/ce2_test_v12avg.parquet
else
  NEWCE=$W/${TAG}_ce2_test.parquet
  say "(T) all-country test CE from $TCE"
  $H -m 12G $P - "$TCE" output/ce2_test_v12avg.parquet $NEWCE <<'PY'
import sys, numpy as np, polars as pl
fce, basef, outf = sys.argv[1:4]; K = ["s1_id", "t_id"]
f = pl.read_parquet(fce); pc = "p_ce" if "p_ce" in f.columns else "p"
f = f.select(K + [pl.col(pc).cast(pl.Float64).alias("p_new")]); assert f.select(K).is_duplicated().sum() == 0, "dup keys"
v = f["p_new"].to_numpy(); assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
base = pl.read_parquet(basef); assert base.select(K).is_duplicated().sum() == 0
print(f"base rows {base.height:,}  new rows {f.height:,}  base-not-in-new {base.join(f, on=K, how='anti').height}  new-not-in-base {f.join(base, on=K, how='anti').height}")
assert (base.height == 1060103 or __import__("os").environ.get("BER_STRICT_MD5", "0") != "1") and f.height == base.height and base.join(f, on=K, how="anti").height == 0, "coverage must be exact"  # packaged copy: reference-run count strict only with BER_STRICT_MD5=1
j = base.with_row_index("_r").join(f, on=K, how="left").sort("_r")
out = j.with_columns(p_ce=pl.col("p_new").cast(base["p_ce"].dtype)).select(base.columns)
assert out.select(K).equals(base.select(K))
out.write_parquet(outf)
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
d = base.rename({"p_ce": "o"}).with_columns(n=out["p_ce"]).join(cty, on="s1_id", how="left")
print(d.group_by("country").agg(pl.len(), pl.col("o").mean().alias("prod_mean"), pl.col("n").mean().alias("new_mean"),
      pl.corr("o", "n", method="spearman").alias("spearman"), ((pl.col("o") < .5) & (pl.col("n") >= .5)).sum().alias("up05"),
      ((pl.col("o") >= .5) & (pl.col("n") < .5)).sum().alias("down05")).sort("country"))
PY
fi
md5sum $NEWCE $XFT
for X in "" _seed1 _seed2; do
  say "stage2_apply seed '$X' model $S2M$X"
  $H -m 28G $P src/stage2_apply.py --scores output/test_scores_v12avg.parquet --pairs data/pairs_test_v10w_feat.parquet --split test \
     --model $S2M$X --ce $NEWCE --extra-feats $XFT \
     --out output/test_scores_${TAG}${X}_s2.parquet --threads 3 > logs/cntall/${TAG}${X}_stage2_apply.log 2>&1
done
say "s2avg_lean"
$H -m 28G $P research/scripts_62/s2avg_lean.py --tag $TAG > logs/cntall/${TAG}_s2avg.log 2>&1; tail -2 logs/cntall/${TAG}_s2avg.log
rm -f output/test_scores_${TAG}_s2.parquet output/test_scores_${TAG}_seed1_s2.parquet output/test_scores_${TAG}_seed2_s2.parquet
md5sum output/test_scores_${TAG}_s3avg_s2.parquet
say "rules"
$H -m 36G bash research/scripts_62/chain_rules_on_t4.sh $TAG output/test_scores_${TAG}_s3avg_s2.parquet > logs/cntall/${TAG}_rules.log 2>&1 || { tail -5 logs/cntall/${TAG}_rules.log; exit 1; }
rm -f output/test_scores_${TAG}_acr.parquet
tail -4 logs/cntall/${TAG}_rules.log
E=submissions/${TAG}_fr
ln -f submissions/v10w_fr/candidate_pairs.tsv $E/candidate_pairs.tsv 2>/dev/null || true
say "official validator"
$H -m 28G $P student_resource/utils/validate_submission.py --matching $E/matching_results.tsv --candidate $E/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test --check-ids > logs/cntall/${TAG}_validate.log 2>&1 || true
tail -3 logs/cntall/${TAG}_validate.log
(cd $E && md5sum matching_results.tsv candidate_pairs.tsv > md5.txt && cat md5.txt)
say "DONE $TAG"
