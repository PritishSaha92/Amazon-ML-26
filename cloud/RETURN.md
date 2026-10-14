# v12: local steps once the cloud bundle is back

Run everything from `$BER_ROOT`, with:
```
P=$BER_PY
export HF_HOME=$BER_SCRATCH/hf POLARS_MAX_THREADS=10 OMP_NUM_THREADS=10
```
The times below assume this machine (32 threads, 123 GB RAM, RTX 5090) with nothing else heavy running. If other agents' builds are running, expect roughly 1.5× longer.

## 0. Pull, verify, unpack (~5 min)
```
rsync -avP ubuntu@$IP:amazon/cloud_v12_return.tar ubuntu@$IP:amazon/cloud_v12_return.md5 $BER_SCRATCH/
(cd $BER_SCRATCH && md5sum -c cloud_v12_return.md5)              # must print OK before the node is terminated
ls data models output logs | grep -i v12                              # expect only data/v12 (s1_v7train) before unpacking
tar --keep-directory-symlink -xf $BER_SCRATCH/cloud_v12_return.tar                           # data/, models/, output/, logs/v12/ (data -> $BER_SCRATCH symlink)
```
Only then terminate the instance (lambda_launch.md step 6).

## 1. Read the cloud verdict (instant)
```
cat logs/v12/gate_v12ce.log                                           # fixed decoder (ownership, alpha 0.7, t 0.50) + GATE line
grep -E "CE features|with_p_ce|STAGE-1 baseline|STAGE-2|SUMMARY|t=0.50" logs/v12/stage2_v12ce.log
$P -c "import json;d=json.load(open('data/v12/ce_coverage_v12.json'));print({k:v for k,v in d.items() if k!='groups'});[print(g) for g in d['groups']]"
grep -hE "trained [0-9]+ rounds|best val macro|fixed own" logs/v12/full_v12.log logs/v12/full_v12b.log
grep -h AUC logs/v12/oof_v12*_f*.log
grep -h "Maximum resident\|Elapsed (wall" logs/v12/*.log | sort | uniq -c | sort -rn | head    # RAM/time record for LOG.md
```
Checks before going further:
* coverage: `violations 0`, `assembled_missing 0`. The `train_old_A` rows route to "B only" and `train_old_B` rows to "A only". The `train_new`, val and audit rows route to mean(A,B).
* stage-1 A/B val F0.5 (fixed own=True alpha=0.7 t=0.65 line): v7 got .98751. The learning curve predicts roughly +0.001 at all refs.

## 2. Stage 2 locally from the returned files (~20-30 min, ~30 GB RAM; optional confirmation)
The band cache came back, so stage2.py loads it and never reads the features file. `--pairs` points at the slim file, which is enough even if the cache had to be rebuilt.
```
test -f data/s2_band_v12avg.parquet || { echo "band cache missing"; exit 1; }
nice -n 10 $P src/stage2.py --tag v12ce_loc --pairs data/v12/pairs_train_v12_s1f.parquet --scores data/oof_scores_v12avg.parquet \
    --model models/lgbm_v12avg --band-cache data/s2_band_v12avg.parquet --ce data/ce_oof_v12.parquet --threads 12 > logs/stage2_v12ce_loc.log 2>&1
grep -E "with_p_ce|STAGE-2|SUMMARY|t=0.50" logs/stage2_v12ce_loc.log
```
Note: `models/stage2_v12ce.*` (fitted on the node) is what the test path below uses. The `_loc` run only confirms it; with a different thread count it can differ in the 5th decimal.

## 3. Gate (instant)
```
$P cloud/v12_eval.py --tag v12ce --ref-tag v7ce          # node model
$P cloud/v12_eval.py --tag v12ce_loc --ref-tag v7ce      # local refit (if step 2 was run)
```
**Gate: ≥ +0.0001 on BOTH val and audit vs 0.99031 / 0.99046**, at the fixed decoder (best-reference ownership, alpha 0.7, t 0.50). `v12_eval.py --tag v7ce` reproduces these baseline numbers exactly. If the gate fails, stop: no test build and no submission. Log the numbers.

## 4. Test path (~60-90 min total)
Skip 4a-4c if the node ran `--with-test`; those files came back in the tar.
```
# 4a. stage-1 test scores with the returned full-fit models (105.2M pairs; ~20-35 min each, sequential)
OMP_NUM_THREADS=12 nice -n 10 $P src/predict.py --pairs data/pairs_test_v7_feat.parquet --model models/lgbm_v12  --tag v12  > logs/predict_v12.log 2>&1
OMP_NUM_THREADS=12 nice -n 10 $P src/predict.py --pairs data/pairs_test_v7_feat.parquet --model models/lgbm_v12b --tag v12b > logs/predict_v12b.log 2>&1
# 4b. average (~2 min)
$P src/average_scores.py --a output/test_scores_v12.parquet --b output/test_scores_v12b.parquet --out output/test_scores_v12avg.parquet > logs/average_test_v12avg.log 2>&1
# 4c. CE for the test band entrants (reuses v7 fold-A/B test scores; scores only what is missing; mean of A and B; ~2-5 min on the 5090)
D=$BER_SCRATCH/v12/ce_test
$P src/v11_ce.py plan --scores output/test_scores_v12avg.parquet --kind test --dir $D | tee logs/v12_ce_plan_test.log
for F in A B; do $P src/score_cross_encoder.py --model models/ce_v7f$F --pairs $D/need_$F.parquet --records data/test_records.parquet --out $D/new_$F.parquet --threads 4 > logs/v12_ce_score_test_$F.log 2>&1; done
$P src/v11_ce.py assemble --scores output/test_scores_v12avg.parquet --kind test --dir $D --out output/cef_test_v12avg.parquet | tail -4
# 4d. stage 2 on test (~3-5 min)
nice -n 10 $P src/stage2_apply.py --scores output/test_scores_v12avg.parquet --pairs data/pairs_test_v7_feat.parquet --split test \
    --model models/stage2_v12ce --ce output/cef_test_v12avg.parquet --out output/test_scores_v12ce_s2.parquet --threads 12 > logs/stage2_apply_v12ce.log 2>&1
# 4e. France rules (acronym raise + category-swap drop) + predict + validator + freeze (~10 min) -> submissions/v12_fr/
scripts/chain_rules_on.sh v12 output/test_scores_v12ce_s2.parquet
# 4f. diff vs the champion file
$P logs/probe_diff.py v12_fr v7_s2ce_acr_nocat
```
Checks:
* `chain_rules_on.sh` must print PASS. The candidate file must be identical to v7_s2ce's (same test pairs file).
* In `logs/v12_ce_plan_test.log`, the missing-CE rows must all be scored: assemble asserts 0 missing.
* In the probe_diff output, US/India change counts should be modest. France is untouched apart from what stage 1/2 changes.

## 5. Log (append-only LOG.md)
Record the cloud wall-clock per step (logs/v12/*.log `Elapsed`), peak RSS, and stage-1 A/B val. Record the stage-2 fixed-decoder val/audit and the gate verdict. Also record the coverage table and the cost ($/h × hours from lambda_launch.md). Upload a submission only if the gate passed and Ankit approves; submissions are rationed.
