#!/bin/bash
# LOCAL finish of the clean v12ce (cloud/RETURN.md): wait for the pulled+verified clean tar, unpack, pull the seed-1/2 stage-2 models
# from the node (cloud/clean_s3.sh), exact recheck vs v7ce, 3-seed test side, France rules -> submissions/v12_fr, diff vs v10w_fr.
set -uo pipefail; cd ${BER_ROOT:?}
P=${BER_PY:?}; export HF_HOME=${BER_SCRATCH:?}/hf POLARS_MAX_THREADS=10 OMP_NUM_THREADS=10 TH=10
S="ssh -i ${SSH_KEY:?} -o ConnectTimeout=20 ubuntu@${NODE_IP:?}"; L=logs/v12_local; mkdir -p $L
say() { echo "[$(date '+%F %T') IST] $*"; }; die() { say "FAILED: $*"; exit 1; }
until grep -q "PULLED + md5 OK" logs/cloud_pull_clean.log; do grep -q "PULL FAILED" logs/cloud_pull_clean.log && die "pull failed"; sleep 60; done
if [ ! -f $L/.unpacked ]; then
  say "unpacking"; ls models output | grep -E "v12" | grep -vE "^$" > $L/preexisting.txt || true; cat $L/preexisting.txt
  (cd ${BER_SCRATCH:?} && tar -xf cloud_v12_return.tar --wildcards "data/*") && tar --keep-directory-symlink -xf ${BER_SCRATCH:?}/cloud_v12_return.tar --exclude "data/*" || die "untar"; date > $L/.unpacked; fi
say "exact recheck vs v7ce (both splits)"; $P cloud/v12_eval.py --tag v12ce --ref-tag v7ce | tee $L/gate_exact_v12ce.txt
say "waiting for the node's clean 3-seed fits"; until $S 'test -f amazon/logs/v12/.done_clean_s3' 2>/dev/null; do sleep 120; done
rsync -a -e "ssh -i ${SSH_KEY:?}" ubuntu@${NODE_IP:?}:'amazon/models/stage2_v12ce_seed[12].*' models/ || die "rsync seeds"
rsync -a -e "ssh -i ${SSH_KEY:?}" ubuntu@${NODE_IP:?}:'amazon/output/*_scores_s2_v12ce_s*.parquet' ubuntu@${NODE_IP:?}:amazon/logs/v12/gate_v12ce_s3avg.log output/ || die "rsync s3 outputs"
mv output/gate_v12ce_s3avg.log $L/ 2>/dev/null; $S 'cd amazon; md5sum models/stage2_v12ce.txt models/stage2_v12ce_seed1.txt models/stage2_v12ce_seed2.txt' > $L/node_models.md5
md5sum models/stage2_v12ce.txt models/stage2_v12ce_seed1.txt models/stage2_v12ce_seed2.txt | diff - $L/node_models.md5 && say "seed models md5 match node" || die "seed model md5 mismatch"
say "local 3-seed val/audit read"; $P cloud/v12_eval.py --tag v12ce_s3avg --ref-tag v7ce | tee $L/gate_exact_v12ce_s3avg.txt
say "3-seed test side"; nice -n 10 bash cloud/s2_test.sh v12ce > $L/s2_test_v12ce.log 2>&1 || die "s2_test (see $L/s2_test_v12ce.log)"; tail -3 $L/s2_test_v12ce.log
say "France rules"; bash scripts/chain_rules_on.sh v12 output/test_scores_v12ce_s3avg_s2.parquet > $L/rules_v12.log 2>&1 || die "rules (see $L/rules_v12.log)"; tail -6 $L/rules_v12.log
grep -h "PASS" logs/predict_v12_fr.log | head -2
md5sum submissions/v12_fr/matching_results.tsv
$P logs/probe_diff.py v12_fr v10w_fr | tee $L/diff_v12_fr_v10w_fr.txt
$P - <<'PY' | tee $L/rates.txt
import polars as pl
cty = pl.read_csv("student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
for sub in ("v12_fr", "v10w_fr"):
    m = pl.read_csv(f"submissions/{sub}/matching_results.tsv", separator="\t", quote_char=None, null_values=[]).with_columns(pl.col("matched_entity_ids").fill_null(""))
    m = cty.join(m, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    m = m.with_columns(n=pl.when(pl.col("matched_entity_ids") == "").then(0).otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1))
    g = m.group_by("country").agg(pl.len().alias("S1"), pl.col("n").mean().round(4).alias("matches_per_S1"), (pl.col("n") == 0).mean().round(5).alias("empty_share")).sort("country")
    print(sub); print(g)
PY
say "LOCAL FINISH DONE"
