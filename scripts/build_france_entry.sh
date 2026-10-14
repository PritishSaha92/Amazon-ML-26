#!/bin/bash
# Build one France-tournament entry = locked finalist v12ce2_E2 (US/India) + France rescored by a given France CE.
#
#   Usage:  bash scripts/build_france_entry.sh <NAME> <FRANCE_CE_FILE>
#           bash scripts/build_france_entry.sh -h
#     NAME            [A-Za-z0-9]+ ; stage-2 tag <NAME>ce2_E2 ; entry submissions/v12ce2_E2_fr<NAME>
#     FRANCE_CE_FILE  parquet with s1_id, t_id, p_ce (or p) covering EXACTLY the 261,127 French rows of the CE band
#   Env (optional):
#     TH=3            threads for polars/OMP (default 3; local cap)
#     ENTRY_DIR=...   write the entry somewhere else (e.g. scratch for a certification run); must not exist
#     NO_LEDGER=1     do not append to LOG.md / research/FRANCE_TOURNAMENT.md (test runs)
#   Run detached:  setsid nohup bash scripts/build_france_entry.sh NAME FILE > logs/france_entries/NAME_run.log 2>&1 < /dev/null &
#
# Steps: fail-fast input checks -> validate FRANCE_CE_FILE -> output/ce2_test_v12avg_<NAME>.parquet (France p_ce swapped, non-France
#   rows asserted identical) -> tag copies of the finalist's 3 stage-2 models (only ce_file changed) + s2_test.sh MAP line -> s2_test
#   -> chain_rules_on (acronym raise, catswap drop, predict) -> France splice onto the finalist (astra_splice_build.py)
#   -> official validator --check-ids (FATAL on fail) -> containment check astra_splice_check.py vs submissions/v12ce2_E2_fr with the
#   control (FATAL unless 0 non-France changes and 0 France/control mismatches) -> md5 + changed-France pair/ref counts
#   -> cleanup of per-seed/_acr/_fr intermediates -> one line in LOG.md + row in research/FRANCE_TOURNAMENT.md (unless NO_LEDGER=1).
# Note: France is rescored inside the FINALIST's stage 2 (v12ce2_E2 boosters + E2 features); only the France CE column changes.
#   This is NOT the recipe of v12ce2_E2_frFRCE (whose France came from the older v12ce stage 2, v12ceFR_fr).
# Certified 27 Sep (frce0 identity run: finalist's own France CE -> byte-identical finalist; see LOG.md "builder certified").
# Logs: logs/france_entries/<NAME>_*.log
set -euo pipefail; cd ${BER_ROOT:?}
usage() { sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'; }
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage; exit 0; }
[ $# -eq 2 ] || { usage; exit 2; }
P=${BER_PY:?}
TH=${TH:-3}; export TH POLARS_MAX_THREADS=$TH OMP_NUM_THREADS=$TH RAYON_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=1
NAME=$1; FCE=$2
[[ $NAME =~ ^[A-Za-z0-9]+$ ]] || { echo "NAME must be alphanumeric"; exit 2; }
TAG=${NAME}ce2_E2; FIN=submissions/v12ce2_E2_fr; ENTRY=${ENTRY_DIR:-submissions/v12ce2_E2_fr${NAME}}; L=logs/france_entries; mkdir -p $L
BASE=output/ce2_test_v12avg.parquet; NEWCE=output/ce2_test_v12avg_${NAME}.parquet; OOF=data/ce2_oof_v12_${NAME}.parquet
say() { echo "[$(date '+%F %T')] $*"; }; die() { say "FAILED: $*"; exit 1; }
T0=$(date +%s)
# fail-fast: inputs, locked finalist, no clobbering
[ -f "$FCE" ] || die "missing France CE file $FCE"
for f in $BASE data/test_records.parquet data/ce2_oof_v12.parquet data/e2_feats_test_v12.parquet data/pairs_test_v10w_feat.parquet \
         data/pairs_test_v7_feat.parquet data/views_test.parquet data/tables/wordlr_test.json output/test_scores_v12avg.parquet \
         models/lgbm_v7avg.json models/stage2_v12ce2_E2{,_seed1,_seed2}.{txt,json} $FIN/matching_results.tsv \
         submissions/v10w_fr/candidate_pairs.tsv submissions/v7_s2ce/candidate_pairs.tsv cloud/s2_test.sh scripts/chain_rules_on.sh \
         research/astra_splice_build.py research/astra_splice_check.py student_resource/utils/validate_submission.py; do
  [ -e "$f" ] || die "missing required input $f"; done
[ -e "$ENTRY" ] && die "$ENTRY already exists (refusing to write into an existing submission folder)"
[ -e submissions/${TAG}_fr ] && die "submissions/${TAG}_fr already exists (chain_rules_on would overwrite it; pick a new NAME)"
for f in $NEWCE models/stage2_${TAG}.json output/test_scores_${TAG}_s3avg_s2.parquet output/test_scores_${TAG}_fr.parquet; do
  [ -e "$f" ] && die "$f already exists (NAME $NAME was used before; pick a new NAME)"; done
[ "$(md5sum < $FIN/matching_results.tsv | cut -d' ' -f1)" = fb0f90f3d79f625c2e6adb3bd1d39b06 ] || { [ "${BER_STRICT_MD5:-0}" = 1 ] && die "finalist md5 changed"; say "WARNING: $FIN differs from the reference md5 fb0f90f3 (expected after retraining)"; }  # packaged copy: strict only with BER_STRICT_MD5=1
FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$FREE" -ge 15 ] || die "root free ${FREE}G < 15G"

# (1)+(2) validate the France CE file and write the swapped test CE file
say "(1-2) validate $FCE and write $NEWCE"
SHIFT=$($P - "$FCE" "$BASE" "$NEWCE" <<'PY' 2> $L/${NAME}_ce.log
import sys, numpy as np, polars as pl
fce, basef, outf = sys.argv[1:4]; K = ["s1_id", "t_id"]
f = pl.read_parquet(fce); pc = "p_ce" if "p_ce" in f.columns else ("p" if "p" in f.columns else None)
assert pc and all(c in f.columns for c in K), f"need s1_id, t_id, p_ce|p; got {f.columns}"
f = f.select(K + [pl.col(pc).cast(pl.Float64).alias("p_new")])
assert f.select(K).is_duplicated().sum() == 0, "duplicate keys in France CE file"
v = f["p_new"].to_numpy(); assert f["p_new"].null_count() == 0 and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), "p not finite in [0,1]"
base = pl.read_parquet(basef)
fr_ids = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1_id"))
bfr = base.join(fr_ids, on="s1_id", how="semi")
miss = bfr.join(f, on=K, how="anti").height; extra = f.join(bfr, on=K, how="anti").height
print(f"France band rows {bfr.height:,}; file rows {f.height:,}; missing {miss}; extra {extra}", file=sys.stderr)
assert bfr.height == f.height and miss == 0 and extra == 0 and (f.height == 261127 or __import__("os").environ.get("BER_STRICT_MD5", "0") != "1"), "coverage mismatch"  # packaged copy: reference-run count strict only with BER_STRICT_MD5=1
j = base.with_row_index("_r").join(f, on=K, how="left").sort("_r")
out = j.with_columns(p_ce=pl.coalesce("p_new", "p_ce").cast(base["p_ce"].dtype)).select(base.columns)
assert out.height == base.height and out.select(K).equals(base.select(K))
isfr = j["p_new"].is_not_null(); assert int(isfr.sum()) == f.height  # packaged copy: every France key replaced exactly once
assert out.filter(~isfr).equals(base.filter(~isfr)), "non-France rows changed"
o = base.filter(isfr)["p_ce"].to_numpy().astype(np.float64); n = out.filter(isfr)["p_ce"].to_numpy().astype(np.float64)
w = out["p_ce"].to_numpy(); assert np.isfinite(w).all() and (w >= 0).all() and (w <= 1).all()
out.write_parquet(outf)
s = (f"France p_ce mean {o.mean():.5f}->{n.mean():.5f}, >0.5 {(o > .5).mean():.4f}->{(n > .5).mean():.4f}, "
     f"cross0.5 up {((o <= .5) & (n > .5)).mean():.4f} down {((o > .5) & (n <= .5)).mean():.4f}")
print(s, file=sys.stderr); print(s)
PY
) || { cat $L/${NAME}_ce.log; die "CE validation/assembly"; }
cat $L/${NAME}_ce.log; say "$NEWCE md5 $(md5sum < $NEWCE | cut -d' ' -f1)"

# (3) tag copies of the finalist's models, OOF symlink, MAP line
say "(3) models/stage2_${TAG}{,_seed1,_seed2} + $OOF + MAP"
ln -sfn ce2_oof_v12.parquet $OOF
for X in "" _seed1 _seed2; do cp models/stage2_v12ce2_E2$X.txt models/stage2_$TAG$X.txt
  $P - models/stage2_v12ce2_E2$X.json models/stage2_$TAG$X.json $OOF <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); assert m["ce_file"] == "data/ce2_oof_v12.parquet", m["ce_file"]
n = dict(m); n["ce_file"] = sys.argv[3]; json.dump(n, open(sys.argv[2], "w"), indent=1)
chk = json.load(open(sys.argv[2])); assert {k for k in set(m) | set(chk) if m.get(k) != chk.get(k)} == {"ce_file"}
PY
done
if ! grep -qF "[$OOF]=" cloud/s2_test.sh; then
  $P - "$OOF" "$NEWCE" <<'PY'
import sys, re
k, v = sys.argv[1:3]; src = open("cloud/s2_test.sh").read().split("\n")
i = next(i for i, l in enumerate(src) if l.startswith("declare -A MAP=("))
j = next(j for j in range(i, len(src)) if src[j].rstrip().endswith(" )"))
src[j] = src[j].rstrip()[:-2].rstrip(); src.insert(j + 1, f"                 [{k}]={v} )")
open("cloud/s2_test.sh", "w").write("\n".join(src))
PY
fi
grep -qF "[$OOF]=$NEWCE" cloud/s2_test.sh || die "MAP patch"
bash -n cloud/s2_test.sh || die "s2_test.sh syntax after MAP patch"

# (4) test-side stage 2, rules, splice, validator
FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$FREE" -ge 15 ] || die "root free ${FREE}G < 15G"
say "(4a) s2_test $TAG"
TH=$TH bash cloud/s2_test.sh $TAG > $L/${NAME}_s2_test.log 2>&1 || { tail -5 $L/${NAME}_s2_test.log; die "s2_test (see $L/${NAME}_s2_test.log)"; }
grep -E "band|OK col|wrote" $L/${NAME}_s2_test.log
say "(4b) chain_rules_on $TAG"
# chain_rules_on.sh hard-exports 8 threads; run it unchanged except for honouring our thread cap
RULES_SRC=$(sed "s/^export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8$/export POLARS_MAX_THREADS=$TH OMP_NUM_THREADS=$TH/" scripts/chain_rules_on.sh)
grep -q "^export POLARS_MAX_THREADS=$TH OMP_NUM_THREADS=$TH$" <<< "$RULES_SRC" || die "chain_rules_on.sh thread line not found"
bash <(echo "$RULES_SRC") $TAG output/test_scores_${TAG}_s3avg_s2.parquet > $L/${NAME}_rules.log 2>&1 || { tail -5 $L/${NAME}_rules.log; die "rules (see $L/${NAME}_rules.log)"; }
say "(4c) splice France of ${TAG}_fr onto $FIN -> $ENTRY"
mkdir -p "$ENTRY"
POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 $P research/astra_splice_build.py --finalist $FIN/matching_results.tsv \
  --control submissions/${TAG}_fr/matching_results.tsv --countries data/test_records.parquet --out $ENTRY/matching_results.tsv \
  --scratch ${BER_SCRATCH:?} --report $ENTRY/splice_report.json > $L/${NAME}_splice.log 2>&1 || die "splice (see $L/${NAME}_splice.log)"
ln -f submissions/v10w_fr/candidate_pairs.tsv $ENTRY/candidate_pairs.tsv 2>/dev/null || cp submissions/v10w_fr/candidate_pairs.tsv $ENTRY/candidate_pairs.tsv
say "(4d) official validator --check-ids"
nice -n 10 $P student_resource/utils/validate_submission.py --matching $ENTRY/matching_results.tsv --candidate $ENTRY/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test --check-ids > $L/${NAME}_validate.log 2>&1 && VRC=0 || VRC=$?
tail -3 $L/${NAME}_validate.log
if [ $VRC -eq 0 ] && grep -q PASS $L/${NAME}_validate.log; then VAL=PASS; else die "validator FAIL rc=$VRC (see $L/${NAME}_validate.log)"; fi
say "(4e) containment check vs $FIN (control submissions/${TAG}_fr)"
POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 $P research/astra_splice_check.py --finalist $FIN/matching_results.tsv \
  --splice $ENTRY/matching_results.tsv --control submissions/${TAG}_fr/matching_results.tsv --countries data/test_records.parquet \
  --scratch ${BER_SCRATCH:?} --report $ENTRY/containment_report.json > $L/${NAME}_containment.log 2>&1 || die "containment FAIL (see $L/${NAME}_containment.log)"
CONT=$($P -c "import json,sys;r=json.load(open(sys.argv[1]));c=r['changed_s1_by_country'];assert r['status']=='PASS' and r['non_france_semantic_changes']==0 and r['france_control_semantic_mismatches']==0 and c['India']==0 and c['US']==0;print(f\"containment PASS: 0 non-France changes, France refs changed {c['France']:,}\")" $ENTRY/containment_report.json) || die "containment report not clean"
echo "$CONT"
(cd $ENTRY && md5sum matching_results.tsv candidate_pairs.tsv > md5.txt && cat md5.txt)
MD5=$(head -1 $ENTRY/md5.txt | cut -d' ' -f1)

# (5) cleanup (keep the s3avg)
rm -f output/test_scores_${TAG}_s2.parquet output/test_scores_${TAG}_seed1_s2.parquet output/test_scores_${TAG}_seed2_s2.parquet \
      output/test_scores_${TAG}_acr.parquet output/test_scores_${TAG}_fr.parquet output/matching_results_${TAG}_fr.tsv output/candidate_pairs_${TAG}_fr.tsv

# (6) France-only diff vs the finalist, LOG.md line, tournament row
DIFF=$($P - $FIN/matching_results.tsv $ENTRY/matching_results.tsv <<'PY'
import sys, polars as pl
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
def load(p):
    d = pl.read_csv(p, separator="\t", quote_char=None, null_values=[""], schema_overrides={"matched_entity_ids": pl.String}).fill_null("")
    return d.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"}).join(cty, on="s1_id", how="left")
a, b = load(sys.argv[1]), load(sys.argv[2])
na, nb = a.filter(pl.col("country") != "France"), b.filter(pl.col("country") != "France")
nonfr = na.join(nb, on=["s1_id", "m"], how="anti").height + nb.join(na, on=["s1_id", "m"], how="anti").height
def pairs(d): return d.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1_id", "m")
fa, fb = a.filter(pl.col("country") == "France"), b.filter(pl.col("country") == "France")
pa, pb = pairs(fa), pairs(fb)
add = pb.join(pa, on=["s1_id", "m"], how="anti").height; rem = pa.join(pb, on=["s1_id", "m"], how="anti").height
e = fa.select("s1_id", ea=pl.col("m") == "").join(fb.select("s1_id", eb=pl.col("m") == ""), on="s1_id")
print(f"France vs finalist: +{add} -{rem} pairs, refs newly non-empty {int((e['ea'] & ~e['eb']).sum())} emptied {int((~e['ea'] & e['eb']).sum())}, "
      f"matches/S1 {len(pa)/len(fa):.4f}->{len(pb)/len(fb):.4f}, empty {e['ea'].mean():.5f}->{e['eb'].mean():.5f}; non-France row diffs {nonfr}")
PY
) || die "diff"
echo "$DIFF"
SEC=$(( $(date +%s) - T0 ))
LINE="- $(date '+%H:%M') FRANCE ENTRY $ENTRY md5 $MD5 validator $VAL, $CONT (scripts/build_france_entry.sh; France CE $FCE; stage 2 = finalist's 3 seeds via tag $TAG). $DIFF. CE shift: $SHIFT. Built in ${SEC}s."
echo "$LINE"
if [ "${NO_LEDGER:-0}" != 1 ]; then
  echo "$LINE" >> LOG.md
  echo "| $(basename $ENTRY) | $MD5 | $FCE (France rows rescored, finalist v12ce2_E2 stage 2) | $SHIFT; $DIFF; validator $VAL; $CONT | built |" >> research/FRANCE_TOURNAMENT.md
fi
say "DONE $ENTRY md5 $MD5 in ${SEC}s"
