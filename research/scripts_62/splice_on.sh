#!/bin/bash
# fr12d: France-only splice onto a new base (US/India from BASE, France from DONOR entry), astra build + check, validator, md5, report.
#   bash research/scripts_62/splice_on.sh <BASE_DIR> <DONOR_DIR> <OUT_DIR>
set -euo pipefail; cd ${BER_ROOT:?}; P=${BER_PY:?}; export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2
B=$1; D=$2; E=$3; CAND=${4:-submissions/v10w_fr/candidate_pairs.tsv}; [ ! -e $E/matching_results.tsv ] || { echo "$E exists"; exit 1; }; mkdir -p $E
nice -n 10 $P research/astra_splice_build.py --finalist $B/matching_results.tsv --control $D/matching_results.tsv --countries data/test_records.parquet --out $E/matching_results.tsv --scratch ${BER_SCRATCH:?} --report $E/splice_report.json > /dev/null
ln -f $CAND $E/candidate_pairs.tsv 2>/dev/null || cp $CAND $E/candidate_pairs.tsv
nice -n 10 $P student_resource/utils/validate_submission.py --matching $E/matching_results.tsv --candidate $E/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids > $E/validate.log 2>&1 || true
grep -q "PASS" $E/validate.log && V=PASS || V=FAIL
nice -n 10 $P research/astra_splice_check.py --finalist $B/matching_results.tsv --splice $E/matching_results.tsv --control $D/matching_results.tsv --countries data/test_records.parquet --scratch ${BER_SCRATCH:?} --report $E/containment_report.json > /dev/null || true
C=$($P -c "import json,sys;r=json.load(open(sys.argv[1]));print(r['status'],'non-France vs base',r['non_france_semantic_changes'],'France vs donor',r['france_control_semantic_mismatches'])" $E/containment_report.json)
(cd $E && md5sum matching_results.tsv candidate_pairs.tsv > md5.txt); M=$(head -1 $E/md5.txt | cut -d' ' -f1)
R=$($P - $B/matching_results.tsv $E/matching_results.tsv <<'PY'
import sys, polars as pl
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
te = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "business_address"]).select(pl.col("entity_id").alias("t_id"), (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("te"))
def load(p):
    d = pl.read_csv(p, separator="\t", quote_char=None, null_values=[""], schema_overrides={"matched_entity_ids": pl.String}).fill_null("")
    return d.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"}).join(cty, on="s1_id").filter(pl.col("country") == "France")
a, b = load(sys.argv[1]), load(sys.argv[2])
def pairs(d): return d.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1_id", pl.col("m").alias("t_id"))
pa, pb = pairs(a), pairs(b); K = ["s1_id", "t_id"]
add = pb.join(pa, on=K, how="anti").join(te, on="t_id", how="left"); rem = pa.join(pb, on=K, how="anti").join(te, on="t_id", how="left")
e = a.select("s1_id", ea=pl.col("m") == "").join(b.select("s1_id", eb=pl.col("m") == ""), on="s1_id")
refs = pl.concat([add.select("s1_id"), rem.select("s1_id")])["s1_id"].n_unique()
print(f"vs base: {refs:,} French refs changed; adds {len(add):,} ({int((~add['te']).sum()):,} non-empty / {int(add['te'].sum()):,} empty target), drops {len(rem):,} ({int((~rem['te']).sum()):,} / {int(rem['te'].sum()):,}); first adds (refs newly non-empty) {int((e['ea'] & ~e['eb']).sum()):,}, emptied refs {int((~e['ea'] & e['eb']).sum()):,}")
PY
)
echo "md5 $M validator $V containment $C; $R"
