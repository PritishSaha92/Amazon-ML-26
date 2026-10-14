cd ${BER_ROOT:?}; P=${BER_PY:?}; export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2
B=submissions/v12fr12x2_E2_frENS_s
for E in UB10all frUB10 UB10ui; do D=submissions/v12fr12x2_E2_${E}_s
  ln -f submissions/v10w_fr/candidate_pairs.tsv $D/candidate_pairs.tsv
  nice -n 10 $P student_resource/utils/validate_submission.py --matching $D/matching_results.tsv --candidate $D/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids > $D/validate.log 2>&1
  echo "$E validator: $(grep -o 'PASS.*\|FAIL.*' $D/validate.log | tail -1)"
  (cd $D && md5sum matching_results.tsv candidate_pairs.tsv > md5.txt && head -1 md5.txt)
done
# France-only file: France must equal UB10all's France, non-France must equal the base
nice -n 10 $P research/astra_splice_check.py --finalist $B/matching_results.tsv --splice submissions/v12fr12x2_E2_frUB10_s/matching_results.tsv --control submissions/v12fr12x2_E2_UB10all_s/matching_results.tsv --countries data/test_records.parquet --scratch ${BER_SCRATCH:?} --report submissions/v12fr12x2_E2_frUB10_s/containment_report.json > /dev/null
$P -c "import json;r=json.load(open('submissions/v12fr12x2_E2_frUB10_s/containment_report.json'));print('frUB10 containment',r['status'],'non-France vs base',r['non_france_semantic_changes'],'France vs UB10all',r['france_control_semantic_mismatches'],r['changed_s1_by_country'])"
$P - <<'PY'
import polars as pl
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
def rd(p): return pl.read_csv(p, separator="\t", quote_char=None, null_values=[""], schema_overrides={"matched_entity_ids": pl.String}).fill_null("").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"})
b = rd("submissions/v12fr12x2_E2_frENS_s/matching_results.tsv")
for e in ("UB10all", "UB10ui", "frUB10"):
    x = rd(f"submissions/v12fr12x2_E2_{e}_s/matching_results.tsv"); assert x["s1_id"].equals(b["s1_id"]), "row order"
    j = b.join(x, on="s1_id", suffix="_n").join(cty, on="s1_id").filter(pl.col("m") != pl.col("m_n"))
    print(e, "row diffs vs base by country:", dict(j.group_by("country").len().rows()), "rows", len(x))
PY
