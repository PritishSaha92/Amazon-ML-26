#!/bin/bash
# LOCAL test-side stage 2 for one tag with 3 seeds + average (after the cloud bundles are unpacked):
#   bash cloud/s2_test.sh <tag>      tags: v12ce v12ce2 v12ce3 v12ce4 v12ce5
# Guards (audit M2/M3): the test CE file(s) must correspond to the train CE file(s) recorded in models/stage2_<tag>*.json; the test CE
# must cover 100% of the test band (0.02-0.98 of output/test_scores_v12avg.parquet) with no missing/-1 rows; the averaged output has
# the stage-2 probability in column p and no p_s2.  Then: scripts/chain_rules_on.sh <name> output/test_scores_<tag>_s3avg_s2.parquet
set -euo pipefail; cd "$(dirname "$0")/.."
P=${P:-${BER_PY:?}}; T=${1:?tag}; TP=${TEST_PAIRS:-data/pairs_test_v10w_feat.parquet}
export POLARS_MAX_THREADS=${TH:-10} OMP_NUM_THREADS=${TH:-10}
declare -A MAP=( [data/ce_oof_v12.parquet]=output/cef_test_v12avg.parquet [data/ce2_oof_v12.parquet]=output/ce2_test_v12avg.parquet
                 [data/ce2f_oof_v12.parquet]=output/ce2f_test_v12avg.parquet [data/ce2d_oof_v12.parquet]=output/ce2d_test_v12avg.parquet [data/ce_oof_v12_char.parquet]=output/cef_test_v12_char.parquet
                 [data/ce_oof_v12_D.parquet]=output/cef_test_v12_D.parquet [data/ce_oof_v12_F.parquet]=output/cef_test_v12_F.parquet
                 [data/ce_oof_v12_existing_S3addr.parquet]=output/cef_test_v12_existing_S3addr.parquet [data/ce_oof_v12_existing_S3name.parquet]=output/cef_test_v12_existing_S3name.parquet
                 [data/ce_oof_v12_W.parquet]=output/cef_test_v12_W.parquet [data/ce_oof_v12_new_S3addr.parquet]=output/cef_test_v12_new_S3addr.parquet [data/ce_oof_v12_new_S3name.parquet]=output/cef_test_v12_new_S3name.parquet
                 [data/ce_oof_v12_FRprobe.parquet]=output/cef_test_v12_frce.parquet
                 [data/ce2_oof_v12_frce0.parquet]=output/ce2_test_v12avg_frce0.parquet
                 [data/ce2_oof_v12_frce0id.parquet]=output/ce2_test_v12avg_frce0id.parquet
                 [data/ce2_oof_v12_invUS.parquet]=output/ce2_test_v12avg_invUS.parquet
                 [data/ce2_oof_v12_triA.parquet]=output/ce2_test_v12avg_triA.parquet
                 [data/ce2_oof_v12_FR12H.parquet]=output/ce2_test_v12avg_FR12H.parquet
                 [data/ce2_oof_v12_FR12hG.parquet]=output/ce2_test_v12avg_FR12hG.parquet
                 [data/ce2_oof_v12_triB.parquet]=output/ce2_test_v12avg_triB.parquet
                 [data/ce2_oof_v12_CNT.parquet]=output/ce2_test_v12avg_CNT.parquet
                 [data/ce2_oof_v12_FR12Q.parquet]=output/ce2_test_v12avg_FR12Q.parquet
                 [data/ce2_oof_v12_FR12.parquet]=output/ce2_test_v12avg_FR12.parquet
                 [data/ce2_oof_v12_ST12.parquet]=output/ce2_test_v12avg_ST12.parquet
                 [data/ce2_oof_v12_FR12x2.parquet]=output/ce2_test_v12avg_FR12x2.parquet
                 [data/ce2_oof_v12_CNTx3g.parquet]=output/ce2_test_v12avg_CNTx3g.parquet
                 [data/ce2_oof_v12_ROLEG.parquet]=output/ce2_test_v12avg_ROLEG.parquet
                 [data/ce2_oof_v12_ENS.parquet]=output/ce2_test_v12avg_ENS.parquet
                 [data/ce2_oof_v12_ROLEGgf.parquet]=output/ce2_test_v12avg_ROLEGgf.parquet
                 [data/ce2_oof_v12_FR12x3.parquet]=output/ce2_test_v12avg_FR12x3.parquet
                 [data/ce2_oof_v12_ENSM.parquet]=output/ce2_test_v12avg_ENSM.parquet
                 [data/ce2_oof_v12_EM.parquet]=output/ce2_test_v12avg_EM.parquet
                 [data/ce2_oof_v12_ENSF.parquet]=output/ce2_test_v12avg_ENSF.parquet )
# --extra-feats (research 43 E2 competition features): train file recorded in the model JSON (extra_feats_file) -> test file
declare -A XMAP=( [data/e2_feats_v12.parquet]=data/e2_feats_test_v12.parquet [${BER_SCRATCH:?}/data/e2_feats_v12.parquet]=${BER_SCRATCH:?}/data/e2_feats_test_v12.parquet
                  [data/e2_feats_v12_exact.parquet]=data/e2_feats_test_v12.parquet [${BER_SCRATCH:?}/data/e2_feats_v12_exact.parquet]=${BER_SCRATCH:?}/data/e2_feats_test_v12.parquet )  # _exact: byte-identical copy of e2_feats_v12 (md5 de9e95f5)
for SD in 0 1 2; do TT=$T; [ $SD -gt 0 ] && TT=${T}_seed$SD
  read CE CE2 CE3 TPL LO < <($P -c "import json;m=json.load(open('models/stage2_$TT.json'));print(m.get('ce_file') or '-', m.get('ce2_file') or '-', m.get('ce3_file') or '-', m.get('ce_template') or '-', m.get('ce_band_lo', 0.02))")
  ARGS=""; [ "$TPL" != "-" ] && ARGS="--ce-template $TPL"   # template recorded in the model json (e.g. independent for v12ce_DF)
  for pair in "ce:$CE" "ce2:$CE2" "ce3:$CE3"; do k=${pair%%:*}; f=${pair#*:}; [ "$f" = "-" ] && continue
    tf=${MAP[$f]:-}; [ -n "$tf" ] || { echo "FATAL: no test CE mapping for train CE $f ($TT)"; exit 1; }
    [ -f "$tf" ] || { echo "FATAL: $tf missing"; exit 1; }
    $P - "$tf" "$LO" <<'PY' || { echo "FATAL: coverage check failed for $tf"; exit 1; }
import sys, polars as pl
K = ["s1_id", "t_id"]; f = sys.argv[1]; lo = float(sys.argv[2])   # ce_band_lo from the model JSON (0.02 default; 0.01 for the W band-extension member, strict >)
sc = pl.read_parquet("output/test_scores_v12avg.parquet", columns=K + ["p"])
band = (sc.filter(pl.col("p").is_between(0.02, 0.98)) if lo == 0.02 else sc.filter((pl.col("p") > lo) & (pl.col("p") <= 0.98))).select(K)
ce = pl.read_parquet(f, columns=K + ["p_ce"]); j = band.join(ce, on=K, how="left")
miss = j["p_ce"].null_count(); neg = int((j["p_ce"] < 0).sum())
print(f"{f}: band {len(band):,}, CE rows {len(ce):,}, missing {miss}, negative {neg}"); assert miss == 0 and neg == 0 and len(ce) == len(band)
PY
    ARGS="$ARGS --$k $tf"; done
  XF=$($P -c "import json;m=json.load(open('models/stage2_$TT.json'));print(m.get('extra_feats_file') or '-')")
  if [ "$XF" != "-" ]; then xtf=${XMAP[$XF]:-}; [ -n "$xtf" ] || { echo "FATAL: no test extra-feats mapping for train file $XF ($TT)"; exit 1; }
    [ -f "$xtf" ] || { echo "FATAL: $xtf missing"; exit 1; }
    ARGS="$ARGS --extra-feats $xtf"; fi   # stage2_apply asserts columns == JSON extra_feats and 100% coverage of all rescored rows (p > 0.01)
  echo "[$TT] stage2_apply$ARGS"
  nice -n 10 $P src/stage2_apply.py --scores output/test_scores_v12avg.parquet --pairs $TP --split test --model models/stage2_$TT $ARGS \
      --out output/test_scores_${TT}_s2.parquet --threads ${TH:-10} > logs/stage2_apply_$TT.log 2>&1
done
$P cloud/s2avg.py --test --tag $T
$P -c "import polars as pl;c=pl.read_parquet_schema('output/test_scores_${T}_s3avg_s2.parquet');assert 'p_s2' not in c and 'p' in c, c;print('OK columns', list(c))"
