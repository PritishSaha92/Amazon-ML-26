cd ${BER_ROOT:?}
P=${BER_PY:?}; L=research/scripts_43
export OMP_NUM_THREADS=8
S2="--band data/s2_band_v12avg.parquet --ce data/ce_oof_v12.parquet --scores data/oof_scores_v12avg.parquet --base-cfg models/stage2_v12ce.json"
say() { echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f "tag r43_v12compPCb" > /dev/null; do sleep 20; done; say "PC-b finished"
# exact features (2 polars threads) in parallel with the c01 sensitivity fits (8 threads)
# packaged copy: waits for the rescore file itself
( until [ -f data/v12/r43_rescore_full.parquet ]; do sleep 60; done
  say "rescore_full landed"; POLARS_MAX_THREADS=2 nice -n 10 $P $L/e2_feats_exact.py --rescore data/v12/r43_rescore_full.parquet --tmp ${BER_SCRATCH:?}/r43/tmp_exact \
     --out ${BER_SCRATCH:?}/data/e2_feats_v12_exact.parquet > $L/e2_v12_exact_feats.log 2>&1 && say "exact feats OK" || say "EXACT FEATS FAILED" ) &
FE=$!
POLARS_MAX_THREADS=8 nice -n 10 $P $L/s2fit.py $S2 --comp data/r43_comp_v12_pc_c01.parquet --tag r43_v12compPCc01 --seeds 0,1,2 > $L/e2_v12compPCc01.log 2>&1
POLARS_MAX_THREADS=8 nice -n 10 $P $L/paired.py r43_v12compPCc01_s3avg v12ce_s3avg; POLARS_MAX_THREADS=8 nice -n 10 $P $L/paired.py r43_v12compPCc01_s3avg r43_v12compPC_s3avg
say "c01 done"
wait $FE; [ -f ${BER_SCRATCH:?}/data/e2_feats_v12_exact.parquet ] || { say "no exact features"; exit 1; }
POLARS_MAX_THREADS=8 nice -n 10 $P $L/s2fit.py $S2 --comp ${BER_SCRATCH:?}/data/e2_feats_v12_exact.parquet --tag r43_v12compEX --seeds 0,1,2 > $L/e2_v12compEX.log 2>&1
POLARS_MAX_THREADS=8 nice -n 10 $P $L/paired.py r43_v12compEX_s3avg v12ce_s3avg; POLARS_MAX_THREADS=8 nice -n 10 $P $L/paired.py r43_v12compEX_s3avg r43_v12compPC_s3avg
POLARS_MAX_THREADS=8 nice -n 10 $P $L/owner_eval.py --universe data/oof_scores_v12avg.parquet --tags v12ce_s3avg,r43_v12compEX_s3avg
say "EXACT_DONE"
# combos via src/stage2.py --extra-feats (honours the CE/CE2/CE3/field blocks exactly), audit withheld, 3 seeds each
export STAGE2_WITHHOLD_AUDIT=1 POLARS_MAX_THREADS=8
B="--pairs data/v12/pairs_train_v12_s1f.parquet --scores data/oof_scores_v12avg.parquet --model models/lgbm_v12avg --band-cache data/s2_band_v12avg.parquet --threads 8 --extra-feats ${BER_SCRATCH:?}/data/e2_feats_v12_exact.parquet"
for SD in 0 1 2; do T=v12ce2_E2; [ $SD = 0 ] || T=v12ce2_E2_seed$SD
  nice -n 10 $P src/stage2.py --tag $T --seed $SD --ce data/ce2_oof_v12.parquet $B > $L/combo_$T.log 2>&1 || say "FAILED $T"; done
$P cloud/s2avg.py --tag v12ce2_E2 --splits val > /dev/null && for R in v12ce_s3avg v12ce2_s3avg r43_v12compEX_s3avg; do POLARS_MAX_THREADS=8 nice -n 10 $P $L/paired.py v12ce2_E2_s3avg $R; done
say "COMBO1_DONE"
for SD in 0 1 2; do T=v12ce2_E2_JF; [ $SD = 0 ] || T=v12ce2_E2_JF_seed$SD
  nice -n 10 $P src/stage2.py --tag $T --seed $SD --ce data/ce2_oof_v12.parquet --ce2 data/ce_oof_v12_existing_S3addr.parquet --ce3 data/ce_oof_v12_existing_S3name.parquet --field-disagreement $B > $L/combo_$T.log 2>&1 || say "FAILED $T"; done
$P cloud/s2avg.py --tag v12ce2_E2_JF --splits val > /dev/null && for R in v12ce_s3avg v12ce2_s3avg v12ce2_E2_s3avg; do POLARS_MAX_THREADS=8 nice -n 10 $P $L/paired.py v12ce2_E2_JF_s3avg $R; done
say "COMBO2_DONE"
