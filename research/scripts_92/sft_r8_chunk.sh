#!/bin/bash
# sfloor R8 one chunk (FIXED cross-chunk target context builder sft_r8_build.py sha256 646ca9b0bf02cd974645f3363cabc50c43674a9983cf7a7ad55020df9fe57556)
cd ${BER_ROOT:?}; P=${BER_PY:?}; R=${BER_SCRATCH:?}/sfloor/test/r8; k=$1; S=$(date +%s)
export POLARS_MAX_THREADS=6 OMP_NUM_THREADS=6 RAYON_NUM_THREADS=6 SFT_DIR=$R
SHA=$(sha256sum research/scripts_92/sft_r8_build.py | cut -c1-64); rc=0
[ -f $R/new_pairs_r8_c$k.parquet ] || tools/heavy.sh -m 30G $P research/scripts_92/sft_r8_build.py $k > $R/build_c$k.log 2>&1 || rc=1
[ $rc = 0 ] && { tools/heavy.sh -m 30G bash research/scripts_92/sft_newfeat.sh r8_c$k > $R/newfeat_c$k.log 2>&1 || rc=2; }
[ $rc = 0 ] && { tools/heavy.sh -m 30G $P research/scripts_92/sft_r8_s1.py $k > $R/s1_c$k.log 2>&1 || rc=3; }
cnt() { [ -f $1 ] && $P -c "import polars as pl;print(pl.scan_parquet('$1').select(pl.len()).collect().item())" || echo NA; }
IN=$($P -c "import polars as pl;print(pl.scan_parquet('$R/new_base_all.parquet').filter(pl.col('chunk')==$k).select(pl.len()).collect().item())")
echo "chunk $k exit $rc rows_in $IN rows_out $(cnt $R/s1_c$k.parquet) admitted $(cnt $R/adm_c$k.parquet) builder_sha256 $SHA wall $(( $(date +%s)-S ))s end $(date +%T)" >> $R/progress.txt
[ $rc = 0 ] && rm -f $R/new_pairs_r8_c$k.parquet $R/new_feat_r8_c$k.parquet
exit $rc
