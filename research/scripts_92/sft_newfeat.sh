#!/bin/bash
# sfloor TEST step 2 (copy of rmiss t_newfeat.sh, sfloor paths): pair features for the NEW R2 rows with PRODUCTION word-LR tables
# (India/US <- data/tables/wordlr_train.json, France <- data/tables/wordlr_test.json); word-LR route-OFF parity on 300k existing rows.
# usage: sft_newfeat.sh TAG   (TAG = R2 | R2fr)
set -e; cd ${BER_ROOT:?}; R=${SFT_DIR:-${BER_SCRATCH:?}/sfloor/test}; P=${BER_PY:?}; T=${1:-R2}
export POLARS_MAX_THREADS=6 OMP_NUM_THREADS=6
say() { echo "[$(date '+%T')] $*"; }
wl() { $P src/pair_features.py --wordlr-only --pairs $1 --out $2.iu.parquet --split test --wordlr-table-in data/tables/wordlr_train.json --wordlr-countries India,US > $2.iu.log 2>&1
       $P src/pair_features.py --wordlr-only --pairs $2.iu.parquet --out $2 --split test --wordlr-table-in data/tables/wordlr_test.json --wordlr-countries France > $2.fr.log 2>&1; rm -f $2.iu.parquet; }
if [ ! -f ${BER_SCRATCH:?}/sfloor/test/wlpar_ok ]; then
say "wordlr parity"
$P -c "
import polars as pl
d=pl.scan_parquet('data/pairs_test_v10w_feat.parquet').head(6_000_000).collect()
pl.concat([d.filter(pl.col('country')==c).head(100_000) for c in ['France','India','US']]).write_parquet('$R/wlpar_in.parquet')"
wl $R/wlpar_in.parquet $R/wlpar_out.parquet
$P -c "
import polars as pl; K=['s1_id','t_id']
a=pl.read_parquet('$R/wlpar_out.parquet'); b=pl.read_parquet('$R/wlpar_in.parquet'); j=a.join(b,on=K,suffix='_B'); bad=0
for c in a.columns:
    if c in K or not a[c].dtype.is_numeric(): continue
    x=j[c].cast(pl.Float64); y=j[c+'_B'].cast(pl.Float64); d=int((((x-y).abs()>0)|(x.is_null()!=y.is_null())).fill_null(False).sum()); bad+=d
    if d: print('MISMATCH',c,d)
print('word-LR route-OFF parity rows',j.height,'mismatches',bad); assert bad==0"
touch $R/wlpar_ok
fi
say "pair_features new rows"
$P src/pair_features.py --pairs $R/new_pairs_$T.parquet --split test --out $R/new_feat0_$T.parquet --workers 6 --wordlr-table-out $R/wl_tmp_$T.json > $R/pf0_$T.log 2>&1
say "wordlr passes"
wl $R/new_feat0_$T.parquet $R/new_feat_$T.parquet
rm -f $R/new_feat0_$T.parquet
$P -c "
import polars as pl, json
f=json.load(open('models/lgbm_v12.json'))['features']
a=pl.scan_parquet('$R/new_feat_$T.parquet').collect_schema(); b=pl.scan_parquet('data/pairs_test_v10w_feat.parquet').collect_schema()
miss=[c for c in f if c not in a]; bad=[c for c in f if c in a and a[c]!=b[c]]
print('stage-1 feats',len(f),'missing',miss,'dtype mismatch',bad, 'extra cols vs v10w', set(a.names())^set(b.names())); assert not miss and not bad"
say done
