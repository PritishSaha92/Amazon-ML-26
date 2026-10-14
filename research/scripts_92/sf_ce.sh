#!/bin/bash
# sfloor: X2 = mean of the 4 fr12 members (fr12_ceA, fr12_ceB, fr12s1_ceA, fr12s1_ceB), --batch 1024 --max-len 96 (UB10/X2 convention)
# usage: sf_ce.sh PAIRS RECORDS OUT   (OUT parquet: s1_id,t_id,fr12_ceA,...,x2)
set -e; cd ${BER_ROOT:?}; P=${BER_PY:?}; IN=$1; REC=$2; OUT=$3
export HF_HOME=${BER_SCRATCH:?}/hf HF_HUB_OFFLINE=1 POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4
for m in fr12_ceA fr12_ceB fr12s1_ceA fr12s1_ceB; do
  s=$(date +%s); $P src/score_cross_encoder.py --model ${BER_SCRATCH:?}/models/$m --pairs $IN --records $REC --out $OUT.$m.parquet --threads 4 --max-len 96 --batch 1024 > $OUT.$m.log 2>&1 || { tail -5 $OUT.$m.log; exit 1; }
  echo "$m $(( $(date +%s) - s ))s $(grep THROUGHPUT $OUT.$m.log | cut -c1-120)"
done
$P - $OUT <<'PY'
import sys, polars as pl
o = sys.argv[1]; K = ['s1_id', 't_id']; M = ['fr12_ceA', 'fr12_ceB', 'fr12s1_ceA', 'fr12s1_ceB']
x = pl.read_parquet(f'{o}.{M[0]}.parquet', columns=K + ['p_ce']).rename({'p_ce': M[0]})
for m in M[1:]: x = x.join(pl.read_parquet(f'{o}.{m}.parquet', columns=K + ['p_ce']).rename({'p_ce': m}), on=K)
x = x.with_columns(x2=((pl.col('fr12_ceA') + pl.col('fr12_ceB')) / 2 + (pl.col('fr12s1_ceA') + pl.col('fr12s1_ceB')) / 2) / 2)
x.write_parquet(o); print('wrote', o, x.height)
PY
