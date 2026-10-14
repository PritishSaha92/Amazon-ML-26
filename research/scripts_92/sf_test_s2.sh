#!/bin/bash
# sfloor TEST: production stage2_apply (float32) x3 seeds in parallel on US/India R8 inputs, then s2avg_lean arithmetic (float32)
set -euo pipefail; cd ${BER_ROOT:?}; P=${BER_PY:?}; T=${BER_SCRATCH:?}/sfloor/test/r8s
export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 RAYON_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
for SD in 0 1 2; do M=models/stage2_v12ce2_E2; [ $SD -gt 0 ] && M=${M}_seed$SD
  tools/heavy.sh -m 24G $P src/stage2_apply.py --scores $T/scores.parquet --pairs $T/s1f.parquet --split test --model $M --ce $T/ce.parquet --extra-feats $T/e2.parquet --out $T/s2_seed$SD.parquet --threads 3 > $T/s2_seed$SD.log 2>&1 &
done; wait
for SD in 0 1 2; do grep -q "^wrote" $T/s2_seed$SD.log || { echo "seed $SD failed"; tail -5 $T/s2_seed$SD.log; exit 1; }; done
tools/heavy.sh -m 38G $P - <<'PY'
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import polars as pl
T = (_BER_S + '/sfloor/test/r8s'); K = ['s1_id', 't_id']
d = [pl.read_parquet(f'{T}/s2_seed{s}.parquet') for s in range(3)]
for x in d[1:]: assert x.select(K).equals(d[0].select(K))
b = d[0].with_columns(((pl.col('p') + d[1]['p'] + d[2]['p']) / 3).cast(d[0].schema['p']).alias('p')); del d
b.write_parquet(f'{T}/s2_s3avg.parquet'); print('wrote s3avg', b.height)
PY
echo S2DONE
