"""sfloor R8: base columns (t_ctx.base_cols from original union view rows) for ALL US/India o<=8 new pairs, once, plus the
frozen chunk id = s1_id.hash(seed=92) % 16. -> r8/new_base_all.parquet (s1_id,t_id,country,chunk + base cols)."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, time, os, polars as pl
sys.path.insert(0, (_BER_R + '/research/scripts_rmiss'))
from t_ctx import base_cols, KEYS
os.chdir(_BER_R); T0 = time.time(); O = (_BER_S + '/sfloor/test/r8')
FILES = ['data/cands_sparsev_test.parquet', 'data/cands_dense_test.parquet', 'data/cands_rdense_test.parquet', 'data/cands_rsparse_test.parquet']
scan = lambda: pl.concat([pl.scan_parquet(f).select('s1_id', 't_id', pl.col('view').cast(pl.String), pl.col('score').cast(pl.Float32), pl.col('rank').cast(pl.Int16)) for f in FILES])
K = dict(scan().group_by('view').agg((pl.col('rank').max().cast(pl.Int32) + 1).alias('K')).collect(engine='streaming').iter_rows())
assert K == {'full': 30, 'addr': 30, 'name': 30, 'dense': 30, 'rdense': 10, 'rfull': 5}, K
K['rname'] = 99
outs = []
for cty in ['India', 'US']:
    new = pl.read_parquet((_BER_S + '/sfloor/test_adds_full.parquet')).filter((pl.col('o') <= 8) & (pl.col('country') == cty)).select(KEYS + ['country'])
    U = scan().join(new.lazy().select(KEYS), on=KEYS, how='semi').collect(engine='streaming')
    pure = new.select(KEYS).join(U.select(KEYS).unique(), on=KEYS, how='anti'); assert pure.height == 0
    nb = base_cols(U, pure, K).join(new, on=KEYS); del U
    nb = nb.drop([c for c in nb.columns if c.startswith('rname_')]).with_columns(chunk=(pl.col('s1_id').hash(seed=92) % 16).cast(pl.Int8))
    assert nb.height == new.height
    print(f'[{time.time()-T0:.0f}s] {cty} {nb.height:,}', flush=True); outs.append(nb)
pl.concat(outs).write_parquet(f'{O}/new_base_all.parquet'); print(f'[{time.time()-T0:.0f}s] done', flush=True)
