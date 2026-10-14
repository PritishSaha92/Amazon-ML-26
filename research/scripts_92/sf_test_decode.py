"""sfloor TEST decode (predict.py ownership arithmetic, alpha 0.7, t 0.50) for US/India on a stage-2 scores file, then UB10.
usage: sf_test_decode.py SCORES OUT_PARQUET [UBX2_EXTRA]   -> per-ref sorted match string for US/India refs; compare/parity vs UB10ui_s.
UB10: drop accepted p1>0.98 keys with 4-member x2 < 0.10; x2 from fr12d test_scored.parquet (+ UBX2_EXTRA); keys lacking x2 -> fail, written out."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, polars as pl
os.chdir(_BER_R); K = ['s1_id', 't_id']; T, AL = 0.50, 0.7
sc, out = sys.argv[1], sys.argv[2]; extra = sys.argv[3] if len(sys.argv) > 3 else ''
x = pl.scan_parquet(sc).select(K + ['country', 'p']).filter(pl.col('country').is_in(['US', 'India'])).collect()
x = x.sort(['t_id', 'p'], descending=[False, True]).with_columns(pl.int_range(pl.len()).over('t_id').alias('r')).filter(pl.col('r') == 0)
x = x.with_columns(pl.col('p').max().over('s1_id').alias('pmax'))
sel = x.filter((pl.col('p') >= T) & (pl.col('p') >= AL * pl.col('pmax'))).select(K + ['country'])
P1 = os.environ.get('SF_P1', 'output/test_scores_v12avg.parquet')   # complete hybrid stage-1 scores (old + new)
p1 = pl.scan_parquet(P1).select(K + [pl.col('p').alias('p1')]).join(sel.lazy(), on=K, how='semi').collect(engine='streaming')
assert p1.select(K).is_duplicated().sum() == 0
sel = sel.join(p1, on=K, how='left'); assert sel['p1'].null_count() == 0 and sel['p1'].is_nan().sum() == 0, 'accepted keys lack p1'
x_ps2 = x.select(K + [pl.col('p').alias('p_s2')]); sel = sel.join(x_ps2, on=K, how='left')
sel.write_parquet(out + '.accepted_preUB10.parquet')
up = sel.filter(pl.col('p1') > 0.98)
xs = [pl.read_parquet((_BER_S + '/fr12d/ub/test_scored.parquet'), columns=K + ['fr12_ceA', 'x2'])]
if extra: xs.append(pl.read_parquet(extra, columns=K + ['fr12_ceA', 'x2']))
xs = pl.concat([d.with_columns(pl.col('fr12_ceA').cast(pl.Float64), pl.col('x2').cast(pl.Float64)) for d in xs]).unique(subset=K, keep='first')
u = up.join(xs, on=K, how='left'); miss = u.filter(pl.col('fr12_ceA').is_null())
if miss.height:
    miss.select(K).write_parquet(out + '.ub_missing.parquet'); print('UB10 missing member scores for', miss.height, '->', out + '.ub_missing.parquet'); sys.exit(2)
# cascade convention: fr12_ceA >= 0.40 -> x2 null (cannot be < 0.10) -> never dropped
u.write_parquet(out + '.upper_ledger.parquet')
drop = u.filter(pl.col('x2').is_not_null() & (pl.col('x2') < 0.10)).select(K)
fin = sel.join(drop, on=K, how='anti').select(K + ['country'])
print('accepted', sel.height, 'upper', up.height, 'UB10 drops', drop.height, sel.join(drop, on=K, how='semi').group_by('country').len().rows())
fin.group_by('s1_id').agg(pl.col('t_id').sort().str.join(',').alias('m')).write_parquet(out)
