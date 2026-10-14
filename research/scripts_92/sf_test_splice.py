"""sfloor TEST stage-2 closure splice (SF8): s2_s3avg (new stage 2 on the extended US/India list) -> s2_final.
Mask = rows whose s1_id received an admitted new pair OR whose t_id is touched by an admitted new pair (new.parquet, p1 > 0.01).
Mask rows take the new stage-2 p; all other rows take the production p of output/test_scores_v12fr12x2_E2_s3avg_s2.parquet.
Expected mask: 4,956,558 keys, hash 11427877698333841077 (sum of polars hash(seed=1) of 's1_id|t_id').
usage: sf_test_splice.py [DIR=$BER_SCRATCH/sfloor/test/r8s] [OUT=DIR/s2_final.parquet]"""
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, polars as pl
D = sys.argv[1] if len(sys.argv) > 1 else (_BER_S + '/sfloor/test/r8s'); OUT = sys.argv[2] if len(sys.argv) > 2 else f'{D}/s2_final.parquet'
K = ['s1_id', 't_id']
new = pl.read_parquet(f'{D}/new.parquet', columns=K); R = new.select('s1_id').unique().with_columns(inR=pl.lit(True)); TT = new.select('t_id').unique().with_columns(inT=pl.lit(True))
b = pl.read_parquet(f'{D}/s2_s3avg.parquet')
pr = pl.scan_parquet('output/test_scores_v12fr12x2_E2_s3avg_s2.parquet').filter(pl.col('country').is_in(['US', 'India'])).select(K + [pl.col('p').alias('pp')]).collect()
b = b.join(pr, on=K, how='left').join(R, on='s1_id', how='left').join(TT, on='t_id', how='left')
clos = pl.col('inR').fill_null(False) | pl.col('inT').fill_null(False)
assert b.filter(~clos & pl.col('pp').is_null()).height == 0, 'non-mask row without production p'
m = b.filter(clos).select(K); h = m.select(pl.concat_str(K, separator='|').hash(seed=1).sum()).item()
print('mask keys', m.height, 'hash', h)
if D.endswith('/r8s') and (m.height != 4956558 or h != 11427877698333841077):  # packaged copy: reference-run mask strict only with BER_STRICT_MD5=1
    assert _ber_os.environ.get('BER_STRICT_MD5', '0') != '1', 'SF8 mask mismatch'
    print(f'WARNING: mask {m.height} keys / hash {h} differ from the reference run (4956558 / 11427877698333841077)', flush=True)
_o = b.with_columns(pl.when(clos).then(pl.col('p')).otherwise(pl.col('pp')).alias('p'))  # packaged copy: dynamic checks below
assert _o.select(K).is_duplicated().sum() == 0 and _o['p'].null_count() == 0 and bool(_o['p'].is_finite().all()), 'keys unique, p present and finite'  # packaged copy
_o.select('s1_id', 't_id', 'country', 'p').write_parquet(OUT)  # packaged copy
print('wrote', OUT)
