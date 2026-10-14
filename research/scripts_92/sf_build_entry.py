"""sfloor: build submissions/v12fr12x2_E2_UB10ui_SF8_s from the base UB10ui_s (a7edfe5d...) + US/India decode of R8.
Rows of US/India refs whose decode differs are rewritten (ids sorted, as predict.py); all other rows byte-identical (France untouched).
candidate_pairs.tsv = base candidate rows, with admitted new R8 pairs (US/India, p1 > 0.01) merged (sorted unique) into their rows.
Also writes the change list parquet (s1_id, country, old, new)."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, hashlib, polars as pl
os.chdir(_BER_R); dec, new_f = sys.argv[1], sys.argv[2]
BASE = 'submissions/v12fr12x2_E2_UB10ui_s'; OUT = 'submissions/v12fr12x2_E2_UB10ui_SF8_s'
_m = hashlib.md5(open(f'{BASE}/matching_results.tsv', 'rb').read()).hexdigest()  # packaged copy: strict only with BER_STRICT_MD5=1
if _m != 'a7edfe5da3b11d1c1182d98ce1cb0e62':
    assert os.environ.get('BER_STRICT_MD5', '0') != '1', _m
    print(f'WARNING: {BASE} md5 {_m} differs from the reference a7edfe5d (expected after retraining)', flush=True)
assert not os.path.exists(OUT); os.makedirs(OUT)
cty = pl.read_parquet('data/test_records.parquet', columns=['entity_id', 'country']).rename({'entity_id': 's1_id'})
cmap = dict(zip(cty['s1_id'], cty['country']))
b = pl.read_csv(f'{BASE}/matching_results.tsv', separator='\t', quote_char=None, schema_overrides={'matched_entity_ids': pl.String}).fill_null('')
b = b.rename({'source1_entity_id': 's1_id', 'matched_entity_ids': 'old'}).join(cty, on='s1_id').filter(pl.col('country').is_in(['US', 'India']))
d = b.join(pl.read_parquet(dec), on='s1_id', how='left').with_columns(pl.col('m').fill_null('')).filter(pl.col('m') != pl.col('old'))
ch = d.select('s1_id', 'country', 'old', pl.col('m').alias('new')); ch.write_parquet(f'{_BER_S}/sfloor/SF8_changes_ui.parquet')
nd = dict(zip(ch['s1_id'], ch['new']))
nw = pl.read_parquet(new_f, columns=['s1_id', 't_id', 'country']).filter(pl.col('country').is_in(['US', 'India'])).group_by('s1_id').agg(pl.col('t_id'))
nc = dict(zip(nw['s1_id'], nw['t_id']))
n = 0
with open(f'{BASE}/matching_results.tsv', 'rb') as fi, open(f'{OUT}/matching_results.tsv', 'wb') as fo:
    fo.write(fi.readline())
    for line in fi:
        s1 = line.decode().rstrip('\r\n').split('\t')[0]
        if s1 in nd: fo.write(f'{s1}\t{nd[s1]}\n'.encode()); n += 1
        else: fo.write(line)
assert n == len(nd)
nc_rows = 0
with open(f'{BASE}/candidate_pairs.tsv', 'rb') as fi, open(f'{OUT}/candidate_pairs.tsv', 'wb') as fo:
    fo.write(fi.readline())
    for line in fi:
        s1, ids = line.decode().rstrip('\r\n').split('\t')
        if s1 in nc:
            u = sorted(set([t for t in ids.split(',') if t]) | set(nc[s1])); fo.write(f'{s1}\t{",".join(u)}\n'.encode()); nc_rows += 1
        else: fo.write(line)
assert nc_rows == len(nc)
# stats
def ex(df, col): return df.select('s1_id', 'country', pl.col(col).str.split(',')).explode(col).filter(pl.col(col) != '').rename({col: 't_id'})
o, w = ex(ch, 'old'), ex(ch, 'new')
add = w.join(o, on=['s1_id', 't_id'], how='anti'); rem = o.join(w, on=['s1_id', 't_id'], how='anti')
print('refs changed', ch.height, ch.group_by('country').len().sort('country').rows())
print('added pairs', add.group_by('country').len().sort('country').rows(), 'removed pairs', rem.group_by('country').len().sort('country').rows())
print('refs emptied', ch.filter(pl.col('new') == '').height, '; candidate rows extended', nc_rows)
for f in ('matching_results.tsv', 'candidate_pairs.tsv'): print(f, hashlib.md5(open(f'{OUT}/{f}', 'rb').read()).hexdigest())
