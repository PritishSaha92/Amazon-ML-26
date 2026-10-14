"""sfloor SF8a ('adds only'): base UB10ui_s (a7edfe5d...) with every base match kept; add a pair accepted in the full R8 decode after UB10 only if
its target is matched to NO reference in the base file. Also: cause split of SF8's removals."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, hashlib, polars as pl
os.chdir(_BER_R); K = ['s1_id', 't_id']; T = (_BER_S + '/sfloor/test/r8s')
BASE = 'submissions/v12fr12x2_E2_UB10ui_s'; OUT = 'submissions/v12fr12x2_E2_UB10ui_SF8a_s'
_m = hashlib.md5(open(f'{BASE}/matching_results.tsv', 'rb').read()).hexdigest()  # packaged copy: strict only with BER_STRICT_MD5=1
if _m != 'a7edfe5da3b11d1c1182d98ce1cb0e62':
    assert os.environ.get('BER_STRICT_MD5', '0') != '1', _m
    print(f'WARNING: {BASE} md5 {_m} differs from the reference a7edfe5d (expected after retraining)', flush=True)
assert not os.path.exists(OUT); os.makedirs(OUT)
cty = pl.read_parquet('data/test_records.parquet', columns=['entity_id', 'country']).rename({'entity_id': 's1_id'})
b = pl.read_csv(f'{BASE}/matching_results.tsv', separator='\t', quote_char=None, schema_overrides={'matched_entity_ids': pl.String}).fill_null('')
b = b.rename({'source1_entity_id': 's1_id', 'matched_entity_ids': 'm'})
bp = b.with_columns(pl.col('m').str.split(',')).explode('m').filter(pl.col('m').is_not_null() & (pl.col('m') != '')).select('s1_id', pl.col('m').alias('t_id'))
assert bp['t_id'].is_duplicated().sum() == 0
dec = pl.read_parquet(f'{T}/dec.parquet').with_columns(pl.col('m').str.split(',')).explode('m').filter(pl.col('m') != '').select('s1_id', pl.col('m').alias('t_id'))
new = pl.read_parquet(f'{T}/new.parquet', columns=K).with_columns(isnew=pl.lit(True))
adds = dec.join(bp, on=K, how='anti').join(cty, on='s1_id').join(new, on=K, how='left').with_columns(pl.col('isnew').fill_null(False))
owned = bp.select('t_id').with_columns(owned=pl.lit(True))
adds = adds.join(owned, on='t_id', how='left').with_columns(pl.col('owned').fill_null(False))
print('SF8 adds', adds.height, 'by country/new/owned-in-base', adds.group_by('country', 'isnew', 'owned').len().sort('country', 'isnew', 'owned').rows())
keep = adds.filter(~pl.col('owned'))
print('SF8a adds', keep.height, keep.group_by('country', 'isnew').len().sort('country', 'isnew').rows(), 'excluded (target owned in base)', adds.height - keep.height)
fin = pl.concat([bp, keep.select(K)]); assert fin['t_id'].is_duplicated().sum() == 0, 'target with two owners'
ka = keep.group_by('s1_id').agg(pl.col('t_id')); kad = dict(zip(ka['s1_id'], ka['t_id']))
nw = pl.read_parquet(f'{T}/new.parquet', columns=['s1_id', 't_id']).group_by('s1_id').agg(pl.col('t_id')); nc = dict(zip(nw['s1_id'], nw['t_id']))
rows = []
with open(f'{BASE}/matching_results.tsv', 'rb') as fi, open(f'{OUT}/matching_results.tsv', 'wb') as fo:
    fo.write(fi.readline())
    for line in fi:
        s1, m = line.decode().rstrip('\r\n').split('\t')
        if s1 in kad:
            m2 = ','.join(sorted(set([t for t in m.split(',') if t]) | set(kad[s1]))); fo.write(f'{s1}\t{m2}\n'.encode()); rows.append((s1, m, m2))
        else: fo.write(line)
assert len(rows) == len(kad)
with open('submissions/v12fr12x2_E2_UB10ui_SF8_s/candidate_pairs.tsv', 'rb') as fi, open(f'{OUT}/candidate_pairs.tsv', 'wb') as fo: fo.write(fi.read())
ch = pl.DataFrame(rows, schema=['s1_id', 'old', 'new'], orient='row').join(cty, on='s1_id'); ch.write_parquet((_BER_S + '/sfloor/SF8a_changes_ui.parquet'))
print('refs changed', ch.height, ch.group_by('country').len().sort('country').rows())
# containment: every added pair in candidate file (new pairs are in SF8 extended cands; old pairs in base cands)
for f in ('matching_results.tsv', 'candidate_pairs.tsv'): print(f, hashlib.md5(open(f'{OUT}/{f}', 'rb').read()).hexdigest())
# ---- SF8 removals by cause (diagnostics only)
if os.environ.get('SF_DIAG', '0') != '1': raise SystemExit(0)  # packaged copy: diagnostics opt-in with SF_DIAG=1
rem = bp.join(dec, on=K, how='anti').join(cty, on='s1_id').filter(pl.col('country').is_in(['US', 'India']))
s2 = pl.read_parquet(f'{T}/s2_final.parquet', columns=K + ['p']).rename({'p': 'p_s2'})
ubd = pl.read_parquet(f'{T}/dec.parquet.upper_ledger.parquet').filter(pl.col('x2').is_not_null() & (pl.col('x2') < 0.10)).select(K).with_columns(ub=pl.lit(True))
newown = dec.rename({'s1_id': 'owner_new'})
r = rem.join(newown, on='t_id', how='left').join(ubd, on=K, how='left').join(s2, on=K, how='left')
r = r.with_columns(cause=pl.when(pl.col('owner_new').is_not_null() & (pl.col('owner_new') != pl.col('s1_id'))).then(pl.lit('transfer'))
                   .when(pl.col('ub').fill_null(False)).then(pl.lit('UB10')).otherwise(pl.lit('alpha/threshold')))
print('SF8 removals', r.height, 'by cause x country', r.group_by('cause', 'country').len().sort('cause', 'country').rows())
tr = r.filter(pl.col('cause') == 'transfer').join(s2.rename({'s1_id': 'owner_new', 'p_s2': 'p_s2_new'}), on=['owner_new', 't_id'], how='left')
tr = tr.join(new.rename({'s1_id': 'owner_new', 'isnew': 'claimant_is_new_pair'}), on=['owner_new', 't_id'], how='left')
q = lambda c: [round(float(tr[c].quantile(x)), 4) for x in (0.25, 0.5, 0.75)]
print('transfers: old accepted pair p_s2 (now) q25/med/q75', q('p_s2'), '; new claimant p_s2', q('p_s2_new'), '; claimant is a new candidate pair:', int(tr['claimant_is_new_pair'].fill_null(False).sum()), 'of', tr.height)
p0 = pl.scan_parquet('output/test_scores_v12fr12x2_E2_s3avg_s2.parquet').select(K + [pl.col('p').alias('p_base')]).join(tr.lazy().select(K), on=K, how='semi').collect()
tr = tr.join(p0, on=K, how='left'); print('transfers: old pair p_s2 in BASE q25/med/q75', q('p_base'))
al = r.filter(pl.col('cause') == 'alpha/threshold').join(p0 if False else pl.scan_parquet('output/test_scores_v12fr12x2_E2_s3avg_s2.parquet').select(K + [pl.col('p').alias('p_base')]).join(r.lazy().select(K), on=K, how='semi').collect(), on=K, how='left')
print('alpha/threshold removals: p_s2 base -> now', al.select('p_base', 'p_s2').describe().filter(pl.col('statistic').is_in(['count', '25%', '50%', '75%'])).rows())
r.write_parquet((_BER_S + '/sfloor/SF8_removals_causes.parquet')); tr.write_parquet((_BER_S + '/sfloor/SF8_transfers.parquet'))
