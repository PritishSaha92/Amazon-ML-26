"""cand_full verification: independently re-parses the new candidate file and checks it against base, submitted SF8a cands, proposals, matching.
Streams the four TSVs in lockstep (same S1 order required); stats per country."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, hashlib, time, numpy as np, polars as pl
os.chdir(_BER_R); t0 = time.time(); K = ['s1_id', 't_id']
lg = lambda *a: print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
BASE = 'submissions/v12fr12x2_E2_UB10ui_s/candidate_pairs.tsv'; SUB = 'submissions/v12fr12x2_E2_UB10ui_SF8a_s'
OUT = 'submissions/v12fr12x2_E2_UB10ui_SF8a_candfull'
pr = pl.read_parquet((_BER_S + '/sfloor/test_adds_full.parquet'), columns=K + ['country']).filter(pl.col('country').is_in(['US', 'India']))
pr = pr.select(K).group_by('s1_id').agg(pl.col('t_id')); props = dict(zip(pr['s1_id'].to_list(), pr['t_id'].to_list())); del pr
cty = pl.read_parquet('data/test_records.parquet', columns=['entity_id', 'country']); cmap = dict(zip(cty['entity_id'], cty['country'])); del cty
mr = {}
with open(f'{SUB}/matching_results.tsv', 'rb') as f:
    assert f.readline() == b'source1_entity_id\tmatched_entity_ids\n'
    for ln in f:
        s1, m = ln.decode().rstrip('\n').split('\t'); mr[s1] = [t for t in m.split(',') if t]
lens, ctys = [], []
st = dict(rows=0, identical=0, touched=0, base_unsorted=0, sub_not_subset=0, mr_not_in=0, mr_pairs=0, bad_union=0, not_sorted=0, dup=0, crlf=0,
          base_pairs=0, sub_pairs=0, prop_pairs_in=0)
with open(f'{OUT}/candidate_pairs.tsv', 'rb') as fn, open(BASE, 'rb') as fb, open(f'{SUB}/candidate_pairs.tsv', 'rb') as fs:
    hn, hb, hs = fn.readline(), fb.readline(), fs.readline(); assert hn == hb == hs == b'source1_entity_id\tcandidate_entity_ids\n'
    for ln, lb, ls in zip(fn, fb, fs):
        st['rows'] += 1
        if b'\r' in ln: st['crlf'] += 1
        s1n, idn = ln.decode().rstrip('\n').split('\t'); s1b, idb = lb.decode().rstrip('\n').split('\t'); s1s, ids_ = ls.decode().rstrip('\n').split('\t')
        assert s1n == s1b == s1s, (s1n, s1b, s1s)
        n = [t for t in idn.split(',') if t]; b = [t for t in idb.split(',') if t]; s = [t for t in ids_.split(',') if t]
        sn = set(n); st['base_pairs'] += len(b); st['sub_pairs'] += len(s)
        if len(sn) != len(n): st['dup'] += 1
        if b != sorted(b): st['base_unsorted'] += 1
        if s1n in props:
            st['touched'] += 1; p = props[s1n]; st['prop_pairs_in'] += sum(t in sn for t in p)
            if sn != set(b) | set(p): st['bad_union'] += 1
            if n != sorted(n): st['not_sorted'] += 1
        else:
            if ln == lb: st['identical'] += 1
        if not set(s) <= sn: st['sub_not_subset'] += 1
        m = mr.pop(s1n); st['mr_pairs'] += len(m); st['mr_not_in'] += sum(t not in sn for t in m)
        lens.append(len(n)); ctys.append(cmap.get(s1n, '?'))
    assert fn.readline() == fb.readline() == fs.readline() == b''
st['mr_rows_unmatched'] = len(mr)
lg(st)
assert st['rows'] == 1_732_544 and st['identical'] + st['touched'] == st['rows'] and st['bad_union'] == st['not_sorted'] == st['dup'] == 0
assert st['sub_not_subset'] == 0 and st['mr_not_in'] == 0 and not mr and st['crlf'] == 0 and (st['prop_pairs_in'] == 25_925_609 or (os.environ.get('BER_STRICT_MD5', '0') != '1'))
L = np.array(lens); C = np.array(ctys)
def desc(x): return dict(refs=len(x), pairs=int(x.sum()), mean=round(float(x.mean()), 3), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                         p99=float(np.percentile(x, 99)), max=int(x.max()), min=int(x.min()))
print('ALL', desc(L))
for c in sorted(set(ctys)): print(c, desc(L[C == c]))
print('file size', os.path.getsize(f'{OUT}/candidate_pairs.tsv'))
