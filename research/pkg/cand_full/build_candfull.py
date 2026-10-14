"""cand_full: alternative candidate file = base candidates (603b3cc1...) UNION all 25,925,609 US/India source-floor proposals.
Merge rule replicates research/scripts_92/sf_build_entry.py: for a touched row write sorted(set(base ids) | set(proposal ids)) (Python str sort),
joined by ',', line '\n'; every untouched row is copied byte for byte. Also copies the final matching_results.tsv (d5d44ade...).
Step 1 checks (proposal count, duplicates, admitted subset, s1_c* union parity, overlap with base) are done here too."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, hashlib, shutil, time, polars as pl
os.chdir(_BER_R); t0 = time.time(); K = ['s1_id', 't_id']
lg = lambda *a: print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
BASE = 'submissions/v12fr12x2_E2_UB10ui_s/candidate_pairs.tsv'; SUB = 'submissions/v12fr12x2_E2_UB10ui_SF8a_s'
OUT = 'submissions/v12fr12x2_E2_UB10ui_SF8a_candfull'; PROP = (_BER_S + '/sfloor/test_adds_full.parquet')
D8 = (_BER_S + '/sfloor/test/r8'); NEW = (_BER_S + '/sfloor/test/r8s/new.parquet')


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


assert not os.path.exists(OUT), f'{OUT} exists'
m_base, m_sub, m_mr = md5(BASE), md5(f'{SUB}/candidate_pairs.tsv'), md5(f'{SUB}/matching_results.tsv')
lg('md5 base cands', m_base, '| submitted cands', m_sub, '| submitted matching', m_mr)
assert (os.environ.get('BER_STRICT_MD5', '0') != '1') or (m_base.startswith('603b3cc1') and m_sub == '29a07b6f8923caa277793a9543a130ed' and m_mr == 'd5d44ade281e7213af54f65cb968be2d')  # packaged copy
lg('md5 proposal parquet', md5(PROP), '| new.parquet', md5(NEW))

# ---- step 1: proposals
pr = pl.read_parquet(PROP, columns=K + ['country'])
lg('test_adds_full rows by country', pr.group_by('country').len().sort('country').rows())
pr = pr.filter(pl.col('country').is_in(['US', 'India']))
lg('US/India proposals', pr.height, '; duplicate keys', pr.select(K).is_duplicated().sum())
assert (pr.height == 25_925_609 or (os.environ.get('BER_STRICT_MD5', '0') != '1')) and pr.select(K).is_duplicated().sum() == 0
assert pr.filter(~pl.col('t_id').str.starts_with('S2-') & ~pl.col('t_id').str.starts_with('S3-')).height == 0
assert pr.filter(~pl.col('s1_id').str.starts_with('S1-')).height == 0
# scored universe (union of chunk stage-1 files) == proposal
allk = pl.concat([pl.read_parquet(f'{D8}/s1_c{k}.parquet', columns=K) for k in range(16)])
lg('s1_c* union rows', allk.height, '; dup', allk.is_duplicated().sum(), '; anti vs proposal', allk.join(pr, on=K, how='anti').height,
   '; proposal anti vs s1_c*', pr.join(allk, on=K, how='anti').height)
assert allk.height == pr.height and allk.join(pr, on=K, how='anti').height == 0
del allk
new = pl.read_parquet(NEW, columns=K + ['p'])
lg('admitted (new.parquet)', new.height, 'min p', new['p'].min(), '; not in proposal', new.join(pr, on=K, how='anti').height)
assert (new.height == 85_332 or (os.environ.get('BER_STRICT_MD5', '0') != '1')) and new.join(pr, on=K, how='anti').height == 0
del new
pr = pr.select(K).group_by('s1_id').agg(pl.col('t_id'))
props = dict(zip(pr['s1_id'].to_list(), pr['t_id'].to_list())); nprop_refs = len(props)
lg('refs with proposals', nprop_refs); del pr

# ---- step 2: merge
os.makedirs(OUT)
seen, touched, overlap, n_rows, n_pairs_out = set(), 0, 0, 0, 0
with open(BASE, 'rb') as fi, open(f'{OUT}/candidate_pairs.tsv', 'wb') as fo:
    hdr = fi.readline(); assert hdr == b'source1_entity_id\tcandidate_entity_ids\n'; fo.write(hdr)
    for line in fi:
        n_rows += 1
        s1, ids = line.decode().rstrip('\r\n').split('\t')
        assert s1 not in seen; seen.add(s1)
        if s1 in props:
            b = set(t for t in ids.split(',') if t); p = set(props[s1])
            overlap += len(b & p); u = sorted(b | p); touched += 1; n_pairs_out += len(u)
            fo.write(f'{s1}\t{",".join(u)}\n'.encode())
        else:
            fo.write(line); n_pairs_out += len([t for t in ids.split(',') if t])
missing = set(props) - seen
lg('rows', n_rows, 'touched', touched, 'proposal refs not in base file', len(missing), 'proposal pairs already in base', overlap, 'pairs out', n_pairs_out)
assert touched == nprop_refs and not missing
shutil.copyfile(f'{SUB}/matching_results.tsv', f'{OUT}/matching_results.tsv')
m_new_mr = md5(f'{OUT}/matching_results.tsv'); assert m_new_mr == 'd5d44ade281e7213af54f65cb968be2d' or (os.environ.get('BER_STRICT_MD5', '0') != '1')
lg('matching_results copied, md5', m_new_mr, '| new candidate_pairs md5', md5(f'{OUT}/candidate_pairs.tsv'))
