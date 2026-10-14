"""sfloor TEST stage-2 inputs for R8 (US/India). usage: sf_test_prep.py step
 new  : concat $BER_SCRATCH/sfloor/test/r8/adm_c*.parquet (p1 > 0.01) -> T/new.parquet; CE need (p1 >= 0.02) -> T/ce_need.parquet
 build: scores (production v12avg US/India + new), s1f (v10w S1F of band rows + new), E2 (exact recompute on touched targets with parity), CE
HYBRID as on val: existing rows keep production features / stage-1 p / CE."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, os, json, glob, time, polars as pl, pyarrow.parquet as pq
sys.path.insert(0, (_BER_R + '/research/scripts_43')); os.chdir(_BER_R)
from comp43 import target_aggs, comp_features
T = (_BER_S + '/sfloor/test/r8s'); os.makedirs(T, exist_ok=True); K = ['s1_id', 't_id']; UI = ['US', 'India']; t0 = time.time()
S1F = json.load(open('models/stage2_v12ce2_E2.json'))['s1_feats']; FE = ['c_best', 'c_margin', 'c_n50', 'c_rank']
lg = lambda *a: print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
step = sys.argv[1]
if step == 'oldsf':
    band = pl.scan_parquet('output/test_scores_v12avg.parquet').filter(pl.col('country').is_in(UI) & (pl.col('p') > 0.01)).select(K).collect(); sf = []
    for rb in pq.ParquetFile('data/pairs_test_v10w_feat.parquet').iter_batches(batch_size=4_000_000, columns=K + S1F):
        sf.append(pl.from_arrow(rb).join(band, on=K, how='semi').with_columns([pl.col(c).cast(pl.Float32) for c in S1F]))
    sf = pl.concat(sf); assert sf.height == band.height; sf.write_parquet(f'{T}/s1f_old.parquet'); lg('old s1f', sf.height); sys.exit()
if step == 'new':
    D8 = (_BER_S + '/sfloor/test/r8'); SHA = '8703b37501d4ad54d47b27e5a49f50448b0df9a59b7c9f1e65da686536d7ce4e'  # packaged copy: sha256 of the packaged research/scripts_92/sft_r8_build.py
    prog = {}
    for ln in open(f'{D8}/progress.txt'):
        w = ln.split()
        if w and w[0] == 'chunk': prog[int(w[1])] = dict(zip(w[2::2], w[3::2]))
    assert sorted(prog) == list(range(16)), sorted(prog)
    assert all(v['exit'] == '0' and v['builder_sha256'] == SHA and v['rows_in'] == v['rows_out'] for v in prog.values())
    fz = pl.read_parquet((_BER_S + '/sfloor/test_adds_full.parquet')).filter(pl.col('country').is_in(UI)).select(K)
    allk = pl.concat([pl.read_parquet(f'{D8}/s1_c{k}.parquet', columns=K + ['p']) for k in range(16)])
    for k in range(16): assert pl.scan_parquet(f'{D8}/s1_c{k}.parquet').select(pl.len()).collect().item() == int(prog[k]['rows_out'])
    assert allk.select(K).is_duplicated().sum() == 0 and allk.height == fz.height and allk.join(fz, on=K, how='anti').height == 0, 'union != frozen test proposal'
    n = pl.concat([pl.read_parquet(f'{D8}/adm_c{k}.parquet') for k in range(16)]).filter(pl.col('country').is_in(UI) & (pl.col('p') > 0.01))
    exp = allk.filter(pl.col('p') > 0.01); assert n.height == exp.height and n.join(exp, on=K, how='anti').height == 0, 'admitted != p1>0.01 subset'
    lg(f'MANIFEST OK: 16 chunks exit 0, builder {SHA[:12]}, union {allk.height:,} == frozen test proposal (US/India o<=8), admitted {n.height:,} == p1>0.01 subset')
    assert n.select(K).is_duplicated().sum() == 0
    assert pl.scan_parquet('output/test_scores_v12avg.parquet').select(K).join(n.lazy().select(K), on=K, how='semi').select(pl.len()).collect().item() == 0
    n.write_parquet(f'{T}/new.parquet'); n.filter(pl.col('p') >= 0.02).select(K).write_parquet(f'{T}/ce_need.parquet')
    lg('admitted new', n.height, n.group_by('country', pl.col('p').cut([0.02, 0.5, 0.98]).alias('b')).len().sort('country', 'b').rows()); sys.exit()
new = pl.read_parquet(f'{T}/new.parquet')
old = pl.scan_parquet('output/test_scores_v12avg.parquet').filter(pl.col('country').is_in(UI)).collect()
sc = pl.concat([old, new.select(old.columns).with_columns(pl.col('p').cast(pl.Float32))]); sc.write_parquet(f'{T}/scores.parquet'); lg('scores', sc.height)
band = sc.filter(pl.col('p') > 0.01).select(K)
sf = pl.concat([new.select(K + [pl.col(c).cast(pl.Float32) for c in S1F]), pl.read_parquet(f'{T}/s1f_old.parquet')]); assert sf.height == band.height and sf.select(K).is_duplicated().sum() == 0; sf.write_parquet(f'{T}/s1f.parquet'); lg('s1f', sf.height)
# E2: exact comp43 recompute on touched targets (production universe = all test stage-1 rows)
TT = new.select('t_id').unique()
u0 = pl.scan_parquet('output/test_scores_v12avg.parquet').select(K + ['p']).join(TT.lazy(), on='t_id', how='semi').collect()
ref = pl.read_parquet('data/e2_feats_test_v12.parquet')
def e2(u, name):
    f = f'{T}/u_{name}.parquet'; u.select('t_id', 'p').write_parquet(f); g, hi = target_aggs(f, log=lg)
    c = comp_features(u.filter(pl.col('p') > 0.01).select(K + ['p']), g, hi); os.remove(f)
    return c.select(K + [pl.col(x).cast(pl.Float32) for x in FE])
c0 = e2(u0, 'ctrl'); j = c0.join(ref, on=K, how='left', suffix='_r')
bad = sum(int((((j[f] - j[f + '_r']).abs() > 0) | j[f + '_r'].is_null()).fill_null(True).sum()) for f in FE)
lg(f'E2 PARITY on touched targets: {c0.height:,} rows, mismatches vs production {bad}'); assert bad == 0
u1 = pl.concat([u0, new.select(K + [pl.col('p').cast(pl.Float32)])]); c1 = e2(u1, 'treat')
e2f = pl.concat([ref.join(TT, on='t_id', how='anti'), c1.select(ref.columns)]); assert e2f.select(K).is_duplicated().sum() == 0
e2f.write_parquet(f'{T}/e2.parquet'); lg('E2 rows', e2f.height, 'recomputed', c1.height, 'old rows changed', c1.join(ref, on=K).filter(pl.any_horizontal([pl.col(f) != pl.col(f + '_right') for f in FE])).height)
# CE: production X2 test file + X2 of new rows with 0.02 <= p <= 0.98
ce = pl.read_parquet((_BER_S + '/cntall/v12fr12x2_E2_ce2_test.parquet'))
nce = pl.read_parquet(f'{T}/newce.parquet'); nb = new.filter(pl.col('p').is_between(0.02, 0.98)).select(K)
assert nb.join(nce, on=K, how='anti').height == 0, 'new band rows lack X2'
ce = pl.concat([ce, nce.join(nb, on=K, how='semi').select(K + [pl.col('x2').cast(pl.Float32).alias('p_ce')])]); assert ce.select(K).is_duplicated().sum() == 0
ce.write_parquet(f'{T}/ce.parquet'); lg('CE rows', ce.height, 'new', nb.height)
