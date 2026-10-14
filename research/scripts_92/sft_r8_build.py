"""sfloor R8 TEST step 1 per chunk (chunk = s1_id.hash(seed=92) % 16, frozen, label-free). usage: sft_r8_build.py K
Emits only the chunk's own new rows (US/India, o<=8). Per-S1 context: old production rows of the chunk's S1s + the chunk's new rows
(complete: all new rows of an S1 are in its chunk). Per-target context: old production rows on the touched targets T + ALL new pairs
(any chunk, full US/India o<=8 set, base cols from r8/new_base_all.parquet) whose t_id is in T -- no truncation.
Chunk 0 also runs (a) route-OFF context parity on untouched production S1s/targets and (b) a synthetic check: context of a subset of
chunk-0 and chunk-1 references computed as two separate chunks == computed as one merged chunk (0 mismatches required)."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, time, os, polars as pl
sys.path.insert(0, (_BER_R + '/research/scripts_rmiss'))
from t_ctx import s1_context, t_context, S1C, TC, STD, KEYS
os.chdir(_BER_R); T0 = time.time(); O = (_BER_S + '/sfloor/test/r8'); K8 = int(sys.argv[1])
def lg(*a): print(f'[{time.time()-T0:.0f}s]', *a, flush=True)
P = pl.scan_parquet('data/pairs_test_v7.parquet'); schema = P.collect_schema(); cols = list(schema.names())
BASEC = KEYS + [f'{v}_score' for v in STD] + ['max_score']
ALL = pl.scan_parquet(f'{O}/new_base_all.parquet')
def cmp(x, ref, cs, nm):
    j = x.select(KEYS + cs).join(ref.select(KEYS + cs), on=KEYS, suffix='_r'); bad = 0
    assert j.height == x.height == ref.height, (j.height, x.height, ref.height)
    for c in cs:
        d = int((((j[c].cast(pl.Float64) - j[c + '_r'].cast(pl.Float64)).abs() > 0) | (j[c].is_null() != j[c + '_r'].is_null())).fill_null(False).sum())
        if d: print(f'  MISMATCH {nm} {c}: {d}'); bad += d
    lg(f'{nm}: {j.height:,} rows, mismatches {bad}'); assert bad == 0
def ctx(own, allnew_c, cty):
    """own: base rows (BASEC) of the new pairs to emit; allnew_c: base rows of ALL new pairs of the country (lazy)."""
    R1 = own.select('s1_id').unique(); T1 = own.select('t_id').unique()
    old = P.filter((pl.col('country') == cty) & (pl.col('s1_id').is_in(R1['s1_id'].implode()) | pl.col('t_id').is_in(T1['t_id'].implode()))).select(BASEC).collect(engine='streaming')
    nt = allnew_c.join(T1.lazy(), on='t_id', how='semi').select(BASEC).collect()
    a = pl.concat([old.filter(pl.col('s1_id').is_in(R1['s1_id'].implode())).with_columns(_own=pl.lit(False)), own.select(BASEC).with_columns(_own=pl.lit(True))])
    a = s1_context(a, STD).filter(pl.col('_own')).select(KEYS + S1C(STD))
    b = pl.concat([old.filter(pl.col('t_id').is_in(T1['t_id'].implode())).with_columns(_own=pl.lit(False)),
                   nt.join(own.select(KEYS), on=KEYS, how='anti').with_columns(_own=pl.lit(False)), own.select(BASEC).with_columns(_own=pl.lit(True))])
    assert b.select(KEYS).is_duplicated().sum() == 0
    b = t_context(b, STD).filter(pl.col('_own')).select(KEYS + TC(STD))
    return a.join(b, on=KEYS), old.height, nt.height
outs = []
for cty in ['India', 'US']:
    allc = ALL.filter(pl.col('country') == cty)
    nb = allc.filter(pl.col('chunk') == K8).collect()
    if K8 == 0:
        R1 = nb.select('s1_id').unique(); T1 = nb.select('t_id').unique()
        sS = P.filter(pl.col('country') == cty).select('s1_id').head(3_000_000).unique().join(R1.lazy(), on='s1_id', how='anti').head(20000).collect()
        ps = P.filter(pl.col('s1_id').is_in(sS['s1_id'].implode())).select(BASEC + S1C(STD)).collect(engine='streaming')
        cmp(s1_context(ps.drop(S1C(STD)), STD), ps, S1C(STD), f'route-OFF parity {cty} per-S1'); del ps
        sT = P.filter(pl.col('country') == cty).select('t_id').head(3_000_000).unique().join(T1.lazy(), on='t_id', how='anti').head(20000).collect()
        pt = P.filter(pl.col('t_id').is_in(sT['t_id'].implode())).select(BASEC + TC(STD)).collect(engine='streaming')
        cmp(t_context(pt.drop(TC(STD)), STD), pt, TC(STD), f'route-OFF parity {cty} per-target'); del pt
        # synthetic: 3,000 refs of chunk 0 + 3,000 refs of chunk 1, emitted as 2 chunks vs as 1 merged chunk
        r0 = nb.select('s1_id').unique().sort('s1_id').head(3000)
        r1 = allc.filter(pl.col('chunk') == 1).select('s1_id').unique().sort('s1_id').head(3000).collect()
        o0 = allc.join(r0.lazy(), on='s1_id', how='semi').collect(); o1 = allc.join(r1.lazy(), on='s1_id', how='semi').collect()
        two = pl.concat([ctx(o0, allc, cty)[0], ctx(o1, allc, cty)[0]]); one = ctx(pl.concat([o0, o1]), allc, cty)[0]
        cmp(two, one, S1C(STD) + TC(STD), f'SYNTHETIC {cty} 2-chunk vs 1-chunk ({o0.height:,}+{o1.height:,} new rows)')
    c, nold, nt = ctx(nb, allc, cty)
    o = nb.join(c, on=KEYS).with_columns(split=pl.lit('test'), t_source=pl.col('t_id').str.slice(1, 1).cast(pl.Int8),
                                        sibling_conflict=pl.lit(None, pl.Int8), is_extra=pl.lit(0, pl.Int8))
    o = o.select([pl.col(x).cast(schema[x]) for x in cols]); assert o.height == nb.height and o.select(KEYS).is_duplicated().sum() == 0
    for x in S1C(STD) + TC(STD) + ['n_views', 'best_rank']: assert o[x].null_count() == 0, x
    lg(f'{cty} chunk {K8}: new rows {o.height:,}; old rows {nold:,}; all-chunk new rows on T {nt:,}; t_n_owners mean {o["t_n_owners"].mean():.2f}')
    outs.append(o)
pl.concat(outs).write_parquet(f'{O}/new_pairs_r8_c{K8}.parquet'); lg('wrote', sum(x.height for x in outs))
