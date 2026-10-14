"""sfloor R8 TEST stage-1 per chunk (same arithmetic as sft_s1.py). Parity on 200k existing rows only for chunk 0.
Writes r8/s1_c{k}.parquet (all rows of the chunk: s1_id,t_id,country,p + S1F Float32) and r8/adm_c{k}.parquet (p > 0.01)."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import json, os, sys, time, numpy as np, polars as pl, pyarrow.parquet as pq, lightgbm as lgb
os.chdir(_BER_R); R = (_BER_S + '/sfloor/test/r8'); T0 = time.time(); k = int(sys.argv[1]); KEYS = ['s1_id', 't_id']
def lg(*a): print(f'[{time.time()-T0:.0f}s]', *a, flush=True)
M = {}
for t in ['v12', 'v12b']:
    cfg = json.load(open(f'models/lgbm_{t}.json')); M[t] = (lgb.Booster(model_file=f'models/lgbm_{t}.txt'), cfg['features'], {c: i for i, c in enumerate(cfg['countries'])}, cfg.get('best_iteration'))
S1F = json.load(open('models/stage2_v12ce2_E2.json'))['s1_feats']
def score(d):
    ps = []
    for t in ['v12', 'v12b']:
        bst, fc, cm, bi = M[t]
        X = np.hstack([d.select(fc).to_numpy().astype(np.float32), d['country'].replace_strict(cm, default=-1).to_numpy().astype(np.float32).reshape(-1, 1)])
        ps.append(bst.predict(X, num_iteration=bi, num_threads=6).astype(np.float32)); del X
    return (0.5 * pl.Series(ps[0]).cast(pl.Float64) + 0.5 * pl.Series(ps[1]).cast(pl.Float64)).cast(pl.Float32)
if k == 0:
    pf = pq.ParquetFile('data/pairs_test_v10w_feat.parquet'); ng = pf.num_row_groups
    u = pl.concat([pl.from_arrow(pf.read_row_group(g)).head(100_000) for g in (0, ng // 2)])
    pu = u.select(KEYS).with_columns(p=score(u)).join(pl.scan_parquet('output/test_scores_v12avg.parquet').join(u.lazy().select(KEYS), on=KEYS, how='semi').collect(), on=KEYS, suffix='_prod')
    bad = int((pu['p'] != pu['p_prod']).sum()); lg(f'PARITY stage-1: {pu.height:,} rows, mismatches {bad}'); assert bad == 0 and pu.height == 200_000; del u, pu
new = pl.read_parquet(f'{R}/new_feat_r8_c{k}.parquet')
o = new.select(KEYS + ['country']).with_columns(p=score(new)).hstack(new.select([pl.col(x).cast(pl.Float32) for x in S1F])); del new
o.write_parquet(f'{R}/s1_c{k}.parquet'); a = o.filter(pl.col('p') > 0.01); a.write_parquet(f'{R}/adm_c{k}.parquet')
lg(f'chunk {k}: rows {o.height} admitted {a.height}')
