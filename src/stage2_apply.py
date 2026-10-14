"""Apply a trained stage-2 model to stage-1 test scores.
Computes the stage-2 features (src/stage2_feats.py) country by country (every reference's full candidate list stays
together), predicts stage-2 p for pairs with stage-1 p > p_min and keeps stage-1 p elsewhere. Output schema matches
predict.py's scores file (s1_id, t_id, country, p) so `predict.py --scores-from <out>` writes the submission.
Run: nice -n 10 python src/stage2_apply.py --scores output/test_scores_v1.parquet --pairs data/pairs_test_v1_feat.parquet \
       --split test --model models/stage2_v1 --out output/test_scores_v1_s2.parquet [--limit-s1 20000]
"""
import argparse, json, os, sys, time, shutil
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
import stage2_feats as S

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True); ap.add_argument("--pairs", required=True, help="stage-1 features file of the same split")
ap.add_argument("--split", default="test"); ap.add_argument("--model", required=True, help="models/stage2_<tag> (no extension)")
ap.add_argument("--out", required=True); ap.add_argument("--limit-s1", type=int, default=0, help="random subset of references (smoke test)")
ap.add_argument("--views", default=""); ap.add_argument("--emb", default=""); ap.add_argument("--emb-ids", default="")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--ce", default="", help="parquet s1_id,t_id,p_ce for this split (required iff the model was trained with --ce)")
ap.add_argument("--ce2", default="", help="second CE parquet s1_id,t_id,p_ce for this split (required iff the model was trained with --ce2)")
ap.add_argument("--ce3", default="", help="third CE parquet s1_id,t_id,p_ce for this split (required iff the model was trained with --ce3)")
ap.add_argument("--allow-missing-ce3", action="store_true", help="do not fail when band rows lack a ce3 score (they get the -1 sentinel)")
ap.add_argument("--extra-feats", default="", help="parquet s1_id,t_id + the model JSON extra_feats columns for this split (required iff the model has extra_feats)")
ap.add_argument("--allow-missing-extra", action="store_true", help="do not fail when rescored rows (p > P_MIN) are absent from --extra-feats (they get NaN)")
ap.add_argument("--ce-template", default="", choices=["", *S.CE_TEMPLATES], help="optional; must equal the model JSON ce_template (read from the JSON; absent = default)")
a = ap.parse_args()
t00 = time.time(); log = lambda *x: print(*x, flush=True)
cfg = json.load(open(a.model + ".json")); model = lgb.Booster(model_file=a.model + ".txt")
FEATS, s1f, countries = cfg["features"], cfg["s1_feats"], cfg["countries"]
assert abs(cfg["p_min"] - S.P_MIN) < 1e-12
assert bool(cfg.get("ce", False)) == bool(a.ce), "model trained with CE features needs --ce (and vice versa)"
assert bool(cfg.get("ce2", False)) == bool(a.ce2), "model trained with CE2 features needs --ce2 (and vice versa)"
assert bool(cfg.get("ce3", False)) == bool(a.ce3), "model trained with CE3 features needs --ce3 (and vice versa)"
tpl = cfg.get("ce_template", "default"); assert tpl in S.CE_TEMPLATES, tpl
assert not a.ce_template or a.ce_template == tpl, f"--ce-template {a.ce_template} contradicts the model JSON ce_template {tpl}"
fd = bool(cfg.get("field_disagreement", False))
assert not fd or (a.ce2 and a.ce3 and tpl == "default"), "field_disagreement model needs --ce2 and --ce3 (default template)"
XF = list(cfg.get("extra_feats") or [])
assert bool(XF) == bool(a.extra_feats), f"model extra_feats {XF} needs --extra-feats (and vice versa)"
if XF:
    fx = S.extra_columns(a.extra_feats); assert fx == XF, f"--extra-feats columns {fx} != model JSON extra_feats {XF}"
    assert FEATS[len(FEATS) - len(XF):] == XF, f"model feature tail {FEATS[-len(XF):]} != extra_feats {XF}"
FEATS_NX = FEATS[:len(FEATS) - len(XF)]   # CE-tail check below runs on the features before the extra block
ce_tail = (S.ce_feats(1) if a.ce else []) + (S.ce_feats(2, tpl) if a.ce2 else []) + (S.ce_feats(3, tpl) if a.ce3 else []) + (S.FIELD if fd else [])
all_ce = set(S.CE + S.CE2 + S.CE3 + S.CE2_IND + S.CE3_IND + S.FIELD)
assert FEATS_NX[len(FEATS_NX) - len(ce_tail):] == ce_tail and not (set(FEATS_NX[:len(FEATS_NX) - len(ce_tail)]) & all_ce), \
    f"model feature order {FEATS_NX[-12:]} != expected CE tail {ce_tail} (template {tpl})"
assert model.feature_name() == FEATS, "booster feature names differ from the model JSON feature list"
views = a.views or f"data/views_{a.split}.parquet"
emb = S.EmbStore(a.emb or f"data/emb_{a.split}_multilingual-e5-small_full.npy", a.emb_ids or f"data/emb_{a.split}_multilingual-e5-small_full_ids.npy")
log(f"stage-2 model {a.model}: {len(FEATS)} features (ce_template {tpl}, field_disagreement {fd}, extra_feats {XF}); embeddings {emb.E.shape}")

lz = pl.scan_parquet(a.scores).select("s1_id", "t_id", "country", pl.col("p").cast(pl.Float32))
keep = None
if a.limit_s1:
    u = lz.select(pl.col("s1_id").unique()).collect()["s1_id"].sort()
    keep = u.sample(min(a.limit_s1, len(u)), seed=0); log(f"limit: {len(keep):,} of {len(u):,} references")
    lz = lz.filter(pl.col("s1_id").is_in(keep.implode()))
ctys = lz.select(pl.col("country").unique()).collect()["country"].sort().to_list()
tmp = a.out + ".parts"; os.makedirs(tmp, exist_ok=True); n_band = 0
for c in ctys:
    t0 = time.time()
    sc = lz.filter(pl.col("country") == c).collect(); log(f"[{c}] {len(sc):,} pairs, {sc['s1_id'].n_unique():,} references")
    b = S.build(sc, countries, views, emb, log)
    b = S.join_stage1(b, a.pairs, s1f)
    if a.ce: b = S.add_ce(b, a.ce); log(f"[{c}] band rows with p_ce: {(b['p_ce'] >= 0).sum():,}")
    if a.ce2: b = S.add_ce_family(b, a.ce2, 2, tpl); log(f"[{c}] band rows with p_ce2: {(b['p_ce2'] >= 0).sum():,}")
    if a.ce3:
        b = S.add_ce_family(b, a.ce3, 3, tpl); inb = pl.col("p").is_between(*S.CE_BAND)
        n_cb, n3 = b.filter(inb).height, b.filter(inb & (pl.col("p_ce3") < 0)).height
        log(f"[{c}] band rows with p_ce3: {(b['p_ce3'] >= 0).sum():,}; CE-band rows ({S.CE_BAND[0]}<=p<={S.CE_BAND[1]}) {n_cb:,}, lacking ce3: {n3:,}")
        assert n3 == 0 or a.allow_missing_ce3, f"[{c}] {n3} CE-band rows lack a ce3 score (pass --allow-missing-ce3 to accept the -1 sentinel)"
    if fd: b = S.add_field_disagreement(b)
    if XF:
        b = S.add_extra(b, a.extra_feats, XF)   # b = every rescored row (S.build keeps p > P_MIN)
        n_hit = int(b["_xf_hit"].sum()); nx = len(b) - n_hit
        log(f"[{c}] rescored rows (p > {S.P_MIN}) with extra feats: {n_hit:,} of {len(b):,} ({n_hit / max(len(b), 1):.4%}), lacking: {nx:,}")
        assert nx == 0 or a.allow_missing_extra, f"[{c}] {nx} rescored rows (p > {S.P_MIN}) lack extra features (pass --allow-missing-extra to accept NaN)"
    miss = b[s1f[0]].null_count(); assert miss == 0, f"{miss} band pairs missing from the stage-1 features file"
    p2 = model.predict(b.select([pl.col(f).cast(pl.Float32).fill_nan(None) for f in FEATS]).to_numpy(), num_threads=a.threads)
    upd = b.select("s1_id", "t_id").with_columns(pl.Series("p2", p2.astype(np.float32))); n_band += len(upd); del b
    out = sc.join(upd, on=["s1_id", "t_id"], how="left").with_columns(p=pl.coalesce("p2", "p")).drop("p2")
    out.write_parquet(f"{tmp}/{c}.parquet"); del sc, out
    log(f"[{c}] band {len(upd):,} rows re-scored (mean p1→p2 over band: {float(np.mean(p2)):.4f}) in {time.time()-t0:.0f}s")
pl.scan_parquet(f"{tmp}/*.parquet").sink_parquet(a.out); shutil.rmtree(tmp)
tot = pl.scan_parquet(a.out).select(pl.len(), pl.col("p").mean()).collect()
log(f"wrote {a.out}: {tot[0,0]:,} pairs ({n_band:,} stage-2), mean p {tot[0,1]:.4f}; total {time.time()-t00:.0f}s")
log(f"next: python src/predict.py --pairs {a.pairs} --model <stage-1 model> --scores-from {a.out} --t {cfg['t']} --alpha {cfg['alpha']} --tag <tag>_s2   (stage-2 policy ownership={cfg['ownership']})")
