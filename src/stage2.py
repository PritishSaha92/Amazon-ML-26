"""Stage 2 (collective re-scoring) — spec research/08_stage2_collective.md.
Input: data/oof_scores_<tag>.parquet from oof_scores.py (train rows = out-of-fold stage-1 p; val/audit = full-model p).
Builds stage-2 features (src/stage2_feats.py) for every pair with stage-1 p > 0.01, trains LightGBM on train-split rows,
predicts val/audit, then runs the SAME decision sweep as train_matcher.py (best-reference ownership, alpha, t) for the
stage-1 baseline and for stage 2 (pairs with p <= 0.01 keep stage-1 p), and reports val/audit slices.
Saves models/stage2_<tag>.txt/.json and the band feature cache data/s2_band_<tag>.parquet.
Run: nice -n 10 python src/stage2.py --tag v1 --pairs data/pairs_train_v1_feat.parquet
"""
import argparse, json, os, sys, time
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from scorer import slice_report, print_report, per_entity_scores
import stage2_feats as S

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--pairs", required=True, help="stage-1 features file (pass-through features)")
ap.add_argument("--scores", default="", help="default data/oof_scores_<tag>.parquet")
ap.add_argument("--model", default="", help="stage-1 model prefix (countries); default models/lgbm_<tag>")
ap.add_argument("--views", default="data/views_train.parquet")
ap.add_argument("--emb", default="data/emb_train_multilingual-e5-small_full.npy")
ap.add_argument("--emb-ids", default="data/emb_train_multilingual-e5-small_full_ids.npy")
ap.add_argument("--s1-feats", default=",".join(S.S1F)); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--ce", default="", help="optional parquet s1_id,t_id,p_ce (OOF for train rows): adds stage2_feats.CE features")
ap.add_argument("--ce2", default="", help="optional second CE parquet s1_id,t_id,p_ce (OOF for train rows): adds stage2_feats.CE2 features")
ap.add_argument("--ce3", default="", help="optional third CE parquet s1_id,t_id,p_ce (OOF for train rows): adds stage2_feats.CE3 features after CE2")
ap.add_argument("--ce-template", default="default", choices=list(S.CE_TEMPLATES), help="how the --ce2/--ce3 blocks are built: default = CE2/CE3 (4th = p_ceK - p_ce); independent = add_ce template per family (4th = ceK_logit - logit)")
ap.add_argument("--field-disagreement", action="store_true", help="append stage2_feats.FIELD (field_signed/field_abs = p_ce3 - p_ce2) after the CE3 block; needs --ce2 (address) + --ce3 (name), default template")
ap.add_argument("--extra-feats", default="", help="optional parquet s1_id,t_id + numeric feature columns (train rows must be producer-consistent/OOF): left-joined on the band, missing -> NaN, appended after the CE/field blocks in the file's column order")
ap.add_argument("--allow-missing-extra", action="store_true", help="do not fail when rescored band rows (p > P_MIN) are absent from --extra-feats (they get NaN)")
ap.add_argument("--band-cache", default="", help="band feature cache to read/write (default data/s2_band_<tag>.parquet)")
ap.add_argument("--rebuild", action="store_true", help="ignore the band cache"); ap.add_argument("--max-rounds", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0, help="LightGBM seed (bagging/feature-fraction); default 0 = the deployed models. The train-holdout split stays fixed.")
a = ap.parse_args()
t00 = time.time(); log = lambda *x: print(*x, flush=True)
scores_path = a.scores or f"data/oof_scores_{a.tag}.parquet"; s1cfg = json.load(open((a.model or f"models/lgbm_{a.tag}") + ".json"))
countries = s1cfg["countries"]; s1f = [c for c in a.s1_feats.split(",") if c]
cache = a.band_cache or f"data/s2_band_{a.tag}.parquet"

sc = pl.read_parquet(scores_path, columns=["s1_id", "t_id", "country", "split", "label", "p"])
if os.path.exists(cache) and not a.rebuild:
    band = pl.read_parquet(cache); log(f"loaded band cache {cache}: {len(band):,}")
else:
    band = S.build(sc, countries, a.views, S.EmbStore(a.emb, a.emb_ids), log)
    band = S.join_stage1(band, a.pairs, s1f); band.write_parquet(cache); log(f"band built + cached ({time.time()-t00:.0f}s)")
FEATS = S.SET + S.CONS + S.EMB + s1f
if a.ce:
    band = S.add_ce(band, a.ce); FEATS = FEATS + S.CE
    log("CE features from", a.ce, band.group_by("split").agg(pl.len(), (pl.col("p_ce") >= 0).sum().alias("with_p_ce")).sort("split"))
if a.ce2:
    band = S.add_ce_family(band, a.ce2, 2, a.ce_template); FEATS = FEATS + S.ce_feats(2, a.ce_template)
    log("CE2 features from", a.ce2, band.group_by("split").agg(pl.len(), (pl.col("p_ce2") >= 0).sum().alias("with_p_ce2")).sort("split"))
if a.ce3:
    band = S.add_ce_family(band, a.ce3, 3, a.ce_template); FEATS = FEATS + S.ce_feats(3, a.ce_template)
    log("CE3 features from", a.ce3, band.group_by("split").agg(pl.len(), (pl.col("p_ce3") >= 0).sum().alias("with_p_ce3")).sort("split"))
if a.field_disagreement:
    assert a.ce2 and a.ce3 and a.ce_template == "default", "--field-disagreement needs --ce2 and --ce3 under --ce-template default"
    band = S.add_field_disagreement(band); FEATS = FEATS + S.FIELD
    log("field-disagreement features", band.group_by("split").agg(pl.len(), pl.col("field_signed").is_not_null().sum().alias("with_field")).sort("split"))
XF = []
if a.extra_feats:
    XF = S.extra_columns(a.extra_feats); clash = set(XF) & set(FEATS); assert not clash, f"extra feature names clash with FEATS: {sorted(clash)}"
    band = S.add_extra(band, a.extra_feats, XF); FEATS = FEATS + XF
    log(f"extra features {XF} from", a.extra_feats, band.group_by("split").agg(pl.len(), pl.col("_xf_hit").sum().alias("with_extra"),
        pl.col("_xf_hit").mean().alias("hit_rate"), (~pl.col("_xf_hit")).sum().alias("missing_extra")).sort("split"))
    nx = int((~band["_xf_hit"]).sum())   # band = every rescored row (p > P_MIN), train/val/audit
    assert nx == 0 or a.allow_missing_extra, f"{nx} rescored band rows (p > {S.P_MIN}) lack extra features (pass --allow-missing-extra to accept NaN)"
log(band.group_by("split").agg(pl.len(), pl.col("label").sum().alias("pos"), pl.col("s1_id").n_unique().alias("refs")).sort("split"))

def X_of(d): return d.select([pl.col(c).cast(pl.Float32).fill_nan(None) for c in FEATS]).to_numpy()
params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, metric="binary_logloss", seed=a.seed)
tr = band.filter(pl.col("split") == "train")
Xtr, ytr = X_of(tr), tr["label"].to_numpy()
# rounds: early stopping on a 10% holdout of train references, then refit on all train references with that many rounds
u = tr["s1_id"].unique().sort(); rng = np.random.default_rng(0); ho = set(u.to_numpy()[rng.permutation(len(u))[: len(u) // 10]])
m_ho = tr["s1_id"].is_in(list(ho)).to_numpy()
dk = dict(feature_name=FEATS, categorical_feature=["country_cat"])
m0 = lgb.train(params, lgb.Dataset(Xtr[~m_ho], ytr[~m_ho], **dk), a.max_rounds, valid_sets=[lgb.Dataset(Xtr[m_ho], ytr[m_ho], **dk)],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
rounds = int(m0.best_iteration * 1.1); log(f"holdout best_iteration {m0.best_iteration} → refit {rounds} rounds ({time.time()-t00:.0f}s)")
model = lgb.train(params, lgb.Dataset(Xtr, ytr, **dk), rounds); del Xtr
imp = sorted(zip(FEATS, model.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(g for _, g in imp)
log("top stage-2 features by gain share:", [(n, round(g / tot, 4)) for n, g in imp[:20]])

# ---- decisions: identical policy/sweep to train_matcher.py ----
truth_all = pl.read_parquet("data/train_gt.parquet"); split = pl.read_parquet("data/split.parquet")
cty = dict(zip(split["s1_id"], split["country"]))

def decide(d: pl.DataFrame, p: np.ndarray, t: float, alpha: float, ownership: bool, margin: float = 0.0) -> dict:
    x = d.select("s1_id", "t_id").with_columns(pl.Series("p", p))
    if ownership:  # each target keeps only its best S1 (abstain if the runner-up is within `margin`)
        x = x.sort(["t_id", "p"], descending=[False, True]).with_columns(
            pl.col("p").shift(-1).over("t_id").fill_null(0.0).alias("p2"), pl.int_range(pl.len()).over("t_id").alias("r"))
        x = x.filter((pl.col("r") == 0) & ((pl.col("p") - pl.col("p2")) >= margin))
    x = x.with_columns(pl.col("p").max().over("s1_id").alias("pmax"))
    x = x.filter((pl.col("p") >= t) & (pl.col("p") >= alpha * pl.col("pmax")))
    g = x.group_by("s1_id").agg(pl.col("t_id"))
    return dict(zip(g["s1_id"], (set(v) for v in g["t_id"])))

def truth_for(split_name, present: pl.Series):
    ids = split.filter((pl.col("split") == split_name) & pl.col("s1_id").is_in(present.implode()))["s1_id"]
    g = truth_all.filter(pl.col("source1_entity_id").is_in(ids.implode()))
    return {k: set(v.split(",")) if v else set() for k, v in zip(g["source1_entity_id"], g["matched_entity_ids"])}

def sweep(d, p, truth):
    best = (-1, None)
    for own in (True, False):
        for alpha in (0.0, 0.5, 0.7):
            for t in np.arange(0.30, 0.96, 0.05):
                r = slice_report(decide(d, p, t, alpha, own), truth, cty)["ALL"]["macro_f05"]
                if r > best[0]: best = (r, (own, alpha, float(t)))
    return best

def by_ntrue(pred, truth):
    s = per_entity_scores(pred, truth); rows = {}
    for k, v in s.items(): rows.setdefault(min(len(truth[k]), 5), []).append(v)
    return {k: (len(v), float(np.mean(v)), float(np.sum(v))) for k, v in sorted(rows.items())}

res = {}; pred2 = {}
for nm in ("val", "audit"):
    d = sc.filter(pl.col("split") == nm).select("s1_id", "t_id", "p")
    bb = band.filter(pl.col("split") == nm)
    p2 = model.predict(X_of(bb), num_threads=a.threads)
    d = d.join(bb.select("s1_id", "t_id").with_columns(pl.Series("p2", p2)), on=["s1_id", "t_id"], how="left", maintain_order="left")
    res[nm] = dict(d=d, p1=d["p"].to_numpy(), p2=d["p2"].fill_null(d["p"]).to_numpy(), truth=truth_for(nm, d["s1_id"].unique()))
del sc
V, A = res["val"], res["audit"]
b1 = sweep(V["d"], V["p1"], V["truth"]); log(f"\nSTAGE-1 baseline: best val {b1[0]:.5f} at ownership/alpha/t = {b1[1]} ({time.time()-t00:.0f}s)")
b2 = sweep(V["d"], V["p2"], V["truth"]); log(f"STAGE-2: best val {b2[0]:.5f} at ownership/alpha/t = {b2[1]} ({time.time()-t00:.0f}s)")
own, alpha, t = b2[1]
WITHHOLD = os.environ.get("STAGE2_WITHHOLD_AUDIT") == "1"   # selection protocol: audit metrics are not computed/printed (scores still written)
if WITHHOLD: log("STAGE2_WITHHOLD_AUDIT=1: audit metrics withheld during selection (evaluate the locked winner separately)")
log("stage-2 val/audit curve (ownership=%s alpha=%s):" % (own, alpha))
for tt in np.arange(0.40, 0.91, 0.05):
    rv = slice_report(decide(V["d"], V["p2"], tt, alpha, own), V["truth"], cty)["ALL"]["macro_f05"]
    if WITHHOLD: log(f"  t={tt:.2f} val {rv:.5f} audit (withheld)"); continue
    ra = slice_report(decide(A["d"], A["p2"], tt, alpha, own), A["truth"], cty)["ALL"]["macro_f05"]
    log(f"  t={tt:.2f} val {rv:.5f} audit {ra:.5f}")
summary = {}
for lab, pk, (o, al, tq) in (("stage-1", "p1", b1[1]), ("stage-2", "p2", b2[1])):
    for nm, R in ((("val", V),) if WITHHOLD else (("val", V), ("audit", A))):
        pr = decide(R["d"], R[pk], tq, al, o); rep = slice_report(pr, R["truth"], cty)
        log(f"\n{lab} {nm.upper()} at ownership={o} alpha={al} t={tq:.2f}"); print_report(rep)
        log("  by n_true (n, mean F, sum F):", {k: (v[0], round(v[1], 4), round(v[2], 1)) for k, v in by_ntrue(pr, R["truth"]).items()})
        summary[f"{lab}_{nm}"] = rep["ALL"]["macro_f05"]
log("\nSUMMARY:", {k: round(v, 5) for k, v in summary.items()})

os.makedirs("models", exist_ok=True); model.save_model(f"models/stage2_{a.tag}.txt")
json.dump(dict(features=FEATS, s1_feats=s1f, ce=bool(a.ce), ce_file=a.ce, **(dict(ce2=True, ce2_file=a.ce2) if a.ce2 else {}), **(dict(ce3=True, ce3_file=a.ce3) if a.ce3 else {}), ce_template=a.ce_template, field_disagreement=bool(a.field_disagreement), extra_feats_file=a.extra_feats, extra_feats=XF, countries=countries, p_min=S.P_MIN, rounds=rounds, seed=a.seed, ownership=own, alpha=alpha, t=round(t, 4),
               stage1_policy=dict(ownership=b1[1][0], alpha=b1[1][1], t=b1[1][2]), **{k: v for k, v in summary.items()}),
          open(f"models/stage2_{a.tag}.json", "w"), indent=1)
for nm, R in res.items():
    R["d"].with_columns(pl.Series("p_s2", R["p2"])).drop("p2").write_parquet(f"output/{nm}_scores_s2_{a.tag}.parquet")
log(f"saved models/stage2_{a.tag}.txt/.json, output/{{val,audit}}_scores_s2_{a.tag}.parquet  total {time.time()-t00:.0f}s")
