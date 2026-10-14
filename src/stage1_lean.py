"""Lean stage-1 LightGBM for large training samples (v11: 800k refs). Same recipe as train_matcher.py / oof_scores.py
(feature list = every numeric column not in ID_COLS + country_cat categorical; same params; early stopping 50 on val AUC for
the full fit; OOF = K folds grouped by s1_id with rng(fold_seed).permutation(n_refs) % K over sorted unique train s1_id and
a fixed number of rounds = best_iteration of the full model), but memory-lean:
  * `prep` streams the pairs file once in row batches into float32 C-order MEMMAPS on disk (<cache>/X_train.f32, X_va.f32 +
    ids parquet): no polars frame of the whole table, no hstack copies. LightGBM bins straight from the memmap (no copy);
    memmap pages are file-backed and reclaimable, so the anonymous RSS is ~ the binned Dataset (+ one fold subset).
  * `full`: fits models/lgbm_<tag>.{txt,json} (json keys as train_matcher.py) + the train_matcher decision sweep report
    + output/{val,audit}_scores_<tag>.parquet (s1_id,t_id,label,p).
  * `oof`: data/oof_scores_<tag>.parquet (s1_id,t_id,country,split,label,p) like oof_scores.py; fold rows are predicted
    in row chunks from the memmap; val/audit rows get the full model's p.
Run: python src/stage1_lean.py prep --pairs X --cache DIR ; python src/stage1_lean.py full --cache DIR --tag T [hp]
     python src/stage1_lean.py oof --cache DIR --tag T [same hp]
"""
import argparse, json, os, sys, time, gc, resource
import numpy as np, polars as pl, lightgbm as lgb, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prep", "full", "oof"])
ap.add_argument("--pairs", default=""); ap.add_argument("--cache", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--rounds", type=int, default=1500); ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--leaves", type=int, default=127); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--ff", type=float, default=0.8); ap.add_argument("--bf", type=float, default=0.8)
ap.add_argument("--min-leaf", type=int, default=100); ap.add_argument("--l2", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=None); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--fold-seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=2_000_000)
a = ap.parse_args()
T0 = time.time()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
def log(*x): print(f"[{time.time()-T0:7.0f}s maxrss {rss():.1f} GB]", *x, flush=True)
ID_COLS = {"s1_id", "t_id", "label", "split", "country", "sibling_conflict", "is_extra"}  # identical to train_matcher.py
C = a.cache; os.makedirs(C, exist_ok=True)

if a.mode == "prep":
    schema = pl.read_parquet_schema(a.pairs)
    feat_cols = [c for c, d in schema.items() if c not in ID_COLS and d.is_numeric()]
    ids = pl.read_parquet(a.pairs, columns=["s1_id", "t_id", "country", "split", "label"])
    countries = ids["country"].unique().sort().to_list(); cty_map = {c: i for i, c in enumerate(countries)}
    is_tr = (ids["split"] == "train").to_numpy(); ntr, nva = int(is_tr.sum()), int((~is_tr).sum())
    ids.filter(pl.Series(is_tr)).write_parquet(f"{C}/ids_train.parquet"); ids.filter(pl.Series(~is_tr)).write_parquet(f"{C}/ids_va.parquet")
    del ids; F = len(feat_cols) + 1
    json.dump(dict(features=feat_cols, countries=countries, n_train=ntr, n_va=nva, pairs=a.pairs), open(f"{C}/meta.json", "w"), indent=1)
    Xtr = np.lib.format.open_memmap(f"{C}/X_train.npy", mode="w+", dtype=np.float32, shape=(ntr, F))
    Xva = np.lib.format.open_memmap(f"{C}/X_va.npy", mode="w+", dtype=np.float32, shape=(nva, F))
    i = j = 0
    for b in pq.ParquetFile(a.pairs).iter_batches(batch_size=1_000_000, columns=["country", "split"] + feat_cols):
        d = pl.from_arrow(b); m = (d["split"] == "train").to_numpy()
        X = np.empty((len(d), F), dtype=np.float32)
        X[:, :-1] = d.select([pl.col(c).cast(pl.Float32) for c in feat_cols]).to_numpy()
        X[:, -1] = d["country"].replace_strict(cty_map, default=-1).to_numpy().astype(np.float32)
        k = int(m.sum()); Xtr[i:i + k] = X[m]; Xva[j:j + len(d) - k] = X[~m]; i += k; j += len(d) - k
        del d, X
    assert i == ntr and j == nva; Xtr.flush(); Xva.flush()
    log(f"prep done: train {ntr:,} va {nva:,} features {len(feat_cols)} -> {C}"); sys.exit(0)

meta = json.load(open(f"{C}/meta.json")); feat_cols, countries = meta["features"], meta["countries"]; names = feat_cols + ["country_cat"]
params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=a.ff,
              bagging_fraction=a.bf, bagging_freq=1, lambda_l2=a.l2, num_threads=a.threads, verbose=-1, metric="auc")
if a.seed is not None: params["seed"] = a.seed
log("lgb params:", {k: v for k, v in params.items() if k not in ("verbose", "num_threads")}, "threads", a.threads)
Xtr = np.load(f"{C}/X_train.npy", mmap_mode="r"); Xva = np.load(f"{C}/X_va.npy", mmap_mode="r")
ids_tr = pl.read_parquet(f"{C}/ids_train.parquet"); ids_va = pl.read_parquet(f"{C}/ids_va.parquet")
y = ids_tr["label"].to_numpy().astype(np.float32)
log(f"train rows {len(ids_tr):,} va rows {len(ids_va):,} features {len(feat_cols)}")

def predict_chunks(model, X, rows=None, num_iteration=None):
    n = len(X) if rows is None else len(rows); out = np.empty(n, dtype=np.float32)
    for lo in range(0, n, a.chunk):
        xb = np.ascontiguousarray(X[lo:lo + a.chunk] if rows is None else X[rows[lo:lo + a.chunk]])
        out[lo:lo + len(xb)] = model.predict(xb, num_iteration=num_iteration, num_threads=a.threads); del xb
    return out

if a.mode == "full":
    from scorer import slice_report, print_report
    vmask = (ids_va["split"] == "val").to_numpy(); vrows = np.flatnonzero(vmask); arows = np.flatnonzero(~vmask)
    dtr = lgb.Dataset(Xtr, label=y, feature_name=names, categorical_feature=["country_cat"], free_raw_data=True)
    dva = lgb.Dataset(np.ascontiguousarray(Xva[vrows]), label=ids_va["label"].to_numpy()[vmask].astype(np.float32), reference=dtr, free_raw_data=True)
    t0 = time.time()
    model = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    log(f"trained {model.best_iteration} rounds in {time.time()-t0:.0f}s; val AUC={model.best_score['valid_0']['auc']:.5f}")
    del dtr, dva; gc.collect()
    imp = sorted(zip(names, model.feature_importance("gain")), key=lambda x: -x[1]); print("top features by gain:", [(n, int(g)) for n, g in imp[:20]])
    va, au = ids_va.filter(pl.Series(vmask)), ids_va.filter(pl.Series(~vmask))
    pv = predict_chunks(model, Xva, vrows, model.best_iteration); pa = predict_chunks(model, Xva, arows, model.best_iteration)
    truth_all = pl.read_parquet("data/train_gt.parquet"); split = pl.read_parquet("data/split.parquet")
    def decide(d, p, t, alpha, ownership):  # == train_matcher.decide (margin 0)
        x = d.select("s1_id", "t_id").with_columns(pl.Series("p", p))
        if ownership:
            x = x.sort(["t_id", "p"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("t_id").alias("r")).filter(pl.col("r") == 0)
        x = x.with_columns(pl.col("p").max().over("s1_id").alias("pmax")).filter((pl.col("p") >= t) & (pl.col("p") >= alpha * pl.col("pmax")))
        g = x.group_by("s1_id").agg(pl.col("t_id")); return dict(zip(g["s1_id"], (set(v) for v in g["t_id"])))
    def truth_for(sn, present):
        ids = split.filter((pl.col("split") == sn) & pl.col("s1_id").is_in(present))["s1_id"]
        g = truth_all.filter(pl.col("source1_entity_id").is_in(ids))
        return {k: set(v.split(",")) if v else set() for k, v in zip(g["source1_entity_id"], g["matched_entity_ids"])}, dict(zip(split["s1_id"], split["country"]))
    truth_v, cty = truth_for("val", va["s1_id"].unique()); best = (-1, None)
    for own in (True, False):
        for alpha in (0.0, 0.5, 0.7):
            for t in np.arange(0.30, 0.96, 0.05):
                r = slice_report(decide(va, pv, t, alpha, own), truth_v, cty)["ALL"]["macro_f05"]
                if r > best[0]: best = (r, (own, alpha, float(t)))
    print(f"\nbest val macro F0.5 = {best[0]:.4f} at ownership={best[1][0]} alpha={best[1][1]} t={best[1][2]:.2f}")
    own, alpha, t = best[1]
    print("\nVAL slices at best setting:"); print_report(slice_report(decide(va, pv, t, alpha, own), truth_v, cty))
    truth_a, _ = truth_for("audit", au["s1_id"].unique())
    print("\nAUDIT slices at val-chosen setting:"); print_report(slice_report(decide(au, pa, t, alpha, own), truth_a, cty))
    for own2, al2, t2 in ((True, 0.7, 0.65), (False, 0.0, 0.75)):
        print(f"fixed own={own2} alpha={al2} t={t2}: val {slice_report(decide(va, pv, t2, al2, own2), truth_v, cty)['ALL']['macro_f05']:.5f} "
              f"audit {slice_report(decide(au, pa, t2, al2, own2), truth_a, cty)['ALL']['macro_f05']:.5f}")
    model.save_model(f"models/lgbm_{a.tag}.txt")
    json.dump(dict(features=feat_cols, countries=countries, ownership=own, alpha=alpha, t=t, best_iteration=model.best_iteration,
                   val_macro_f05=best[0], trained_by="stage1_lean.py", pairs=meta["pairs"]), open(f"models/lgbm_{a.tag}.json", "w"), indent=1)
    va.select("s1_id", "t_id", "label").with_columns(pl.Series("p", pv)).write_parquet(f"output/val_scores_{a.tag}.parquet")
    au.select("s1_id", "t_id", "label").with_columns(pl.Series("p", pa)).write_parquet(f"output/audit_scores_{a.tag}.parquet")
    log(f"saved models/lgbm_{a.tag}.*, output/{{val,audit}}_scores_{a.tag}.parquet"); sys.exit(0)

# ---- oof ----
from sklearn.metrics import roc_auc_score
cfg = json.load(open(f"models/lgbm_{a.tag}.json")); assert cfg["features"] == feat_cols; rounds = int(cfg["best_iteration"])
log(f"OOF {a.folds} folds, rounds {rounds}")
u = ids_tr["s1_id"].unique().sort(); rng = np.random.default_rng(a.fold_seed); fold_of = rng.permutation(len(u)) % a.folds
fold = ids_tr.select("s1_id").join(pl.DataFrame({"s1_id": u, "f": fold_of.astype(np.int8)}), on="s1_id", how="left", maintain_order="left")["f"].to_numpy()
# free_raw_data=True: with False, every full.subset() copies the raw rows (reference.data[used_indices]) = +19 GB anon per fold
full = lgb.Dataset(Xtr, label=y, feature_name=names, categorical_feature=["country_cat"], free_raw_data=True, params={"verbose": -1})
full.construct(); log("binned")
import mmap
def drop_pages():  # file-backed memmap pages are only needed while binning / predicting fold rows: release them from RSS
    try: Xtr._mmap.madvise(mmap.MADV_DONTNEED)
    except Exception as e: log(f"madvise failed: {e}")
drop_pages(); log("memmap pages released")
oof = np.zeros(len(y), dtype=np.float32)
for f in range(a.folds):
    t0 = time.time(); tr = np.flatnonzero(fold != f).astype(np.int32); te = np.flatnonzero(fold == f)
    m = lgb.train(params, full.subset(tr), num_boost_round=rounds); del tr
    oof[te] = predict_chunks(m, Xtr, te)
    log(f"fold {f}: test {len(te):,}  AUC {roc_auc_score(y[te], oof[te]):.5f}  {time.time()-t0:.0f}s")
    np.save(f"{C}/oof_{a.tag}_partial.npy", oof); del m; gc.collect(); drop_pages()
del full; gc.collect()
model = lgb.Booster(model_file=f"models/lgbm_{a.tag}.txt"); pva = predict_chunks(model, Xva, None, cfg["best_iteration"])
out = pl.concat([ids_tr.select("s1_id", "t_id", "country", "split", pl.col("label").cast(pl.Int8)).with_columns(pl.Series("p", oof)),
                 ids_va.select("s1_id", "t_id", "country", "split", pl.col("label").cast(pl.Int8)).with_columns(pl.Series("p", pva))])
out.write_parquet(f"data/oof_scores_{a.tag}.parquet")
print(out.group_by("split").agg(pl.len(), pl.col("p").mean().alias("mean_p"), pl.col("label").mean().alias("pos")))
log(f"wrote data/oof_scores_{a.tag}.parquet ({len(out):,} rows)")
