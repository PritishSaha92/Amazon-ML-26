"""NODE-SIDE (needs the stage1_lean cache: X_train.npy, ids_train.parquet, X_va.npy, ids_va.parquet, meta.json, and the fold
models C/oof_<tag>_f<k>.txt + models/lgbm_<tag>.{txt,json}). Producer-consistent rescoring for research 43 E2 (astra's recipe):
  fold part: for each fold k and tag, every rival row (r,t) of a fold-k TRAIN band target t with r NOT in fold k (other train folds
             and val/audit) is scored by the fold-k model C/oof_<tag>_f<k>.txt (the model that excluded fold k, so never saw the
             query's label). Same-fold rivals already carry fold-k OOF p.
  full part: every TRAIN rival row of a VAL/AUDIT band target is scored by the full-fit models/lgbm_<tag> (num_iteration =
             best_iteration, as val/audit p), so val/audit queries see ONE producer (full-fit A/B average), as at test.
Rivals with OOF/full p < --pmin are not rescored (they keep their existing p; they cannot move c_best/c_n50/c_rank of a band row).
Output: --out parquet (part, k, s1_id, t_id, p_<tag>...). Fold assignment = v12_tools/stage1_lean (rng(fold_seed).permutation % K).
Run (node): python research/scripts_43/node_producer_rescore.py --cache $CACHE --scores data/oof_scores_v12avg.parquet
            --band data/s2_band_v12avg.parquet --tags v12,v12b --out data/v12/r43_rescore.parquet --threads 40"""
import argparse, json, time, gc
import numpy as np, polars as pl, lightgbm as lgb
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--scores", required=True); ap.add_argument("--band", required=True)
ap.add_argument("--tags", default="v12,v12b"); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--fold-seed", type=int, default=0); ap.add_argument("--pmin", type=float, default=0.001)
ap.add_argument("--chunk", type=int, default=2_000_000); ap.add_argument("--fold-model", default="{C}/oof_{tag}_f{k}.txt")
ap.add_argument("--limit-targets", type=int, default=0, help="smoke test: random subset of band targets")
a = ap.parse_args(); T0 = time.time(); C = a.cache; K = ["s1_id", "t_id"]
log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
tags = a.tags.split(",")
ids_tr = pl.read_parquet(f"{C}/ids_train.parquet", columns=K).with_row_index("row")
ids_va = pl.read_parquet(f"{C}/ids_va.parquet", columns=K + ["split"]).with_row_index("row")
u = ids_tr["s1_id"].unique().sort(); fold_of = np.random.default_rng(a.fold_seed).permutation(len(u)) % a.folds
folds = pl.DataFrame({"s1_id": u, "f": fold_of.astype(np.int8)})
sc = pl.read_parquet(a.scores, columns=K + ["split", "p"])
band = pl.read_parquet(a.band, columns=K + ["split"])
if a.limit_targets:
    tt = band["t_id"].unique().sort().sample(a.limit_targets, seed=0); band = band.filter(pl.col("t_id").is_in(tt.implode()))
band = band.join(folds, on="s1_id", how="left")
Xtr = np.load(f"{C}/X_train.npy", mmap_mode="r"); Xva = np.load(f"{C}/X_va.npy", mmap_mode="r")
def pred(model, X, rows, num_iteration=None):
    order = np.argsort(rows); rs = rows[order]; out = np.empty(len(rows), np.float32)
    for lo in range(0, len(rs), a.chunk):
        out[order[lo:lo + a.chunk]] = model.predict(np.ascontiguousarray(X[rs[lo:lo + a.chunk]]), num_iteration=num_iteration, num_threads=a.threads)
    return out
tr_idx = ids_tr.join(folds, on="s1_id", how="left").join(sc.select(K + ["p"]), on=K, how="left")
va_idx = ids_va.join(sc.select(K + ["p"]), on=K, how="left")
assert tr_idx["p"].null_count() == 0 and va_idx["p"].null_count() == 0
parts = []
for k in range(a.folds):
    Tk = band.filter((pl.col("split") == "train") & (pl.col("f") == k)).select("t_id").unique()
    rt = tr_idx.join(Tk, on="t_id", how="semi").filter((pl.col("f") != k) & (pl.col("p") >= a.pmin))
    rv = va_idx.join(Tk, on="t_id", how="semi").filter(pl.col("p") >= a.pmin)
    out = pl.concat([rt.select(K), rv.select(K)]).with_columns(pl.lit("fold").alias("part"), pl.lit(k, pl.Int8).alias("k"))
    for tag in tags:
        m = lgb.Booster(model_file=a.fold_model.format(C=C, tag=tag, k=k))
        p = np.concatenate([pred(m, Xtr, rt["row"].to_numpy()), pred(m, Xva, rv["row"].to_numpy())]); del m; gc.collect()
        out = out.with_columns(pl.Series(f"p_{tag}", p))
    parts.append(out); log(f"fold {k}: {len(Tk):,} band targets, rescored {len(rt):,} train + {len(rv):,} val/audit rival rows")
Tva = band.filter(pl.col("split") != "train").select("t_id").unique()
rt = tr_idx.join(Tva, on="t_id", how="semi").filter(pl.col("p") >= a.pmin)
out = rt.select(K).with_columns(pl.lit("full").alias("part"), pl.lit(-1, pl.Int8).alias("k"))
for tag in tags:
    cfg = json.load(open(f"models/lgbm_{tag}.json")); m = lgb.Booster(model_file=f"models/lgbm_{tag}.txt")
    out = out.with_columns(pl.Series(f"p_{tag}", pred(m, Xtr, rt["row"].to_numpy(), int(cfg["best_iteration"]))))
parts.append(out); log(f"full: {len(Tva):,} val/audit band targets, rescored {len(rt):,} train rival rows")
res = pl.concat(parts); res.write_parquet(a.out); log(f"wrote {a.out}: {len(res):,} rows")
