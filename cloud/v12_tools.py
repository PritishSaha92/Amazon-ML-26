"""v12 (cloud, ALL labelled train references) helpers. Run from the project root. No existing src/ file is modified;
the recipe code is src/stage1_lean.py / src/v11_ce.py / src/pair_features.py etc. These helpers only split, merge and check.

  complement --out F          train-split S1 NOT in data/pairs_train_v7.parquet's train rows -> F (column s1_id).
                              (v7 file already holds its 400k train refs + all 80k val/audit refs.)
                              Uses data/split.parquet and the v7 train-ref list data/v12/s1_v7train.parquet (built locally,
                              uploaded; = train-split s1_id of pairs_train_v7_feat.parquet) -> no 1.8 GB pairs file upload.
  shard --pairs P --n N --outdir D      row-split a build_pairs output into N parquet shards (pair_features is row-wise:
                              every per-S1 / per-target context column is already computed by build_pairs on the full union;
                              v7's own file was featurised in 2M-row chunks).
  concat --inputs a,b,... --out O       stream row groups into one zstd parquet (schemas must be identical, asserted).
  fold --cache C --tag T --fold f [hp]  ONE OOF fold of src/stage1_lean.py `oof`, bit-for-bit the same recipe: same fold
                              assignment (rng(fold_seed).permutation(n_refs) % K over sorted unique train s1_id), full Dataset
                              binned from the memmap then .subset(train rows), rounds = best_iteration of models/lgbm_<T>.json,
                              chunked prediction. Writes C/oof_<T>_f<f>.npy (p for the fold's rows in row order).
  merge --cache C --tag T [--folds 5]   assemble data/oof_scores_<T>.parquet exactly like stage1_lean.py `oof` (train rows =
                              fold predictions, val/audit = full model p) after asserting every train row is covered once.
  coverage --scores S --ce-file F --out J   CE fold-ownership audit of the assembled band (reviewer rule): for every band row,
                              which CE(s) scored it, and assert no row was scored by a CE whose TRAINING manifest
                              (data/cefold_v7/ce_train_hard_v7f{A,B}.parquet s1_id) contains the row's S1; JSON summary.
  slim --pairs P --out O      s1_id,t_id,country,split,label + stage-2 pass-through columns (stage2_feats.S1F) of the full
                              features file, so stage 2's band can be rebuilt locally without the 14 GB features file.
"""
import argparse, json, os, sys, time, gc, resource
import numpy as np, polars as pl, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["complement", "shard", "concat", "fold", "merge", "coverage", "slim"])
ap.add_argument("--out", default=""); ap.add_argument("--pairs", default=""); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--outdir", default=""); ap.add_argument("--inputs", default="")
ap.add_argument("--cache", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--fold", type=int, default=-1)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--fold-seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=24); ap.add_argument("--chunk", type=int, default=2_000_000)
ap.add_argument("--leaves", type=int, default=127); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--ff", type=float, default=0.8); ap.add_argument("--bf", type=float, default=0.8)
ap.add_argument("--min-leaf", type=int, default=100); ap.add_argument("--l2", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--scores", default=""); ap.add_argument("--ce-dir", default=""); ap.add_argument("--ce-file", default=""); ap.add_argument("--band", default="0.02,0.98")
a = ap.parse_args(); T0 = time.time()
def log(*x): print(f"[{time.time()-T0:7.0f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB]", *x, flush=True)

if a.mode == "complement":
    sp = pl.read_parquet("data/split.parquet")
    old = pl.read_parquet("data/v12/s1_v7train.parquet").select("s1_id").unique()
    assert len(old) == 400_000, len(old)
    assert old.join(sp.filter(pl.col("split") == "train"), on="s1_id", how="anti").is_empty(), "v7 train refs not all train-split"
    new = sp.filter(pl.col("split") == "train").join(old, on="s1_id", how="anti")
    log("train-split refs", int((sp["split"] == "train").sum()), "| in v7 file", len(old), "| complement", len(new),
        new.group_by("country").len().sort("country").to_dict(as_series=False))
    new.select("s1_id").sort("s1_id").write_parquet(a.out); log("wrote", a.out)

elif a.mode == "shard":
    os.makedirs(a.outdir, exist_ok=True); pf = pq.ParquetFile(a.pairs); n = pf.metadata.num_rows
    per = -(-n // a.n); shard, w, k, tot = 0, None, 0, 0
    for b in pf.iter_batches(batch_size=1_000_000):
        off = 0
        while off < b.num_rows:
            if w is None: w = pq.ParquetWriter(f"{a.outdir}/shard_{shard:02d}.parquet", b.schema, compression="zstd")
            take = min(per - k, b.num_rows - off); w.write_batch(b.slice(off, take)); k += take; off += take; tot += take
            if k == per: w.close(); w = None; shard += 1; k = 0
    if w is not None: w.close(); shard += 1
    assert tot == n; log(f"{n:,} rows -> {shard} shards of <= {per:,} rows in {a.outdir}")

elif a.mode == "concat":
    ins = [f for f in a.inputs.split(",") if f]; fs = [pq.ParquetFile(f) for f in ins]; sch = fs[0].schema_arrow
    for f, p in zip(fs, ins): assert f.schema_arrow.equals(sch), (p, set(f.schema_arrow.names) ^ set(sch.names))
    w = pq.ParquetWriter(a.out + ".tmp", sch, compression="zstd"); n = 0
    for f in fs:
        for i in range(f.num_row_groups):
            t = f.read_row_group(i); w.write_table(t, row_group_size=1_000_000); n += t.num_rows
    w.close(); os.replace(a.out + ".tmp", a.out)
    assert n == sum(f.metadata.num_rows for f in fs); log("wrote", a.out, f"{n:,} rows =", " + ".join(f"{f.metadata.num_rows:,}" for f in fs))

elif a.mode in ("fold", "merge"):
    import lightgbm as lgb
    C = a.cache; meta = json.load(open(f"{C}/meta.json")); feat_cols = meta["features"]; names = feat_cols + ["country_cat"]
    cfg = json.load(open(f"models/lgbm_{a.tag}.json")); assert cfg["features"] == feat_cols; rounds = int(cfg["best_iteration"])
    ids_tr = pl.read_parquet(f"{C}/ids_train.parquet")
    u = ids_tr["s1_id"].unique().sort(); rng = np.random.default_rng(a.fold_seed); fold_of = rng.permutation(len(u)) % a.folds
    fold = ids_tr.select("s1_id").join(pl.DataFrame({"s1_id": u, "f": fold_of.astype(np.int8)}), on="s1_id", how="left", maintain_order="left")["f"].to_numpy()
    Xtr = np.load(f"{C}/X_train.npy", mmap_mode="r"); y = ids_tr["label"].to_numpy().astype(np.float32)
    def predict_chunks(model, X, rows=None, num_iteration=None):  # == stage1_lean.predict_chunks
        n = len(X) if rows is None else len(rows); out = np.empty(n, dtype=np.float32)
        for lo in range(0, n, a.chunk):
            xb = np.ascontiguousarray(X[lo:lo + a.chunk] if rows is None else X[rows[lo:lo + a.chunk]])
            out[lo:lo + len(xb)] = model.predict(xb, num_iteration=num_iteration, num_threads=a.threads); del xb
        return out
    if a.mode == "fold":
        assert 0 <= a.fold < a.folds
        params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=a.ff,
                      bagging_fraction=a.bf, bagging_freq=1, lambda_l2=a.l2, num_threads=a.threads, verbose=-1, metric="auc")
        if a.seed is not None: params["seed"] = a.seed
        log("tag", a.tag, "fold", a.fold, "rounds", rounds, "params", {k: v for k, v in params.items() if k not in ("verbose",)})
        full = lgb.Dataset(Xtr, label=y, feature_name=names, categorical_feature=["country_cat"], free_raw_data=False, params={"verbose": -1})
        full.construct(); log("binned")
        tr = np.flatnonzero(fold != a.fold).astype(np.int32); te = np.flatnonzero(fold == a.fold)
        m = lgb.train(params, full.subset(tr), num_boost_round=rounds); del tr, full; gc.collect(); log("trained")
        p = predict_chunks(m, Xtr, te)
        from sklearn.metrics import roc_auc_score
        np.save(f"{C}/oof_{a.tag}_f{a.fold}.npy", p); m.save_model(f"{C}/oof_{a.tag}_f{a.fold}.txt")
        log(f"fold {a.fold}: test {len(te):,} AUC {roc_auc_score(y[te], p):.5f} -> {C}/oof_{a.tag}_f{a.fold}.npy")
    else:
        oof = np.full(len(y), np.nan, dtype=np.float32)
        for f in range(a.folds):
            te = np.flatnonzero(fold == f); p = np.load(f"{C}/oof_{a.tag}_f{f}.npy"); assert len(p) == len(te), (f, len(p), len(te))
            assert np.isnan(oof[te]).all(); oof[te] = p
        assert not np.isnan(oof).any(), "train rows without an OOF prediction"
        from sklearn.metrics import roc_auc_score
        log(f"OOF AUC all folds {roc_auc_score(y, oof):.5f}")
        ids_va = pl.read_parquet(f"{C}/ids_va.parquet"); Xva = np.load(f"{C}/X_va.npy", mmap_mode="r")
        model = lgb.Booster(model_file=f"models/lgbm_{a.tag}.txt"); pva = predict_chunks(model, Xva, None, cfg["best_iteration"])
        out = pl.concat([ids_tr.select("s1_id", "t_id", "country", "split", pl.col("label").cast(pl.Int8)).with_columns(pl.Series("p", oof)),
                         ids_va.select("s1_id", "t_id", "country", "split", pl.col("label").cast(pl.Int8)).with_columns(pl.Series("p", pva))])
        out.write_parquet(f"data/oof_scores_{a.tag}.parquet")
        print(out.group_by("split").agg(pl.len(), pl.col("p").mean().alias("mean_p"), pl.col("label").mean().alias("pos")))
        log(f"wrote data/oof_scores_{a.tag}.parquet ({len(out):,} rows)")

elif a.mode == "coverage":
    K = ["s1_id", "t_id"]; lo, hi = map(float, a.band.split(","))
    band = pl.read_parquet(a.scores, columns=K + ["split", "p"]).filter(pl.col("p").is_between(lo, hi)).drop("p")
    folds = pl.read_parquet("data/cefold_v7/folds.parquet")
    manA = pl.read_parquet("data/cefold_v7/ce_train_hard_v7fA.parquet", columns=["s1_id"]).unique()
    manB = pl.read_parquet("data/cefold_v7/ce_train_hard_v7fB.parquet", columns=["s1_id"]).unique()
    b = band.join(folds, on="s1_id", how="left").with_columns(
        inA=pl.col("s1_id").is_in(manA["s1_id"].implode()), inB=pl.col("s1_id").is_in(manB["s1_id"].implode()))
    b = b.with_columns(usesA=pl.col("fold").is_null() | (pl.col("fold") == "B"), usesB=pl.col("fold").is_null() | (pl.col("fold") == "A"))
    viol = b.filter((pl.col("usesA") & pl.col("inA")) | (pl.col("usesB") & pl.col("inB")))
    grp = (pl.when(pl.col("split") != "train").then(pl.col("split")).when(pl.col("fold").is_null()).then(pl.lit("train_new"))
           .otherwise(pl.lit("train_old_") + pl.col("fold")))
    route = pl.when(pl.col("usesA") & pl.col("usesB")).then(pl.lit("mean(A,B)")).when(pl.col("usesA")).then(pl.lit("A only")).otherwise(pl.lit("B only"))
    tab = b.with_columns(grp.alias("grp"), route.alias("route")).group_by("grp", "route").agg(
        rows=pl.len(), refs=pl.col("s1_id").n_unique(), in_manA=pl.col("inA").sum(), in_manB=pl.col("inB").sum()).sort("grp")
    print(tab)
    ce = pl.read_parquet(a.ce_file) if a.ce_file else None  # assembled s1_id,t_id,p_ce (must cover every band row)
    res = dict(band=a.band, band_rows=len(b), manifest_A_s1=len(manA), manifest_B_s1=len(manB), violations=len(viol),
               groups=tab.to_dicts(), assembled_rows=(len(ce) if ce is not None else None),
               assembled_missing=(len(band.join(ce, on=K, how="anti")) if ce is not None else None))
    json.dump(res, open(a.out, "w"), indent=1, default=str); log("coverage ->", a.out, json.dumps({k: v for k, v in res.items() if k != "groups"}))
    assert len(viol) == 0, f"{len(viol)} band rows would be scored by a CE that trained on their S1"
    if ce is not None: assert res["assembled_missing"] == 0, "band rows without an assembled p_ce"

elif a.mode == "slim":
    sys.path.insert(0, "src"); import stage2_feats as S
    cols = ["s1_id", "t_id", "country", "split", "label"] + S.S1F; pf = pq.ParquetFile(a.pairs)
    w = None
    for b in pf.iter_batches(batch_size=2_000_000, columns=cols):
        if w is None: w = pq.ParquetWriter(a.out + ".tmp", b.schema, compression="zstd")
        w.write_batch(b)
    w.close(); os.replace(a.out + ".tmp", a.out); log("wrote", a.out, pq.ParquetFile(a.out).metadata.num_rows, "rows", cols)
