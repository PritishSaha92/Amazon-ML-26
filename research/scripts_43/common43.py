"""research 43 harness: fast fixed decoder (== cloud/v12_eval.decide: ownership argmax per target among the split's refs,
p >= t and p >= alpha * max owned p of the ref) + a stage-2 fit that replicates src/stage2.py exactly (params, 10% train-ref
holdout with seed-0 permutation, early stop 100, refit 1.1x) and writes output/{val,audit}_scores_s2_<tag>.parquet in
stage2.py's format (so cloud/v12_eval.py reads it). Audit is predicted/saved but never scored here."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, json, time
import numpy as np, polars as pl
ROOT = _BER_R; os.chdir(ROOT); sys.path.insert(0, "src")
import stage2_feats as S
K = ["s1_id", "t_id"]
TH = int(os.environ.get("R43_THREADS", "8"))
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)

class Split:
    """All S1 of a split (incl. those without candidates → empty prediction) + truth; score(rows, p)."""
    def __init__(self, nm):
        sp = pl.read_parquet("data/split.parquet").filter(pl.col("split") == nm)
        self.ents = sp["s1_id"]; self.n = len(sp)
        self.eidx = pl.DataFrame({"s1_id": sp["s1_id"], "e": np.arange(self.n, dtype=np.int64)})
        gt = pl.read_parquet("data/train_gt.parquet").filter(pl.col("source1_entity_id").is_in(sp["s1_id"].implode()))
        tr = gt.select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").str.split(",").alias("t_id")).explode("t_id")
        tr = tr.filter(pl.col("t_id").is_not_null() & (pl.col("t_id") != ""))
        self.truth_pairs = tr.with_columns(pl.lit(1.0).alias("y"))
        nt = tr.group_by("s1_id").agg(pl.len().alias("nt"))
        self.ntrue = self.eidx.join(nt, on="s1_id", how="left", maintain_order="left")["nt"].fill_null(0).to_numpy().astype(np.float64)
        self.cty = self.eidx.join(sp.select("s1_id", "country"), on="s1_id", how="left", maintain_order="left")["country"].to_numpy()

    def prep(self, d: pl.DataFrame):
        """d: s1_id,t_id rows of this split → (e, tg, y) arrays aligned with d."""
        x = d.select(K).join(self.eidx, on="s1_id", how="left", maintain_order="left").join(self.truth_pairs, on=K, how="left", maintain_order="left")
        assert x["e"].null_count() == 0
        tg = x["t_id"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
        return dict(e=x["e"].to_numpy(), tg=tg, y=x["y"].fill_null(0.0).to_numpy())

    def score(self, P, p, ts=(0.5,), alpha=0.7, per_country=False, raw=False):
        e, tg, y = P["e"], P["tg"], P["y"]; p = np.asarray(p, np.float64)
        o = np.lexsort((-p, tg)); first = np.ones(len(o), bool); first[1:] = tg[o][1:] != tg[o][:-1]
        own = o[first]; e, y, pp = e[own], y[own], p[own]
        pmax = np.zeros(self.n); np.maximum.at(pmax, e, pp)
        out = {}
        for t in ts:
            sel = (pp >= t) & (pp >= alpha * pmax[e])
            tp = np.bincount(e[sel], weights=y[sel], minlength=self.n); npred = np.bincount(e[sel], minlength=self.n).astype(float)
            fp = npred - tp; fn = self.ntrue - tp
            f = np.where(tp > 0, 1.25 * tp / np.maximum(1.25 * tp + fp + 0.25 * fn, 1e-12), 0.0)
            f = np.where(self.ntrue == 0, (npred == 0).astype(float), f)
            if raw: out[round(float(t), 2)] = f; continue
            out[round(float(t), 2)] = (f.mean(), {c: f[self.cty == c].mean() for c in np.unique(self.cty)}) if per_country else f.mean()
        return out

TGRID = tuple(np.round(np.arange(0.30, 0.951, 0.05), 2))

def X_of(d, feats): return d.select([pl.col(c).cast(pl.Float32).fill_nan(None) for c in feats]).to_numpy()

def fit_stage2(band, feats, seed=0, threads=TH, max_rounds=3000):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, num_threads=threads, verbose=-1, metric="binary_logloss", seed=seed)
    tr = band.filter(pl.col("split") == "train")
    Xtr, ytr = X_of(tr, feats), tr["label"].to_numpy()
    u = tr["s1_id"].unique().sort(); rng = np.random.default_rng(0); ho = set(u.to_numpy()[rng.permutation(len(u))[: len(u) // 10]])
    m_ho = tr["s1_id"].is_in(list(ho)).to_numpy()
    dk = dict(feature_name=feats, categorical_feature=["country_cat"])
    t0 = time.time()
    m0 = lgb.train(params, lgb.Dataset(Xtr[~m_ho], ytr[~m_ho], **dk), max_rounds, valid_sets=[lgb.Dataset(Xtr[m_ho], ytr[m_ho], **dk)],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = int(m0.best_iteration * 1.1); ho_ll = m0.best_score["valid_0"]["binary_logloss"]
    model = lgb.train(params, lgb.Dataset(Xtr, ytr, **dk), rounds); del Xtr
    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(g for _, g in imp)
    log(f"  seed {seed}: best_it {m0.best_iteration} holdout ll {ho_ll:.6f} refit {rounds} ({time.time()-t0:.0f}s); top gain",
        [(n, round(g / tot, 4)) for n, g in imp[:12]])
    return model, dict(rounds=rounds, ho_ll=ho_ll, gain={n: g / tot for n, g in imp})

def write_split_scores(model, band, feats, sc_split, nm, tag, threads=TH):
    """stage2.py output format: all rows of the split's stage-1 scores (s1_id,t_id,p) + p_s2 (band → model, else p)."""
    bb = band.filter(pl.col("split") == nm)
    p2 = model.predict(X_of(bb, feats), num_threads=threads)
    d = sc_split.join(bb.select(K).with_columns(pl.Series("p2", p2)), on=K, how="left", maintain_order="left")
    d = d.with_columns(pl.col("p2").fill_null(pl.col("p")).alias("p_s2")).drop("p2")
    d.write_parquet(f"output/{nm}_scores_s2_{tag}.parquet"); return d

def s3avg(tags, nm, out_tag):
    fs = [pl.read_parquet(f"output/{nm}_scores_s2_{t}.parquet") for t in tags]
    base = fs[0]
    for f in fs[1:]: assert f.select(K).equals(base.select(K))
    d = base.with_columns((sum(f["p_s2"].cast(pl.Float64) for f in fs) / len(fs)).cast(base["p_s2"].dtype).alias("p_s2"))
    d.write_parquet(f"output/{nm}_scores_s2_{out_tag}.parquet"); return d
