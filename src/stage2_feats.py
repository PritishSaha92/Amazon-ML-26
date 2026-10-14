"""Stage-2 feature construction (shared by stage2.py and stage2_apply.py). Spec: research/08_stage2_collective.md.

Input `sc`: one row per candidate pair (s1_id, t_id, country, p [, split, label]) covering ALL candidates of every
reference present (set features need the complete candidate list). Output: rows with p > P_MIN, with
  set structure per reference : p, logit, rank, p above/below + gaps, counts p>0.1/0.5/0.9, sum excl. self, max, n cands,
                                rank within same source, # of that source with p>0.9, source, country
  consensus vs the reference's confident records (p>0.9, excl. self): # sharing house number / addr_core / name_core,
                                house number / name equal to the reference's
  e5 record-to-record         : cos to reference, max cos to confident, p-weighted mean cos, max cos to any, # cos>0.95,
                                cos to centroid of confident (excl. self), differences vs the reference cos
  stage-1 pass-through        : S1F columns joined from the stage-1 features file.
Nothing is computed from other references' scores for the same record (no target-side p features).
"""
import time
import numpy as np, polars as pl, pyarrow.parquet as pq

P_MIN = 0.01
S1F = ["t_addr_empty", "t_name_core_s1_count", "t_addr_core_s1_count", "num_grade", "hn_grade", "t_margin", "t_n_owners",
       "t_best_other_score", "s1_n_cands", "nc_jw", "ac_tset", "max_score"]
SET = ["p", "logit", "s_rank", "s_n", "s_n10", "s_n50", "s_n90", "s_sum", "s_max", "p_above", "p_below", "gap_below",
       "gap_above", "s_sum_others", "s_n90_others", "s_src_rank", "s_src_n90", "is_s2", "country_cat"]
CONS = ["tt_same_name", "tt_same_hn", "tt_same_addr", "hn_eq_s1", "name_eq_s1"]
CE = ["p_ce", "ce_logit", "ce_minus_p", "ce_logit_minus_logit"]  # optional (--ce): cross-encoder p inside the stage-1 band
CE2 = ["p_ce2", "ce2_logit", "ce2_minus_p", "ce2_minus_ce"]  # optional (--ce2): a second cross-encoder family (e.g. e5-large folds)
CE3 = ["p_ce3", "ce3_logit", "ce3_minus_p", "ce3_minus_ce"]  # optional (--ce3): a third cross-encoder family (mirrors CE2)
# --ce-template independent: CE2/CE3 built with the add_ce template (4th feature = family logit - stage-1 logit), as in
# research/astra_three_ce_stack.py; "default" = the CE2/CE3 lists above (4th feature = family p - p_ce).
CE_TEMPLATES = ("default", "independent")
CE_BAND = (0.02, 0.98)  # stage-1 p range every CE family must score (chain_v7ce_folds.sh / ce2_arm.sh band rule)
FIELD = ["field_signed", "field_abs"]  # optional (--field-disagreement, default template, needs --ce2 + --ce3): name CE (ce3) vs address CE (ce2)
CE2_IND = ["p_ce2", "ce2_logit", "ce2_minus_p", "ce2_logit_minus_logit"]
CE3_IND = ["p_ce3", "ce3_logit", "ce3_minus_p", "ce3_logit_minus_logit"]
EMB = ["cos_s1", "tt_maxconf", "tt_wmean", "tt_maxany", "tt_n95", "tt_cent", "tt_cent_minus_s1", "tt_maxconf_minus_s1"]


def set_features(sc: pl.DataFrame, countries: list) -> pl.DataFrame:
    cty_map = {c: i for i, c in enumerate(countries)}
    sc = sc.with_columns(s_sum=pl.col("p").sum().over("s1_id"), s_n=pl.len().over("s1_id"),
                         s_n10=(pl.col("p") > 0.1).sum().over("s1_id"), s_n50=(pl.col("p") > 0.5).sum().over("s1_id"),
                         s_n90=(pl.col("p") > 0.9).sum().over("s1_id"), s_max=pl.col("p").max().over("s1_id"),
                         s_rank=pl.col("p").rank("ordinal", descending=True).over("s1_id"))
    b = sc.filter(pl.col("p") > P_MIN); del sc
    b = b.with_columns(src=pl.col("t_id").str.slice(0, 2)).sort(["s1_id", "p"], descending=[False, True])
    b = b.with_columns(p_above=pl.col("p").shift(1).over("s1_id"), p_below=pl.col("p").shift(-1).over("s1_id").fill_null(0.0),
                       s_src_rank=pl.col("p").rank("ordinal", descending=True).over(["s1_id", "src"]),
                       s_src_n90=(pl.col("p") > 0.9).sum().over(["s1_id", "src"]))
    return b.with_columns(gap_below=pl.col("p") - pl.col("p_below"), gap_above=(pl.col("p_above") - pl.col("p")).fill_null(0),
                          s_sum_others=pl.col("s_sum") - pl.col("p"), s_n90_others=pl.col("s_n90") - (pl.col("p") > 0.9).cast(pl.UInt32),
                          logit=(pl.col("p") / (1 - pl.col("p") + 1e-9)).log(), is_s2=(pl.col("src") == "S2").cast(pl.Int8),
                          country_cat=pl.col("country").replace_strict(cty_map, default=-1, return_dtype=pl.Int32)).drop("src")


class EmbStore:
    """fp16 e5 embeddings on disk (mmap) + id → row lookup."""
    def __init__(self, emb_path, ids_path):
        self.E = np.load(emb_path, mmap_mode="r")
        self.idx = pl.DataFrame({"id": np.load(ids_path, allow_pickle=True).astype(str), "row": np.arange(len(self.E), dtype=np.int64)})

    def rows(self, ids: pl.Series) -> np.ndarray:
        r = pl.DataFrame({"id": ids}).join(self.idx, on="id", how="left", maintain_order="left")["row"]
        assert r.null_count() == 0, f"{r.null_count()} ids without embedding"
        return r.to_numpy()

    def gather(self, rows: np.ndarray) -> np.ndarray:
        u, inv = np.unique(rows, return_inverse=True)  # sorted unique → sequential-ish mmap reads
        return np.asarray(self.E[u], dtype=np.float32)[inv]


def _emb_chunk(et, p, starts, sizes, out, off):
    """et/p: rows of whole groups (local indexing); starts local; writes into out[...] at off + local index."""
    L = 2
    while sizes.size and L // 2 < sizes.max():
        sel = np.flatnonzero((sizes >= 2) & (sizes <= L) & (sizes > L // 2))
        G = max(1, int(3e8 // (L * L * 4 * 4 + L * 384 * 4 * 3)))  # ~300 MB per sub-batch
        for c in range(0, len(sel), G):
            gs = sel[c:c + G]; st = starts[gs]; sz = sizes[gs]
            M = np.arange(L)[None, :] < sz[:, None]; I = np.where(M, st[:, None] + np.arange(L)[None, :], 0)
            X = et[I] * M[..., None]; P = np.where(M, p[I], 0.0).astype(np.float32); conf = (P > 0.9) & M
            S = np.einsum("gid,gjd->gij", X, X)
            valid = M[:, :, None] & M[:, None, :] & ~np.eye(L, dtype=bool)[None]
            maxany = np.where(valid, S, -np.inf).max(2)
            W = np.where(valid, P[:, None, :], 0.0)
            wmean = (np.where(valid, S, 0) * W).sum(2) / np.maximum(W.sum(2), 1e-9)
            n95 = (valid & (S > 0.95)).sum(2)
            maxconf = np.where(valid & conf[:, None, :], S, -np.inf).max(2); maxconf[np.isinf(maxconf)] = np.nan
            C = (X * conf[..., None]).sum(1)                        # sum of the reference's confident records
            Ci = C[:, None, :] - X * conf[..., None]                # ... excluding this record
            nn = np.linalg.norm(Ci, axis=2); dot = (X * Ci).sum(2)
            with np.errstate(all="ignore"): cent = np.where(nn > 1e-6, dot / nn, np.nan)
            ii = off + I[M]
            out["tt_maxany"][ii] = maxany[M]; out["tt_wmean"][ii] = wmean[M]; out["tt_n95"][ii] = n95[M]
            out["tt_maxconf"][ii] = maxconf[M]; out["tt_cent"][ii] = cent[M]
        L *= 2


def emb_features(b: pl.DataFrame, emb: EmbStore, chunk_rows=1_000_000) -> pl.DataFrame:
    """b sorted by s1_id (contiguous groups). Row chunks aligned to group boundaries; padded size-bucketed matmuls.
    Singleton groups / no confident others → NaN (tt_n95 = 0), as in the prototype."""
    n = len(b); p = b["p"].to_numpy().astype(np.float32)
    t_rows = emb.rows(b["t_id"]); s_rows = emb.rows(b["s1_id"])
    s1 = b["s1_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1]; sizes = np.diff(np.r_[starts, n]); del s1
    out = {k: np.full(n, np.nan, np.float32) for k in ("tt_maxconf", "tt_wmean", "tt_maxany", "tt_cent")}
    out["tt_n95"] = np.zeros(n, np.float32); cos_s1 = np.empty(n, np.float32)
    g0 = 0
    while g0 < len(starts):
        g1 = int(np.searchsorted(starts, starts[g0] + chunk_rows, side="right")); g1 = max(g1, g0 + 1)
        a, z = starts[g0], (starts[g1] if g1 < len(starts) else n)
        et = emb.gather(t_rows[a:z]); es = emb.gather(s_rows[a:z])
        cos_s1[a:z] = (et * es).sum(1); del es
        _emb_chunk(et, p[a:z], starts[g0:g1] - a, sizes[g0:g1], out, a); del et
        g0 = g1
    return b.with_columns(pl.Series("cos_s1", cos_s1), *[pl.Series(k, v) for k, v in out.items()]).with_columns(
        tt_cent_minus_s1=pl.col("tt_cent") - pl.col("cos_s1"), tt_maxconf_minus_s1=pl.col("tt_maxconf") - pl.col("cos_s1"))


def consensus_features(b: pl.DataFrame, views_path: str) -> pl.DataFrame:
    need = pl.concat([b["s1_id"], b["t_id"]]).unique()
    V = (pl.scan_parquet(views_path).select("entity_id", "name_core", "addr_core", pl.col("addr_numbers").list.first().alias("hn"))
         .filter(pl.col("entity_id").is_in(need.implode())).collect())
    V = V.with_columns([pl.when(pl.col(c).str.len_chars() > 0).then(pl.col(c)).alias(c) for c in ("name_core", "addr_core", "hn")])
    b = b.join(V.rename({"entity_id": "t_id", "name_core": "t_nc", "addr_core": "t_ac", "hn": "t_hn"}), on="t_id", how="left", maintain_order="left")
    b = b.join(V.rename({"entity_id": "s1_id", "name_core": "s_nc", "addr_core": "s_ac", "hn": "s_hn"}), on="s1_id", how="left", maintain_order="left")
    conf = b.filter(pl.col("p") > 0.9)
    for key, nm in (("t_nc", "tt_same_name"), ("t_hn", "tt_same_hn"), ("t_ac", "tt_same_addr")):
        c = conf.filter(pl.col(key).is_not_null()).group_by(["s1_id", key]).agg(pl.len().cast(pl.Int32).alias(nm))
        b = b.join(c, on=["s1_id", key], how="left", maintain_order="left").with_columns(
            pl.col(nm).fill_null(0) - ((pl.col("p") > 0.9) & pl.col(key).is_not_null()).cast(pl.Int32))
    b = b.with_columns(hn_eq_s1=(pl.col("t_hn") == pl.col("s_hn")).fill_null(False).cast(pl.Int8),
                       name_eq_s1=(pl.col("t_nc") == pl.col("s_nc")).fill_null(False).cast(pl.Int8))
    return b.drop("t_nc", "t_ac", "t_hn", "s_nc", "s_ac", "s_hn")


def join_stage1(b: pl.DataFrame, pairs_path: str, s1f: list, batch=4_000_000) -> pl.DataFrame:
    keys = b.select("s1_id", "t_id"); parts = []
    for rb in pq.ParquetFile(pairs_path).iter_batches(batch_size=batch, columns=["s1_id", "t_id"] + s1f):
        parts.append(pl.from_arrow(rb).join(keys, on=["s1_id", "t_id"], how="semi").with_columns([pl.col(c).cast(pl.Float32) for c in s1f]))
    f = pl.concat(parts)
    return b.join(f, on=["s1_id", "t_id"], how="left", maintain_order="left")


def build(sc: pl.DataFrame, countries, views_path, emb: EmbStore, log=print) -> pl.DataFrame:
    """All stage-2 features except the stage-1 pass-through (join_stage1). sc must hold complete candidate lists."""
    t0 = time.time()
    b = set_features(sc, countries); log(f"  set features: {len(b):,} band rows ({time.time()-t0:.0f}s)")
    b = consensus_features(b, views_path); log(f"  consensus ({time.time()-t0:.0f}s)")
    b = emb_features(b, emb); log(f"  e5 features ({time.time()-t0:.0f}s)")
    return b


def add_ce(b: pl.DataFrame, ce_path: str) -> pl.DataFrame:
    """Optional cross-encoder features: join p_ce (s1_id, t_id, p_ce) from ce_path onto band rows b (needs p, logit).
    Rows without a CE score (outside the stage-1 CE band) get p_ce = -1 (sentinel) and null for the derived features."""
    ce = pl.read_parquet(ce_path, columns=["s1_id", "t_id", "p_ce"]).unique(subset=["s1_id", "t_id"])
    b = b.drop([c for c in CE if c in b.columns]).join(ce.with_columns(pl.col("p_ce").cast(pl.Float64)), on=["s1_id", "t_id"], how="left", maintain_order="left")
    pc = pl.col("p_ce").clip(1e-6, 1 - 1e-6)
    return b.with_columns(ce_logit=(pc / (1 - pc)).log(), ce_minus_p=pl.col("p_ce") - pl.col("p")).with_columns(
        ce_logit_minus_logit=pl.col("ce_logit") - pl.col("logit"), p_ce=pl.col("p_ce").fill_null(-1.0))


def add_ce2(b: pl.DataFrame, ce_path: str) -> pl.DataFrame:
    """Optional second cross-encoder: join p_ce from ce_path as p_ce2 onto band rows b (needs p; uses p_ce if add_ce ran first).
    Rows without a score get p_ce2 = -1 (sentinel) and null derived features; ce2_minus_ce is null unless both CEs scored the row."""
    ce = pl.read_parquet(ce_path, columns=["s1_id", "t_id", "p_ce"]).unique(subset=["s1_id", "t_id"]).rename({"p_ce": "p_ce2"})
    b = b.drop([c for c in CE2 if c in b.columns]).join(ce.with_columns(pl.col("p_ce2").cast(pl.Float64)), on=["s1_id", "t_id"], how="left", maintain_order="left")
    pc = pl.col("p_ce2").clip(1e-6, 1 - 1e-6)
    d_ce = (pl.when(pl.col("p_ce") >= 0).then(pl.col("p_ce2") - pl.col("p_ce")) if "p_ce" in b.columns else pl.lit(None, dtype=pl.Float64))
    return b.with_columns(ce2_logit=(pc / (1 - pc)).log(), ce2_minus_p=pl.col("p_ce2") - pl.col("p"), ce2_minus_ce=d_ce).with_columns(
        p_ce2=pl.col("p_ce2").fill_null(-1.0))


def add_ce3(b: pl.DataFrame, ce_path: str) -> pl.DataFrame:
    """Optional third cross-encoder: join p_ce from ce_path as p_ce3 onto band rows b (needs p; uses p_ce if add_ce ran first).
    Mirrors add_ce2 exactly: rows without a score get p_ce3 = -1 (sentinel) and null derived features; ce3_minus_ce is null unless
    both the first CE (p_ce) and this CE scored the row."""
    ce = pl.read_parquet(ce_path, columns=["s1_id", "t_id", "p_ce"]).unique(subset=["s1_id", "t_id"]).rename({"p_ce": "p_ce3"})
    b = b.drop([c for c in CE3 if c in b.columns]).join(ce.with_columns(pl.col("p_ce3").cast(pl.Float64)), on=["s1_id", "t_id"], how="left", maintain_order="left")
    pc = pl.col("p_ce3").clip(1e-6, 1 - 1e-6)
    d_ce = (pl.when(pl.col("p_ce") >= 0).then(pl.col("p_ce3") - pl.col("p_ce")) if "p_ce" in b.columns else pl.lit(None, dtype=pl.Float64))
    return b.with_columns(ce3_logit=(pc / (1 - pc)).log(), ce3_minus_p=pl.col("p_ce3") - pl.col("p"), ce3_minus_ce=d_ce).with_columns(
        p_ce3=pl.col("p_ce3").fill_null(-1.0))


def ce_feats(k: int, template: str = "default") -> list:
    """Feature names of CE family k (1 = --ce, 2 = --ce2, 3 = --ce3) under a --ce-template."""
    assert template in CE_TEMPLATES, template
    if k == 1: return list(CE)
    return list({(2, "default"): CE2, (3, "default"): CE3, (2, "independent"): CE2_IND, (3, "independent"): CE3_IND}[(k, template)])


def add_ce_indep(b: pl.DataFrame, ce_path: str, k: int) -> pl.DataFrame:
    """CE family k (2 or 3) with the add_ce template, exact same expressions/clipping/sentinel: p_ce<k> (-1 if unscored),
    ce<k>_logit, ce<k>_minus_p = p_ce<k> - p, ce<k>_logit_minus_logit = ce<k>_logit - logit (null derived features if unscored)."""
    P, L, M, LL = ce_feats(k, "independent")
    ce = pl.read_parquet(ce_path, columns=["s1_id", "t_id", "p_ce"]).unique(subset=["s1_id", "t_id"]).rename({"p_ce": P})
    b = b.drop([c for c in dict.fromkeys(ce_feats(k, "default") + ce_feats(k, "independent")) if c in b.columns])
    b = b.join(ce.with_columns(pl.col(P).cast(pl.Float64)), on=["s1_id", "t_id"], how="left", maintain_order="left")
    pc = pl.col(P).clip(1e-6, 1 - 1e-6)
    return b.with_columns(**{L: (pc / (1 - pc)).log(), M: pl.col(P) - pl.col("p")}).with_columns(
        **{LL: pl.col(L) - pl.col("logit"), P: pl.col(P).fill_null(-1.0)})


def add_ce_family(b: pl.DataFrame, ce_path: str, k: int, template: str = "default") -> pl.DataFrame:
    """Dispatch: family 2/3 under --ce-template (default -> add_ce2/add_ce3, independent -> add_ce_indep)."""
    assert k in (2, 3) and template in CE_TEMPLATES, (k, template)
    if template == "independent": return add_ce_indep(b, ce_path, k)
    return add_ce2(b, ce_path) if k == 2 else add_ce3(b, ce_path)


def add_field_disagreement(b: pl.DataFrame) -> pl.DataFrame:
    """Joint-field features after the CE3 block (research/ASTRA_joint_field_v12_contract.json): field_signed = p_ce3 - p_ce2,
    field_abs = |p_ce3 - p_ce2|; null unless both p_ce2 >= 0 and p_ce3 >= 0 (-1 = unscored sentinel)."""
    assert "p_ce2" in b.columns and "p_ce3" in b.columns, "field disagreement needs the CE2 and CE3 blocks"
    diff = pl.when((pl.col("p_ce2") >= 0) & (pl.col("p_ce3") >= 0)).then(pl.col("p_ce3") - pl.col("p_ce2"))
    return b.drop([c for c in FIELD if c in b.columns]).with_columns(field_signed=diff, field_abs=diff.abs())


def extra_columns(path: str) -> list:
    """--extra-feats: the feature columns of a per-pair parquet (s1_id, t_id + numeric columns), in the file's column order."""
    cols = [c for c in pq.read_schema(path).names if c not in ("s1_id", "t_id")]
    assert cols, f"{path}: no feature columns besides s1_id, t_id"
    return cols


def add_extra(b: pl.DataFrame, path: str, cols: list = None) -> pl.DataFrame:
    """Optional extra per-pair features (e.g. research 43 E2 competition features): left-join the numeric columns of `path`
    on (s1_id, t_id) onto band rows b, cast to Float64; rows absent from the file get NaN (= missing for LightGBM) and
    _xf_hit = False (so callers can count coverage). Appended to FEATS after the CE/field blocks, in the file's column order."""
    cols = cols if cols is not None else extra_columns(path)
    names = pq.read_schema(path).names
    assert "s1_id" in names and "t_id" in names, f"{path}: needs s1_id, t_id"
    clash = set(cols) & set(b.columns); assert not clash, f"extra feature columns collide with band columns: {sorted(clash)}"
    x = pl.read_parquet(path, columns=["s1_id", "t_id"] + cols)
    assert x.select(pl.struct("s1_id", "t_id").is_duplicated().any()).item() is False, f"{path}: duplicate (s1_id, t_id) rows"
    x = x.select("s1_id", "t_id", *[pl.col(c).cast(pl.Float64) for c in cols]).with_columns(_xf_hit=pl.lit(True))
    b = b.drop("_xf_hit", strict=False).join(x, on=["s1_id", "t_id"], how="left", maintain_order="left")
    return b.with_columns(pl.col(cols).fill_null(float("nan")), pl.col("_xf_hit").fill_null(False))
