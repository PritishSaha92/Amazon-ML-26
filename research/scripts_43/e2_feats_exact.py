"""EXACT producer-consistent E2 features from the pmin-0 node rescore (astra audit, research/ASTRA_R43_E2_AUDIT.md), with 4 assertions:
 A1 fold assignment: local rng(0).permutation(n)%5 over sorted unique train s1_id of the OOF file; every fold-part rival row must lie
    outside its fold k and every fold-k query target must be present → verified through A2's exact key-set equality.
 A2 override key sets: part=fold,k == {(r,t) in OOF : t in targets of fold-k TRAIN band rows, r not a fold-k train ref} (all splits);
    part=full == {(r,t) in OOF : r train, t in targets of VAL/AUDIT band rows}; zero missing / extra / duplicate keys.
 A3 every producer column (p_v12, p_v12b) finite on every override row (no nan-ignoring mean).
 A4 md5 of inputs/code logged; fresh output files (no reuse of any earlier feature cache); tmp dir wiped first.
--cutoff c (sensitivity): universe for fold k = fold-k train rows + override rows whose OLD p >= c only (others dropped); val/audit = val/audit rows + full rows with old p >= c."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, argparse, hashlib, shutil
sys.path.insert(0, (_BER_R + "/research/scripts_43"))
from common43 import *
from comp43 import target_aggs, comp_features, F
ap = argparse.ArgumentParser(); ap.add_argument("--universe", default="data/oof_scores_v12avg.parquet"); ap.add_argument("--band", default="data/s2_band_v12avg.parquet")
ap.add_argument("--rescore", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cutoff", type=float, default=-1.0)
ap.add_argument("--tmp", default=(_BER_S + "/r43/tmp_exact")); ap.add_argument("--a2-pmin", type=float, default=0.0, help="expected key sets use OOF p >= this (0 = the pmin-0 rerun)")
a = ap.parse_args(); t0 = time.time()
md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
log("A4 md5:", {f: md5(f) for f in (a.rescore, a.universe, a.band, __file__, "research/scripts_43/comp43.py", "research/scripts_43/node_producer_rescore.py")})
shutil.rmtree(a.tmp, ignore_errors=True); os.makedirs(a.tmp); assert not os.path.exists(a.out), f"A4: {a.out} exists — refusing to reuse a cache"
U = pl.read_parquet(a.universe, columns=K + ["split", "p"])
u = U.filter(pl.col("split") == "train")["s1_id"].unique().sort()
folds = pl.DataFrame({"s1_id": u, "f": (np.random.default_rng(0).permutation(len(u)) % 5).astype(np.int8)})
U = U.join(folds, on="s1_id", how="left")
band = pl.read_parquet(a.band, columns=K + ["split", "p"]).join(folds, on="s1_id", how="left")
R = pl.read_parquet(a.rescore); pc = [c for c in R.columns if c.startswith("p_")]
log(f"rescore {len(R):,} rows, producer columns {pc}; parts", R.group_by("part", "k").agg(pl.len()).sort("part", "k").rows())
# A3
nonfin = {c: int((~R[c].cast(pl.Float64).is_finite()).sum() + R[c].null_count()) for c in pc}
log(f"A3 non-finite/null per producer column: {nonfin} -> {'PASS' if sum(nonfin.values()) == 0 and len(pc) == 2 else 'FAIL'}"); assert sum(nonfin.values()) == 0 and len(pc) == 2
R = R.with_columns(((pl.col(pc[0]) + pl.col(pc[1])) / 2).cast(pl.Float32).alias("pn")).select("part", "k", *K, "pn")
# A2 (+A1)
ok = True
for k in list(range(5)) + [-1]:
    if k >= 0:
        Tk = band.filter((pl.col("split") == "train") & (pl.col("f") == k)).select("t_id").unique()
        exp = U.join(Tk, on="t_id", how="semi").filter(~((pl.col("split") == "train") & (pl.col("f") == k)) & (pl.col("p") >= a.a2_pmin)).select(K)
        got = R.filter((pl.col("part") == "fold") & (pl.col("k") == k)).select(K)
    else:
        Tv = band.filter(pl.col("split") != "train").select("t_id").unique()
        exp = U.join(Tv, on="t_id", how="semi").filter((pl.col("split") == "train") & (pl.col("p") >= a.a2_pmin)).select(K)
        got = R.filter(pl.col("part") == "full").select(K)
    miss = exp.join(got, on=K, how="anti").height; extra = got.join(exp, on=K, how="anti").height; dup = got.is_duplicated().sum()
    log(f"A2 part {'fold k=%d' % k if k >= 0 else 'full'}: expected {exp.height:,} got {got.height:,} missing {miss} extra {extra} dup {dup}")
    ok &= (miss == 0 and extra == 0 and dup == 0)
log(f"A2 (and A1 via fold-exact key sets) -> {'PASS' if ok else 'FAIL'}"); assert ok
old = U.select(K + [pl.col("p").alias("p_old")])
def universe_for(over, base_filter):
    """base rows (kept at their existing p) + override rows (pn); with a cutoff, override rows need old p >= cutoff."""
    ov = over.join(old, on=K, how="left")
    if a.cutoff >= 0: ov = ov.filter(pl.col("p_old") >= a.cutoff)
    x = pl.concat([U.filter(base_filter).select("t_id", "p"), ov.select("t_id", pl.col("pn").alias("p"))]) if a.cutoff >= 0 else \
        U.join(over, on=K, how="left").with_columns(pl.coalesce("pn", "p").alias("p")).select("t_id", "p")
    x.write_parquet(f"{a.tmp}/u.parquet"); return target_aggs(f"{a.tmp}/u.parquet", log=log)
outs = []
for k in range(5):
    g, hi = universe_for(R.filter((pl.col("part") == "fold") & (pl.col("k") == k)).select(K + ["pn"]), (pl.col("split") == "train") & (pl.col("f") == k))
    q = band.filter((pl.col("split") == "train") & (pl.col("f") == k)).select(K + ["p"]); outs.append(comp_features(q, g, hi)); log(f"fold {k}: {len(q):,} queries")
g, hi = universe_for(R.filter(pl.col("part") == "full").select(K + ["pn"]), pl.col("split") != "train")
q = band.filter(pl.col("split") != "train").select(K + ["p"]); outs.append(comp_features(q, g, hi))
o = band.select(K).join(pl.concat(outs), on=K, how="left", maintain_order="left")
assert o["c_best"].null_count() == 0 and o.height == band.height
o.select(K + [pl.col(f).cast(pl.Float32) for f in ("c_best", "c_margin", "c_n50", "c_rank")]).write_parquet(a.out); log(f"wrote {a.out} md5 {md5(a.out)} ({time.time()-t0:.0f}s)")
print(o.with_columns(band["split"]).group_by("split").agg(pl.len(), (pl.col("c_best") >= .5).mean().alias("cbest>=.5"), pl.col("c_ncand").mean(), pl.col("c_margin").mean()).sort("split"))
