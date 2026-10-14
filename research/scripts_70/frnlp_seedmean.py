"""Seed-mean of count-aware CE files (arithmetic mean of p_ce over the seeds present). Writes
$BER_SCRATCH/frnlp/FRNLPm_oof.parquet, FRNLPm_val.parquet, frnlpm_test_all.parquet (+ .seeds.json listing seeds and input md5s).
Usage: python frnlp_seedmean.py 0 1 2   (seed 0 = FRNLP_oof / frnlp_test_all; seed s = FRNLP{s}_oof / frnlp{s}_test_all)"""
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, json, hashlib
import polars as pl
O = (_BER_S + "/frnlp"); K = ["s1_id", "t_id"]; seeds = sys.argv[1:]
md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
def f_oof(s): return f"{O}/FRNLP_oof.parquet" if s == "0" else f"{O}/FRNLP{s}_oof.parquet"
def f_test(s): return f"{O}/frnlp_test_all.parquet" if s == "0" else f"{O}/frnlp{s}_test_all.parquet"
meta = dict(seeds=seeds, inputs={})
for kind, fn, out in (("oof", f_oof, "FRNLPm_oof"), ("test", f_test, "frnlpm_test_all")):
    if kind == "oof" and not all(__import__("os").path.exists(fn(s)) for s in seeds): continue  # packaged copy: OOF files optional (test-only run)
    base = None
    for s in seeds:
        f = fn(s); meta["inputs"][f] = md5(f)
        x = pl.read_parquet(f, columns=K + ["p_ce"] + (["country"] if kind == "test" else [])).rename({"p_ce": f"p_s{s}"})
        base = x if base is None else base.join(x.drop("country", strict=False), on=K, how="inner")
        assert len(base) == len(x), f"key mismatch {f}"
    base = base.with_columns(p_ce=pl.mean_horizontal([f"p_s{s}" for s in seeds]).cast(pl.Float32))
    cols = K + ["p_ce"] + [f"p_s{s}" for s in seeds] + (["country"] if kind == "test" else [])
    base.select(cols).write_parquet(f"{O}/{out}.parquet"); meta[out] = dict(rows=len(base), md5=md5(f"{O}/{out}.parquet"))
    if kind == "oof":
        vb = pl.scan_parquet("data/s2_band_v12avg.parquet").filter(pl.col("split") == "val").select(K).collect()
        v = base.select(K + ["p_ce"]).join(vb, on=K, how="semi"); v.write_parquet(f"{O}/FRNLPm_val.parquet")
        meta["FRNLPm_val"] = dict(rows=len(v), md5=md5(f"{O}/FRNLPm_val.parquet"))
json.dump(meta, open(f"{O}/FRNLPm.seeds.json", "w"), indent=1); print(json.dumps(meta, indent=1))
