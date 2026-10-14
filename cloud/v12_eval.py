"""Fixed-decoder val/audit gate for a stage-2 run (same decoder as the champion comparison: best-reference ownership,
alpha 0.7, t 0.50; identical to stage2.decide / train_matcher.decide with margin 0).
Input: output/{val,audit}_scores_s2_<tag>.parquet (s1_id, t_id, p, p_s2) written by src/stage2.py.
Gate: >= +0.0001 on BOTH val and audit vs the v7ce champion stack (val 0.99031 / audit 0.99046).
Run: python cloud/v12_eval.py --tag v12ce [--ref-tag v7ce]
"""
import sys, argparse, os
import polars as pl
sys.path.insert(0, "src"); from scorer import slice_report
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--ref-tag", default="")
ap.add_argument("--t", type=float, default=0.50); ap.add_argument("--alpha", type=float, default=0.7)
ap.add_argument("--base-val", type=float, default=0.99031); ap.add_argument("--base-audit", type=float, default=0.99046)
ap.add_argument("--splits", default="val,audit", help="selection protocol: pass 'val' to withhold audit")
ap.add_argument("--min-gain", type=float, default=0.0001)
a = ap.parse_args()
truth_all = pl.read_parquet("data/train_gt.parquet"); split = pl.read_parquet("data/split.parquet"); cty = dict(zip(split["s1_id"], split["country"]))

def decide(x):
    x = x.sort(["t_id", "p"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("t_id").alias("r")).filter(pl.col("r") == 0)
    x = x.with_columns(pl.col("p").max().over("s1_id").alias("pmax")).filter((pl.col("p") >= a.t) & (pl.col("p") >= a.alpha * pl.col("pmax")))
    g = x.group_by("s1_id").agg(pl.col("t_id")); return dict(zip(g["s1_id"], (set(v) for v in g["t_id"])))

def score(tag, s):
    d = pl.read_parquet(f"output/{s}_scores_s2_{tag}.parquet", columns=["s1_id", "t_id", "p_s2"]).rename({"p_s2": "p"})
    ids = split.filter(pl.col("split") == s)["s1_id"]  # every val/audit S1, incl. ones without candidates (scored as empty)
    g = truth_all.filter(pl.col("source1_entity_id").is_in(ids.implode()))
    truth = {k: set(v.split(",")) if v else set() for k, v in zip(g["source1_entity_id"], g["matched_entity_ids"])}
    r = slice_report(decide(d), truth, cty); return r["ALL"]["macro_f05"], {k: round(v["macro_f05"], 5) for k, v in r.items() if k != "ALL"}

res, base = {}, {"val": a.base_val, "audit": a.base_audit}; base_src = "rounded defaults"
for s in [x.strip() for x in a.splits.split(",") if x.strip()]:
    f, per = score(a.tag, s); res[s] = f
    line = f"{a.tag} {s}: macro F0.5 {f:.5f}  per-slice {per}"
    if a.ref_tag and os.path.exists(f"output/{s}_scores_s2_{a.ref_tag}.parquet"):
        fr, _ = score(a.ref_tag, s); base[s] = fr; base_src = f"exact {a.ref_tag} artifacts"
        line += f"  | {a.ref_tag} {fr:.5f}  delta {f - fr:+.5f}"
    print(line)
if "audit" not in res:
    dv = res["val"] - base["val"]; print(f"VAL-ONLY (audit withheld; frozen policy ownership/alpha 0.7/t 0.50): val {res['val']:.5f} delta {dv:+.5f} vs {base['val']:.5f} [{base_src}]"); sys.exit(0)
dv, da = res["val"] - base["val"], res["audit"] - base["audit"]
ok = dv >= a.min_gain and da >= a.min_gain
print(f"GATE (>= +{a.min_gain} on both vs {base['val']:.5f}/{base['audit']:.5f} [{base_src}]): val {dv:+.5f} audit {da:+.5f} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 3)
