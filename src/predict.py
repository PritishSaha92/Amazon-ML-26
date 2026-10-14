"""Test inference: score test pairs with a trained matcher, apply the val-chosen decision policy, and write
output/matching_results.tsv + output/candidate_pairs.tsv, then run the official validator.
Run: python src/predict.py --pairs data/pairs_test_feat.parquet --model models/lgbm_<tag> [--t 0.65] [--tag name]
"""
import argparse, json, os, sys, subprocess
import numpy as np, polars as pl, lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True); ap.add_argument("--model", required=True, help="path prefix without extension")
ap.add_argument("--t", type=float, default=None, help="override threshold"); ap.add_argument("--alpha", type=float, default=None)
ap.add_argument("--margin", type=float, default=0.0); ap.add_argument("--tag", default="")
ap.add_argument("--country-t", default="", help='per-country threshold overrides, e.g. "France:0.7"')
ap.add_argument("--country-alpha", default="", help='per-country alpha overrides, e.g. "France:0.55"')
ap.add_argument("--blend-ce", default="", help="parquet with s1_id,t_id,p_ce (cross-encoder scores for the ambiguous band); p := (1-w)*p + w*p_ce on those rows")
ap.add_argument("--w", type=float, default=0.5)
ap.add_argument("--scores-from", default="", help="reuse saved per-pair scores parquet (skips model scoring)")
a = ap.parse_args()
cfg = json.load(open(a.model + ".json"))
# packaged copy: the booster file is only needed when scoring features; with --scores-from only the policy JSON is read
model = None if a.scores_from else lgb.Booster(model_file=a.model + ".txt")
feat_cols = cfg["features"]; cty_map = {c: i for i, c in enumerate(cfg["countries"])}
t = a.t if a.t is not None else cfg["t"]; alpha = a.alpha if a.alpha is not None else cfg["alpha"]; own = cfg["ownership"]
cty_t = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in a.country_t.split(",") if kv}
cty_a = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in a.country_alpha.split(",") if kv}
print(f"policy: ownership={own} alpha={alpha} t={t} margin={a.margin} country_t={cty_t}" + (f" country_alpha={cty_a}" if cty_a else ""))

import pyarrow.parquet as pq
df = pl.read_parquet(a.pairs, columns=["s1_id", "t_id"])  # candidate list (ids only; features are streamed below)
os.makedirs("output", exist_ok=True); sfx = f"_{a.tag}" if a.tag else ""
if a.scores_from:
    x = pl.read_parquet(a.scores_from); print("loaded scores", len(x))
else:  # stream the feature file in row batches so a 150M-row file fits in RAM
    pf = pq.ParquetFile(a.pairs); parts = []
    for i, b in enumerate(pf.iter_batches(batch_size=4_000_000, columns=["s1_id", "t_id", "country"] + feat_cols)):
        d = pl.from_arrow(b)
        X = d.select(feat_cols).to_numpy().astype(np.float32)
        # unseen countries (France) get -1 → LightGBM routes them like a missing categorical value
        cty = d["country"].replace_strict(cty_map, default=-1).to_numpy().astype(np.float32).reshape(-1, 1)
        p = model.predict(np.hstack([X, cty]), num_iteration=cfg.get("best_iteration")); del X, cty
        parts.append(d.select("s1_id", "t_id", "country").with_columns(pl.Series("p", p.astype(np.float32))))
        print(f"  batch {i}: {len(d):,} rows", flush=True)
    x = pl.concat(parts); del parts
    x.write_parquet(f"output/test_scores{sfx}.parquet")  # per-pair scores: reused for cross-encoder blending / threshold variants
    print("scored", len(x), "pairs; mean p", float(x["p"].mean()))
if a.blend_ce:
    ce = pl.read_parquet(a.blend_ce).select("s1_id", "t_id", "p_ce")
    x = x.join(ce, on=["s1_id", "t_id"], how="left").with_columns(
        pl.when(pl.col("p_ce").is_not_null()).then((1 - a.w) * pl.col("p") + a.w * pl.col("p_ce")).otherwise(pl.col("p")).alias("p")).drop("p_ce")
    print("blended cross-encoder on", len(ce), "band pairs")
if own:
    x = x.sort(["t_id", "p"], descending=[False, True]).with_columns(
        pl.col("p").shift(-1).over("t_id").fill_null(0.0).alias("p2"), pl.int_range(pl.len()).over("t_id").alias("r"))
    x = x.filter((pl.col("r") == 0) & ((pl.col("p") - pl.col("p2")) >= a.margin))
x = x.with_columns(pl.col("p").max().over("s1_id").alias("pmax"),
                   pl.col("country").replace_strict(cty_t, default=t, return_dtype=pl.Float64).alias("t_c"))
a_c = pl.col("country").replace_strict(cty_a, default=alpha, return_dtype=x.schema["pmax"]) if cty_a else alpha  # pmax dtype: same arithmetic as the scalar path
sel = x.filter((pl.col("p") >= pl.col("t_c")) & (pl.col("p") >= a_c * pl.col("pmax")))

s1 = pl.read_parquet("data/test_records.parquet").filter(pl.col("source") == 1).select("entity_id", "country")
m = sel.group_by("s1_id").agg(pl.col("t_id").sort().str.join(",").alias("ids"))
c = df.group_by("s1_id").agg(pl.col("t_id").unique().sort().str.join(",").alias("ids"))
for name, g in (("matching_results", m), ("candidate_pairs", c)):
    out = s1.join(g, left_on="entity_id", right_on="s1_id", how="left").with_columns(pl.col("ids").fill_null(""))
    col = "matched_entity_ids" if name == "matching_results" else "candidate_entity_ids"
    out.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("ids").alias(col)).write_csv(f"output/{name}{sfx}.tsv", separator="\t", quote_style="never")
# diagnostics per country: matches per entity, share empty, share of targets assigned (train reference: 3.5, 5.6%, 73–75%)
diag = s1.join(sel.group_by("s1_id").len(), left_on="entity_id", right_on="s1_id", how="left").with_columns(pl.col("len").fill_null(0))
tgt = pl.read_parquet("data/test_records.parquet").filter(pl.col("source") != 1).group_by("country").len().rename({"len": "n_tgt"})
rep = diag.group_by("country").agg(pl.len().alias("n_s1"), pl.col("len").mean().alias("match_per_s1"), (pl.col("len") == 0).mean().alias("empty_share"),
                                   pl.col("len").sum().alias("n_assigned")).join(tgt, on="country").with_columns((pl.col("n_assigned") / pl.col("n_tgt")).alias("tgt_assigned_share")).sort("country")
print(rep)
r = subprocess.run([sys.executable, "student_resource/utils/validate_submission.py", "--matching", f"output/matching_results{sfx}.tsv",
                    "--candidate", f"output/candidate_pairs{sfx}.tsv", "--test-dir", "student_resource/dataset/test", "--check-ids"],
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:])
if r.returncode != 0:
    print(f"FAIL: official validator returned {r.returncode} for output/matching_results{sfx}.tsv / output/candidate_pairs{sfx}.tsv", flush=True)
    sys.exit(2)   # propagate: a chain must never freeze a file that did not PASS
