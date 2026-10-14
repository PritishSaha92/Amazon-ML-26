"""Post-hoc acronym rule on saved per-pair scores (research/10_residual_errors.md, intervention 1).

p := max(p, P_SET) for candidate pairs (s1_id = reference, t_id = target record) where ALL hold:
  1. acronym: the target's compact name (name_core, letters only, lowercased, >= 2 letters) equals the initials of the
     reference's name_core words, OR the target's compact name_words equals the initials of the reference's name_words
     (views parquet: name_core already drops stopwords / legal forms);
  2. it is the only acronym-matching candidate reference for that target record (among all candidate pairs in --scores);
  3. hn_exact (house number exact match);  4. shift_sig < 2 (no house-number shift signature);
  5. current p >= --p-min (default 0.001, the research agent's population filter; the only acronym-rule false positives
     on val/audit — 4 pairs — had p <= 2e-5: different addresses sharing a number, e.g. "gns services" -> "gs").
Restricted to --country (default France; "ALL" = every country).
Run:
  python src/apply_acronym_rule.py --scores output/test_scores_v7ce_s2.parquet --pairs data/pairs_test_v7_feat.parquet \
      --views data/views_test.parquet --out output/test_scores_v7ce_s2_acr.parquet [--country France|ALL|France,US]
Output has the same columns/rows/order as --scores (the score column, p_s2 if present else p, is updated).
"""
import argparse, time
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--views", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--country", default="France", help='comma list or "ALL"')
ap.add_argument("--p-set", type=float, default=0.95); ap.add_argument("--p-min", type=float, default=0.001)
a = ap.parse_args(); T0 = time.time()

sc = pl.read_parquet(a.scores)
pcol = "p_s2" if "p_s2" in sc.columns else "p"
print(f"{a.scores}: {len(sc):,} rows, score column {pcol!r}")
fe = pl.scan_parquet(a.pairs).select("s1_id", "t_id", "country", "shift_sig", "hn_exact", "t_addr_empty")
if a.country.upper() != "ALL":
    fe = fe.filter(pl.col("country").is_in(a.country.split(",")))
key = sc.select("s1_id", "t_id").lazy().join(fe, on=["s1_id", "t_id"], how="inner").collect()
print(f"candidate pairs in scope ({a.country}): {len(key):,}")

v = pl.read_parquet(a.views, columns=["entity_id", "name_core", "name_words"])
letters = lambda c: c.fill_null("").str.to_lowercase().str.replace_all(r"[^a-z]", "")
initials = lambda c: (c.fill_null("").str.to_lowercase().str.split(" ")
                      .list.eval(pl.element().str.slice(0, 1)).list.join("").str.replace_all(r"[^a-z]", ""))
rv = v.select(pl.col("entity_id").alias("s1_id"), initials(pl.col("name_core")).alias("ini_c"), initials(pl.col("name_words")).alias("ini_w"))
tv = v.select(pl.col("entity_id").alias("t_id"), letters(pl.col("name_core")).alias("tc"), letters(pl.col("name_words")).alias("tw"))
x = key.join(rv, on="s1_id", how="left").join(tv, on="t_id", how="left")
x = x.with_columns((((pl.col("tc").str.len_chars() >= 2) & (pl.col("tc") == pl.col("ini_c")))
                    | ((pl.col("tw").str.len_chars() >= 2) & (pl.col("tw") == pl.col("ini_w")))).fill_null(False).alias("acro"))
x = x.with_columns(pl.col("acro").sum().over("t_id").alias("nacro"))
rule = x.filter(pl.col("acro") & (pl.col("nacro") == 1) & (pl.col("hn_exact") > 0) & (pl.col("shift_sig") < 2))
rule = rule.select("s1_id", "t_id", "country", "t_addr_empty").join(sc.select("s1_id", "t_id", pcol), on=["s1_id", "t_id"])
rule = rule.filter(pl.col(pcol) >= a.p_min)
print(f"acronym pairs {int(x['acro'].sum()):,}; unique-acronym targets {int((x['acro'] & (x['nacro'] == 1)).sum()):,}; "
      f"rule pairs (unique + hn_exact + no shift + p >= {a.p_min}) {len(rule):,}")
rep = rule.group_by("country").agg(pl.len().alias("matched"), (pl.col(pcol) < a.p_set).sum().alias("raised"),
                                   ((pl.col(pcol) < 0.5)).sum().alias("raised_from_below_0.5"),
                                   pl.col("t_addr_empty").mean().alias("t_addr_empty_share"), pl.col(pcol).mean().alias("mean_p_before")).sort("country")
print(rep)
if "label" in sc.columns:
    lab = rule.join(sc.select("s1_id", "t_id", "label"), on=["s1_id", "t_id"])
    print("labelled: rule pairs", len(lab), "true", int(lab["label"].sum()), "| raised pairs", int((lab[pcol] < a.p_set).sum()),
          "true among raised", int(lab.filter(pl.col(pcol) < a.p_set)["label"].sum()))
flag = rule.select("s1_id", "t_id").with_columns(pl.lit(True).alias("_acr"))
out = sc.join(flag, on=["s1_id", "t_id"], how="left", maintain_order="left").with_columns(
    pl.when(pl.col("_acr")).then(pl.max_horizontal(pl.col(pcol), pl.lit(a.p_set)).cast(sc.schema[pcol])).otherwise(pl.col(pcol)).alias(pcol)).drop("_acr")
assert len(out) == len(sc) and out.columns == sc.columns
out.write_parquet(a.out); print(f"wrote {a.out} ({len(out):,} rows) in {time.time() - T0:.0f}s")
