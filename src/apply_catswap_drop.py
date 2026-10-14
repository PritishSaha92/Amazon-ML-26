"""Post-hoc France category-word-swap drop (research/12_target_level_transductive.md, recommendation 1: LB probe).

p := min(p, --p-set) (default 0.05) for candidate pairs (s1_id = reference, t_id = target) where ALL hold:
  1. country in --country (default France; "ALL" = every country present in the table);
  2. hn_exact == 1 and shift_sig == 0 (exact house number, no shift signature: the address otherwise matches);
  3. the target's name_core differs from the reference's name_core by exactly ONE replaced word: equal word counts,
     exactly one distinct inserted word and one distinct deleted word (views: name_core; features n_ins == n_del == 1);
  4. the inserted word AND the deleted word are both in the country's "flat weak replacement" pool, derived from the
     word-LR table (data/tables/wordlr_{split}.json, "rep" table): rep LR in [--lr-lo, --lr-hi] (default 1.5-2.5)
     and pos+neg count >= --min-count (default 500). No hand-typed list; the pool is printed.
Run:
  python src/apply_catswap_drop.py --scores output/test_scores_v7ce_s2_acr.parquet --pairs data/pairs_test_v7_feat.parquet \
      --views data/views_test.parquet --table data/tables/wordlr_test.json --out output/test_scores_v7ce_s2_acr_nocat.parquet
  [--ins-only: require only the inserted word in the pool]  [--flag-out pairs.parquet: write the hit list]
Output has the same columns/rows/order as --scores (score column p_s2 if present else p).
Extension flags (research 14 / LOG 2026-09-26 "France swap-drop tiers"; defaults keep the original behaviour bit-for-bit):
  --exclude-variants   skip pairs whose replaced/inserted words look like the SAME word (plural/gender inflection,
                       truncation or subsequence abbreviation "st"->"saint", "frs"->"freres", Levenshtein <= 1 (<= 2 for
                       words >= 8 chars): true-noise edits, not a category swap)
  --addr-guard street  additionally require strong address agreement: identical addr_tokens set OR identical addr_core
                       token set (views), not just hn_exact
  Tiers used for the 2026-09-26 France extension (applied on top of output/test_scores_v7ce_s2_acr_nocat.parquet):
    T1  --lr-lo 1.2 --lr-hi 3.0 --min-count 100 --exclude-variants [--addr-guard street]
"""
import argparse, json, time
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--views", required=True)
ap.add_argument("--table", required=True); ap.add_argument("--out", default="")
ap.add_argument("--country", default="France", help='comma list or "ALL"')
ap.add_argument("--lr-lo", type=float, default=1.5); ap.add_argument("--lr-hi", type=float, default=2.5)
ap.add_argument("--min-count", type=int, default=500); ap.add_argument("--p-set", type=float, default=0.05)
ap.add_argument("--ins-only", action="store_true"); ap.add_argument("--flag-out", default="")
ap.add_argument("--exclude-variants", action="store_true"); ap.add_argument("--addr-guard", choices=["none", "street"], default="none")
a = ap.parse_args(); T0 = time.time()

def _lev(x, y):
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i]
        for j, cy in enumerate(y, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
        prev = cur
    return prev[-1]
def _stem(w):
    for suf in ("elles", "elle", "aux", "eux", "es", "s", "x", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3: return w[:-len(suf)]
    return w
def is_variant(u, w):
    """same word up to inflection / abbreviation / one-or-two-letter typo"""
    if _stem(u) == _stem(w): return True
    sh, lo = (u, w) if len(u) <= len(w) else (w, u)
    if len(sh) >= 3 and lo.startswith(sh): return True
    it = iter(lo)
    if len(sh) <= 4 and sh[:1] == lo[:1] and all(ch in it for ch in sh): return True
    if abs(len(u) - len(w)) > 2: return False
    return _lev(u, w) <= (1 if len(sh) < 8 else 2)

W = json.load(open(a.table))
ctys = [c for c in W if c != "_meta"] if a.country.upper() == "ALL" else a.country.split(",")
pool = {}
for c in ctys:
    rep = W.get(c, {}).get("rep", {})
    ws = sorted(((w, v) for w, v in rep.items() if a.lr_lo <= v[0] <= a.lr_hi and v[1] + v[2] >= a.min_count), key=lambda t: -(t[1][1] + t[1][2]))
    pool[c] = {w for w, _ in ws}
    print(f"pool {c} ({len(ws)} words; rep LR in [{a.lr_lo},{a.lr_hi}], count >= {a.min_count}): "
          + ", ".join(f"{w} {v[0]:.2f} ({v[1]}/{v[2]})" for w, v in ws))
pw = pl.DataFrame([(c, w) for c in pool for w in pool[c]], schema=["country", "w"], orient="row")

sc = pl.read_parquet(a.scores)
pcol = "p_s2" if "p_s2" in sc.columns else "p"
print(f"{a.scores}: {len(sc):,} rows, score column {pcol!r}")
fe = (pl.scan_parquet(a.pairs).filter(pl.col("country").is_in(list(pool)) & (pl.col("hn_exact") == 1) & (pl.col("shift_sig") == 0)
                                      & (pl.col("n_ins") == 1) & (pl.col("n_del") == 1))
      .select("s1_id", "t_id", "country").collect(engine="streaming"))
key = fe.join(sc.select("s1_id", "t_id"), on=["s1_id", "t_id"], how="semi")
print(f"pairs in scope (country, hn_exact, shift_sig 0, 1 ins + 1 del word, in --scores): {len(key):,}")
ids = pl.concat([key["s1_id"], key["t_id"]]).unique().implode()
v = pl.scan_parquet(a.views).select("entity_id", "name_core", "addr_core", "addr_tokens").filter(pl.col("entity_id").is_in(ids)).collect()
v = v.with_columns(pl.col("name_core").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("w"))
tok = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
v = v.with_columns(tok("addr_core").alias("ac"), tok("addr_tokens").alias("at"))
x = (key.join(v.select(pl.col("entity_id").alias("s1_id"), pl.col("w").alias("ws"), pl.col("name_core").alias("nc_s"), pl.col("ac").alias("ac_s"), pl.col("at").alias("at_s")), on="s1_id")
        .join(v.select(pl.col("entity_id").alias("t_id"), pl.col("w").alias("wt"), pl.col("name_core").alias("nc_t"), pl.col("ac").alias("ac_t"), pl.col("at").alias("at_t")), on="t_id"))
x = x.with_columns(((pl.col("at_s").list.set_symmetric_difference("at_t").list.len() == 0)
                    | (pl.col("ac_s").list.set_symmetric_difference("ac_t").list.len() == 0)).alias("addr_eq")).drop("ac_s", "ac_t", "at_s", "at_t")
x = x.with_columns(pl.col("wt").list.set_difference("ws").alias("ins"), pl.col("ws").list.set_difference("wt").alias("dele"))
x = x.filter((pl.col("ws").list.len() == pl.col("wt").list.len()) & (pl.col("ins").list.len() == 1) & (pl.col("dele").list.len() == 1))
x = x.with_columns(pl.col("ins").list.first().alias("w_ins"), pl.col("dele").list.first().alias("w_del")).drop("ins", "dele", "ws", "wt")
print(f"one-word replacements: {len(x):,}")
if a.exclude_variants:
    wp = x.select("w_ins", "w_del").unique()
    wp = wp.with_columns(pl.struct("w_ins", "w_del").map_elements(lambda r: is_variant(r["w_ins"], r["w_del"]), return_dtype=pl.Boolean).alias("_var"))
    x = x.join(wp, on=["w_ins", "w_del"], how="left"); print(f"  same-word variants excluded: {int(x['_var'].sum()):,}"); x = x.filter(~pl.col("_var")).drop("_var")
if a.addr_guard == "street":
    print(f"  address guard (addr_tokens or addr_core token set identical): keeps {int(x['addr_eq'].sum()):,} of {len(x):,}"); x = x.filter("addr_eq")
x = x.join(pw.rename({"w": "w_ins"}).with_columns(pl.lit(True).alias("ins_pool")), on=["country", "w_ins"], how="left")
x = x.join(pw.rename({"w": "w_del"}).with_columns(pl.lit(True).alias("del_pool")), on=["country", "w_del"], how="left")
x = x.with_columns(pl.col("ins_pool").fill_null(False), pl.col("del_pool").fill_null(False))
rule = x.filter(pl.col("ins_pool") & (True if a.ins_only else pl.col("del_pool")))
rule = rule.join(sc.select("s1_id", "t_id", pcol, *(["label"] if "label" in sc.columns else [])), on=["s1_id", "t_id"])
print(f"rule pairs ({'inserted word' if a.ins_only else 'inserted AND deleted word'} in pool): {len(rule):,}; "
      f"with p >= 0.5: {int((rule[pcol] >= 0.5).sum()):,}; lowered (p > {a.p_set}): {int((rule[pcol] > a.p_set).sum()):,}")
agg = [pl.len().alias("n"), (pl.col(pcol) >= 0.5).sum().alias("p_ge_0.5"), pl.col(pcol).mean().alias("mean_p")]
if "label" in rule.columns:
    agg += [pl.col("label").sum().alias("true"), (pl.col("label").cast(pl.Int64) * (pl.col(pcol) >= 0.5)).sum().alias("true_p_ge_0.5")]
    lab = rule
    print(f"labelled: rule pairs {len(lab)}, true {int(lab['label'].sum())}; p>=0.5: {int((lab[pcol] >= 0.5).sum())}, "
          f"true among them {int(lab.filter(pl.col(pcol) >= 0.5)['label'].sum())}")
pl.Config.set_tbl_rows(100)
print("per inserted word:"); print(rule.group_by("country", "w_ins").agg(agg).sort("n", descending=True))
print("per country:"); print(rule.group_by("country").agg(agg).sort("country"))
print("examples:"); print(rule.filter(pl.col(pcol) >= 0.5).select("nc_s", "nc_t", pcol).head(15))
if a.flag_out: rule.write_parquet(a.flag_out); print("wrote", a.flag_out)
if a.out:
    flag = rule.select("s1_id", "t_id").with_columns(pl.lit(True).alias("_cs"))
    out = sc.join(flag, on=["s1_id", "t_id"], how="left", maintain_order="left").with_columns(
        pl.when(pl.col("_cs")).then(pl.min_horizontal(pl.col(pcol), pl.lit(a.p_set)).cast(sc.schema[pcol])).otherwise(pl.col(pcol)).alias(pcol)).drop("_cs")
    assert len(out) == len(sc) and out.columns == sc.columns
    out.write_parquet(a.out); print(f"wrote {a.out} ({len(out):,} rows)")
print(f"done in {time.time() - T0:.0f}s")
