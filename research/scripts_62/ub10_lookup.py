"""fr12d UB10 lookup for TEST: one file keyed by (s1_id, t_id) with p1, the 4 X2 member scores (null = 'not scored': cascade skipped
because member 1 fr12_ceA >= 0.40, which makes a 4-member mean < 0.10 impossible), x2 mean (null when not fully scored), and ub10 flag.
  extend <submission.tsv> [...] : find accepted upper-band (p1 > 0.98) pairs of those files missing from the lookup -> new_keys.parquet
  merge                         : fold newly scored member files into the lookup (strict key assertions)
Lookup: $BER_SCRATCH/fr12d/ub/ub10_lookup_test.parquet. Consumers: apply UB10 to THEIR OWN final accepted upper-band pairs and fail if any
such pair is absent from the lookup (research/scripts_62/ub10_apply.py)."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, polars as pl
os.chdir(_BER_R); U = (_BER_S + "/fr12d/ub"); K = ["s1_id", "t_id"]; L = f"{U}/ub10_lookup_test.parquet"
M = ["fr12_ceA", "fr12_ceB", "fr12s1_ceA", "fr12s1_ceB"]
def finish(s):
    s = s.with_columns(x2=((pl.col("fr12_ceA") + pl.col("fr12_ceB")) / 2 + (pl.col("fr12s1_ceA") + pl.col("fr12s1_ceB")) / 2) / 2)
    return s.with_columns(ub10=pl.col("x2").is_not_null() & (pl.col("x2") < 0.10), scored="fully" if False else pl.when(pl.col("fr12_ceA") >= 0.40).then(pl.lit("member1_only")).otherwise(pl.lit("all4")))
def chk(s):
    assert s.select(K).is_duplicated().sum() == 0 and s["fr12_ceA"].null_count() == 0
    bad = s.filter((pl.col("fr12_ceA") < 0.40) & pl.col("x2").is_null()); assert bad.height == 0, f"{bad.height} pairs with member1 < 0.40 lack members 2-4"
    ok = s.filter(pl.col("fr12_ceA") >= 0.40); assert all(ok[m].null_count() == ok.height for m in M[1:]), "skipped pairs must be null"
mode = sys.argv[1]
if not os.path.exists(L):   # seed from the base scoring
    s = pl.read_parquet(f"{U}/test_scored.parquet").rename({"p": "p1"}).select(K + ["country", "p1"] + M)
    s = finish(s); chk(s); s.write_parquet(L); print("seeded lookup", len(s))
lk = pl.read_parquet(L)
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
if mode == "extend":
    acc = []
    for f in sys.argv[2:]:
        d = pl.read_csv(f, separator="\t", quote_char=None, null_values=[""], schema_overrides={"matched_entity_ids": pl.String}).fill_null("")
        acc.append(d.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"}).with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1_id", pl.col("m").alias("t_id")))
    a = pl.concat(acc).unique()
    p1 = pl.scan_parquet("output/test_scores_v12avg.parquet").select(K + ["p"]).filter(pl.col("p") > 0.98).join(a.lazy(), on=K, how="semi").collect(engine="streaming")
    new = p1.join(lk.select(K), on=K, how="anti").join(cty, on="s1_id").rename({"p": "p1"})
    new.write_parquet(f"{U}/new_keys.parquet"); print("accepted upper-band in given files", len(p1), "missing from lookup", len(new), dict(new.group_by("country").len().rows()))
elif mode == "merge":
    new = pl.read_parquet(f"{U}/new_keys.parquet")
    m1 = pl.read_parquet(f"{U}/new_fr12_ceA.parquet", columns=K + ["p_ce"]); assert len(m1) == len(new) and m1.join(new, on=K, how="anti").height == 0
    s = new.join(m1.rename({"p_ce": "fr12_ceA"}), on=K)
    need = s.filter(pl.col("fr12_ceA") < 0.40).select(K)
    for m in M[1:]:
        if need.height:
            x = pl.read_parquet(f"{U}/new_{m}.parquet", columns=K + ["p_ce"]); assert len(x) == len(need) and x.join(need, on=K, how="anti").height == 0, m
            s = s.join(x.rename({"p_ce": m}), on=K, how="left")
        else: s = s.with_columns(pl.lit(None, pl.Float32).alias(m))
    s = finish(s.select(K + ["country", "p1"] + M)); chk(s)
    out = pl.concat([lk, s.select(lk.columns)], how="vertical_relaxed"); chk(out); out.write_parquet(L)
    print("merged", len(s), "new; lookup", len(out), "ub10 flags", int(out["ub10"].sum()))
