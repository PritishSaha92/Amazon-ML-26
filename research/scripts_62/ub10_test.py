"""fr12d UB10 on TEST (frozen: drop an accepted pair when stage-1 p1 > 0.98 and mean of the 4 X2 members < 0.10; drop only).
Base = submissions/v12fr12x2_E2_frENS_s (md5 2cf2e79d...). Cascade: member 1 = fr12_ceA on all upper-band accepted pairs; members 2-4
only where member 1 < 0.40 (a 4-member mean < 0.10 needs member 1 < 0.40); others 'not scored' and never dropped.
  keys    -> $BER_SCRATCH/fr12d/ub/test_keys.parquet
  cascade -> test_stage2.parquet (member-1 < 0.40)
  apply   -> drop list, per-country report, entries submissions/v12fr12x2_E2_UB10all_s and submissions/v12fr12x2_E2_frUB10_s (TSV rows rewritten
             in place: only the matched_entity_ids field of changed rows is edited; row order, header and all other rows byte-identical)"""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, csv, polars as pl
os.chdir(_BER_R); mode = sys.argv[1]; U = (_BER_S + "/fr12d/ub"); K = ["s1_id", "t_id"]
BASE = "submissions/v12fr12x2_E2_frENS_s/matching_results.tsv"
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
def acc():
    d = pl.read_csv(BASE, separator="\t", quote_char=None, null_values=[""], schema_overrides={"matched_entity_ids": pl.String}).fill_null("")
    d = d.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"})
    return d, d.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1_id", pl.col("m").alias("t_id"))
if mode == "keys":
    _, a = acc()
    p1 = pl.scan_parquet("output/test_scores_v12avg.parquet").select(K + ["p"]).filter(pl.col("p") > 0.98).join(a.lazy(), on=K, how="semi").collect(engine="streaming")
    p1.join(cty, on="s1_id").write_parquet(f"{U}/test_keys.parquet")
    print("accepted pairs", len(a), "upper-band accepted", len(p1)); print(p1.join(cty, on="s1_id").group_by("country").len()); sys.exit()
if mode == "cascade":
    s = pl.read_parquet(f"{U}/test_m1.parquet", columns=K + ["p_ce"])
    s.filter(pl.col("p_ce") < 0.40).select(K).write_parquet(f"{U}/test_stage2.parquet"); print("member-1", len(s), "need members 2-4", int((s["p_ce"] < 0.40).sum())); sys.exit()
keys = pl.read_parquet(f"{U}/test_keys.parquet")
def chk(df, keyset, name):
    assert df.select(K).is_duplicated().sum() == 0, f"{name}: duplicate keys"
    v = df["p_ce"]; assert v.null_count() == 0 and v.is_nan().sum() == 0 and float(v.min()) >= 0 and float(v.max()) <= 1, f"{name}: bad values"
    assert len(df) == len(keyset) and df.select(K).join(keyset, on=K, how="anti").height == 0, f"{name}: key set mismatch ({len(df)} vs {len(keyset)})"
_m1 = pl.read_parquet(f"{U}/test_m1.parquet", columns=K + ["p_ce"]); chk(_m1, keys.select(K), "m1")
_st = _m1.filter(pl.col("p_ce") < 0.40).select(K)
for m in ("fr12_ceB", "fr12s1_ceA", "fr12s1_ceB"): chk(pl.read_parquet(f"{U}/test_{m}.parquet", columns=K + ["p_ce"]), _st, m)
print(f"keys: upper-band accepted {len(keys):,} = member-1 {len(_m1):,}; members 2-4 each exactly {len(_st):,} (member 1 < 0.40)")
s = keys.join(pl.read_parquet(f"{U}/test_m1.parquet", columns=K + ["p_ce"]).rename({"p_ce": "fr12_ceA"}), on=K, how="left")
for m in ("fr12_ceB", "fr12s1_ceA", "fr12s1_ceB"): s = s.join(pl.read_parquet(f"{U}/test_{m}.parquet", columns=K + ["p_ce"]).rename({"p_ce": m}), on=K, how="left")
assert s["fr12_ceA"].null_count() == 0
s = s.with_columns(x2=((pl.col("fr12_ceA") + pl.col("fr12_ceB")) / 2 + (pl.col("fr12s1_ceA") + pl.col("fr12s1_ceB")) / 2) / 2)
st2 = s.filter(pl.col("fr12_ceA") < 0.40); assert st2["x2"].null_count() == 0, "stage-2 members missing"
s.write_parquet(f"{U}/test_scored.parquet")
drop = s.filter(pl.col("x2").is_not_null() & (pl.col("x2") < 0.10)).select(K + ["country"]); drop.write_parquet(f"{U}/test_dropped.parquet")
d, a = acc(); dd = drop.group_by("s1_id").agg(pl.col("t_id"))
# keep original order of remaining ids
nd = dict(zip(dd["s1_id"], (set(v) for v in dd["t_id"])))
def fix(s1, m): return ",".join(t for t in m.split(",") if t and t not in nd[s1]) if s1 in nd else m
for c in ("France", "India", "US"):
    ku = keys.filter(pl.col("country") == c); kd = drop.filter(pl.col("country") == c)
    refs = kd["s1_id"].unique().to_list(); emptied = sum(1 for r in refs if fix(r, d.filter(pl.col("s1_id") == r)["m"][0]) == "")
    print(f"TEST {c}: upper-band accepted {len(ku):,}; member-1 >= 0.40 (not scored further) {int((s.filter(pl.col('country') == c)['fr12_ceA'] >= 0.40).sum()):,}; "
          f"dropped {len(kd):,}; refs changed {len(refs):,}; refs emptied {emptied:,}; drop RATE {len(kd)/max(len(ku),1)*100:.4f}%")
cmap = dict(zip(cty["s1_id"], cty["country"]))
def write(out, countries):
    os.makedirs(os.path.dirname(out), exist_ok=True); assert not os.path.exists(out)
    n = 0
    with open(BASE, "rb") as fi, open(out, "wb") as fo:
        fo.write(fi.readline())
        for line in fi:
            txt = line.decode("utf-8"); s1, m = txt.rstrip("\r\n").split("\t")
            if s1 in nd and cmap.get(s1) in countries:
                m2 = fix(s1, m); fo.write(f"{s1}\t{m2}\n".encode()); n += m2 != m
            else: fo.write(line)
    print(out, "rows changed", n)
write("submissions/v12fr12x2_E2_UB10all_s/matching_results.tsv", {"France", "India", "US"})
write("submissions/v12fr12x2_E2_frUB10_s/matching_results.tsv", {"France"})
write("submissions/v12fr12x2_E2_UB10ui_s/matching_results.tsv", {"India", "US"})
