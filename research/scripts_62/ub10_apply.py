"""fr12d: apply the frozen UB10 policy to ANY test submission via the lookup $BER_SCRATCH/fr12d/ub/ub10_lookup_test.parquet.
Drops an accepted pair iff stage-1 p1 > 0.98 and lookup ub10 == True. FAILS LOUDLY if an accepted upper-band pair of the input is missing
from the lookup (extend it first: research/scripts_62/ub10_lookup.py extend <file>, score new_keys, merge).
  python ub10_apply.py <in matching_results.tsv> <out matching_results.tsv> --countries India,US[,France]
Only matched_entity_ids of affected rows change; header, row order and all other rows are byte-identical."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, sys, argparse, polars as pl
os.chdir(_BER_R); K = ["s1_id", "t_id"]
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--countries", required=True); a = ap.parse_args()
C = set(a.countries.split(",")); assert not os.path.exists(a.out)
lk = pl.read_parquet((_BER_S + "/fr12d/ub/ub10_lookup_test.parquet"), columns=K + ["ub10"])
cty = pl.read_parquet("data/test_records.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
d = pl.read_csv(a.inp, separator="\t", quote_char=None, null_values=[""], schema_overrides={"matched_entity_ids": pl.String}).fill_null("")
acc = d.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"}).with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1_id", pl.col("m").alias("t_id"))
up = pl.scan_parquet("output/test_scores_v12avg.parquet").select(K + ["p"]).filter(pl.col("p") > 0.98).join(acc.lazy(), on=K, how="semi").collect(engine="streaming")
miss = up.join(lk, on=K, how="anti")
if miss.height: sys.exit(f"FAIL: {miss.height} accepted upper-band pairs have no UB10 lookup entry; extend the lookup first")
drop = up.join(lk.filter(pl.col("ub10")), on=K, how="semi").join(cty, on="s1_id").filter(pl.col("country").is_in(list(C)))
nd = {}; [nd.setdefault(s, set()).add(t) for s, t in zip(drop["s1_id"], drop["t_id"])]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); n = 0
with open(a.inp, "rb") as fi, open(a.out, "wb") as fo:
    fo.write(fi.readline())
    for line in fi:
        s1, m = line.decode("utf-8").rstrip("\r\n").split("\t")
        if s1 in nd:
            m2 = ",".join(t for t in m.split(",") if t and t not in nd[s1]); fo.write(f"{s1}\t{m2}\n".encode()); n += m2 != m
        else: fo.write(line)
print(f"upper-band accepted {len(up):,}; UB10 drops in {sorted(C)}: {len(drop):,} pairs, {n:,} rows changed", dict(drop.group_by("country").len().rows()))
