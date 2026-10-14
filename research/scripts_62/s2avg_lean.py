"""fr12d: memory-lean equivalent of `cloud/s2avg.py --test --tag T` (which OOMs at the 40G cap on 105M-row files).
Streams the three seed files in aligned row chunks; asserts the (s1_id, t_id) keys are identical row-by-row (stage2_apply writes the
row skeleton of output/test_scores_v12avg.parquet for every seed, so s2avg's left join on the base == row-wise alignment);
p = (p0 + p1 + p2) / 3 cast to the base dtype; drops p_s2. Output schema = base schema minus p_s2.
  python research/scripts_62/s2avg_lean.py --tag T [--check-against output/test_scores_v12ce2_E2_s3avg_s2.parquet --seedfiles-of v12ce2_E2]"""
import argparse, polars as pl, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--chunk", type=int, default=8_000_000); a = ap.parse_args()
K = ["s1_id", "t_id"]; fs = [f"output/test_scores_{a.tag}{x}_s2.parquet" for x in ("", "_seed1", "_seed2")]
out = f"output/test_scores_{a.tag}_s3avg_s2.parquet"
n = [pq.ParquetFile(f).metadata.num_rows for f in fs]; assert len(set(n)) == 1, n; N = n[0]
sch = pl.read_parquet_schema(fs[0]); assert "p" in sch
w = None; tot = 0.0
for off in range(0, N, a.chunk):
    b = pl.scan_parquet(fs[0]).slice(off, a.chunk).collect()
    o = [pl.scan_parquet(f).select(K + ["p"]).slice(off, a.chunk).collect() for f in fs[1:]]
    for x in o: assert x.select(K).equals(b.select(K)), f"key misalignment at chunk {off}"
    b = b.with_columns(((pl.col("p") + o[0]["p"] + o[1]["p"]) / 3).cast(sch["p"]).alias("p"))
    b = b.drop([c for c in ("p_s2",) if c in b.columns]); t = b.to_arrow()
    if w is None: w = pq.ParquetWriter(out, t.schema)
    w.write_table(t); tot += float(b["p"].sum()); print(f"rows {off + len(b):,}/{N:,}", flush=True)
w.close(); print(f"wrote {out} rows {N:,} (mean p {tot / N:.6f})")
c = pl.read_parquet_schema(out); assert "p_s2" not in c and "p" in c; print("OK columns", list(c))
