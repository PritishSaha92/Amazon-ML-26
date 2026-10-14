"""S1-grouped split of the TRAIN set. Retrieval is unsupervised and always runs against the full gallery;
this split only governs which S1 entities (and their linked targets) the supervised matcher may see.
  train : everything else (fit classifier; negatives may be mined from these S1's candidates)
  val   : 60k S1 entities, stratified by country — model selection, thresholds, ownership policy
  audit : 20k S1 entities, stratified — touched only for final reporting
Run: python src/split.py  → data/split.parquet (s1_id, split, country)"""
import polars as pl, numpy as np
rng = np.random.default_rng(20260925)
s1 = pl.read_parquet("data/train_records.parquet").filter(pl.col("source") == 1).select("entity_id", "country")
parts = []
for c, g in sorted(s1.group_by("country"), key=lambda kv: kv[0][0]):  # sorted: polars group_by order is random
    ids = g["entity_id"].to_numpy().copy(); rng.shuffle(ids)
    frac = len(ids) / len(s1)
    nv, na = int(round(60_000 * frac)), int(round(20_000 * frac))
    lab = np.array(["train"] * len(ids), dtype=object); lab[:nv] = "val"; lab[nv:nv + na] = "audit"
    parts.append(pl.DataFrame({"s1_id": ids, "split": lab, "country": [c[0]] * len(ids)}))
out = pl.concat(parts)
out.write_parquet("data/split.parquet")
print(out.group_by("split", "country").len().sort("split", "country"))
