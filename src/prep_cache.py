"""Cache all TSVs as parquet with a `split` and `source` column for fast reloads.
Run: python src/prep_cache.py   → data/{train,test}_records.parquet, data/train_pairs.parquet
"""
import polars as pl, os
D = "student_resource/dataset/"
os.makedirs("data", exist_ok=True)
def load(split):
    dfs = []
    for s in (1, 2, 3):
        df = pl.read_csv(f"{D}{split}/{split}_source{s}.tsv", separator="\t", quote_char=None,
                         schema_overrides={c: pl.Utf8 for c in ["entity_id","business_name","business_address","country"]},
                         missing_utf8_is_empty_string=True)
        dfs.append(df.with_columns(pl.lit(s, dtype=pl.Int8).alias("source")))
    out = pl.concat(dfs).with_columns(pl.col("business_address").fill_null(""), pl.col("business_name").fill_null(""))
    out.write_parquet(f"data/{split}_records.parquet")
    print(split, out.shape, out.group_by("source","country").len().sort("source","country"))
load("train"); load("test")
gt = pl.read_csv(f"{D}train/train_ground_truth.tsv", separator="\t", quote_char=None,
                 schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}, missing_utf8_is_empty_string=True)
gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
pairs = (gt.filter(pl.col("matched_entity_ids") != "")
           .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "t_id"}))
pairs.write_parquet("data/train_pairs.parquet"); gt.write_parquet("data/train_gt.parquet")
print("pairs", pairs.shape, "unique targets", pairs["t_id"].n_unique())
