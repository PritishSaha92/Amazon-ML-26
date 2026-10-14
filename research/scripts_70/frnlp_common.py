"""frnlp count-aware CE: shared helpers.

Name crowding per record, label-free: n_same = number of S1 (reference, source==1) records in the SAME country of the
SAME records file (= the gallery the pair lives in: train_records for train/val pairs, test_records for test pairs)
whose name_core (src/text_views.py, precomputed views) equals this record's name_core. Includes the record itself when
it is an S1. Empty name_core -> bucket 'none'.
Text per side: "same-name refs: <bucket> | name | address" (prefix, so truncation never removes the count).
"""
import polars as pl

BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 29, "10-29"), (30, 99, "30-99"), (100, 10**12, "100+")]


def bucket_expr(col="n_same"):
    e = pl.when(pl.col(col).is_null()).then(pl.lit("none"))
    for lo, hi, lab in BUCKETS:
        e = e.when(pl.col(col).is_between(lo, hi)).then(pl.lit(lab))
    return e.otherwise(pl.lit("none"))


def count_table(records_path, namecore_path):
    """entity_id, country, n_same, bucket for every record of the file."""
    r = pl.read_parquet(records_path, columns=["entity_id", "country", "source"])
    v = pl.read_parquet(namecore_path, columns=["entity_id", "name_core"])
    r = r.join(v, on="entity_id", how="left")
    assert r["name_core"].null_count() == 0, "missing name_core"
    s1 = r.filter((pl.col("source") == 1) & (pl.col("name_core") != "")).group_by("country", "name_core").agg(n_same=pl.len())
    r = r.join(s1, on=["country", "name_core"], how="left").with_columns(
        pl.when(pl.col("name_core") == "").then(None).otherwise(pl.col("n_same").fill_null(0)).alias("n_same"))
    return r.select("entity_id", "country", "source", "n_same", bucket=bucket_expr())


def prefix(bucket_col):
    return pl.lit("same-name refs: ") + pl.col(bucket_col) + pl.lit(" | ")
