"""Node: entity_id -> name_core for train/test records (same function + data/tables as the production views)."""
import os as _ber_os; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import multiprocessing as mp, sys, os, time
import polars as pl
sys.path.insert(0, "src")
import text_views as tv

TB = tv.Tables.load("data/tables")
def work(names):
    return [tv._name_views(n or "", TB)["name_core"] for n in names]

if __name__ == "__main__":
    T0 = time.time()
    for s in ["train", "test"]:
        r = pl.read_parquet(f"data/{s}_records.parquet", columns=["entity_id", "business_name"])
        names = r["business_name"].fill_null("").to_list(); C = 50_000
        with mp.get_context("fork").Pool(int(os.environ.get("NPROC", 100))) as p:
            out = [x for ch in p.imap(work, [names[i:i + C] for i in range(0, len(names), C)]) for x in ch]
        r.select("entity_id").with_columns(name_core=pl.Series(out)).write_parquet(f"{_BER_S}/frnlp/namecore_{s}.parquet")
        print(f"[{time.time()-T0:.0f}s] {s}: {len(out):,}", flush=True)
