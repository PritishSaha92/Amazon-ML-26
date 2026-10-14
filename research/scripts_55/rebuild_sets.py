"""Rebuild data/v12/cefold2/ce_train_hard_v12f2{A,B}.parquet locally.
The node's builder input (data/pairs_train_v12_feat.parquet, 115M rows with best_rank/sibling_conflict/shift_sig) never came back,
so the pool SAMPLING cannot be re-run locally. The node's own per-row manifest (manifest_train_{A,B}.parquet = s1_id,t_id,label,pool
of the node's file, in its row order, from cloud_v12v_return.tar) is used instead; country and text are re-attached exactly as
src/build_ce_train_hard.py does (attach_text on data/train_records.parquet, split='train', same column order)."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import hashlib, json, sys, os, polars as pl
sys.path.insert(0, (_BER_R + "/src")); from ce_common import attach_text
D = (_BER_S + "/data/v12/cefold2"); os.chdir(_BER_S)
man = json.load(open(f"{D}/manifest.json")); out = {}
ctry = pl.read_parquet("data/split.parquet", columns=["s1_id", "country"])
for F in "AB":
    m = pl.read_parquet(f"{D}/manifest_train_{F}.parquet")
    x = m.with_row_index("_i").join(ctry, on="s1_id", how="left").sort("_i").drop("_i")
    assert x["country"].null_count() == 0
    x = x.select("s1_id", "t_id", "country", "label", "pool")
    x = attach_text(x, "data/train_records.parquet").with_columns(split=pl.lit("train"))
    p = f"{D}/ce_train_hard_v12f2{F}.parquet"; x.write_parquet(p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    pools = {f"{k}|{l}": n for k, l, n in x.group_by("pool", "label").len().sort("pool").iter_rows()}
    r = dict(file=p, sha256=h, sha256_node=man[F]["sha256"], sha_match=h == man[F]["sha256"], pairs=len(x), pairs_node=man[F]["pairs"],
             unique_owners=x["s1_id"].n_unique(), pos=int(x["label"].sum()), pools_match=pools == man[F]["pools"],
             exposure_after_pos_dup1=len(x) + int((x["pool"] == "pos_hard").sum()),
             empty_a=int((x["a"] == "").sum()), empty_b=int((x["b"] == "").sum()), schema=str(x.schema))
    assert r["pairs"] == man[F]["pairs"] and r["unique_owners"] == man[F]["unique_owners"] and r["pos"] == man[F]["pos"] and r["pools_match"]
    out[F] = r; print(F, json.dumps(r), flush=True)
json.dump(out, open(f"{D}/fr12_rebuild_check.json", "w"), indent=1)
