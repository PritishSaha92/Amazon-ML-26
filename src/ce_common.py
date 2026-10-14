"""Shared helpers for the cross-encoder reranker (train_cross_encoder.py / score_cross_encoder.py).

Pair text = "name | address" for each side, fed to the tokenizer as (text, text_pair) so the model sees
<s> S1 </s></s> target </s> (XLM-R layout). Missing name/address become empty strings.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np, polars as pl, torch
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()


def record_text(records_path: str, ids: pl.Series | None = None) -> pl.DataFrame:
    """entity_id -> text ("name | address"), optionally restricted to `ids`."""
    r = pl.scan_parquet(records_path).select("entity_id", "business_name", "business_address")
    if ids is not None:
        r = r.join(pl.DataFrame({"entity_id": ids.unique()}).lazy(), on="entity_id", how="semi")
    return r.select("entity_id", text=pl.col("business_name").fill_null("").str.strip_chars() + " | "
                    + pl.col("business_address").fill_null("").str.strip_chars()).collect()


def attach_text(pairs: pl.DataFrame, records_path: str) -> pl.DataFrame:
    """Adds columns a (S1 text) and b (target text) to a frame with s1_id, t_id. Keeps row order."""
    txt = record_text(records_path, pl.concat([pairs["s1_id"], pairs["t_id"]]))
    out = (pairs.with_row_index("_i")
           .join(txt.rename({"entity_id": "s1_id", "text": "a"}), on="s1_id", how="left")
           .join(txt.rename({"entity_id": "t_id", "text": "b"}), on="t_id", how="left")
           .sort("_i").drop("_i"))
    return out.with_columns(pl.col("a").fill_null(""), pl.col("b").fill_null(""))


def tokenize_pairs(tok, a: list[str], b: list[str], max_len: int, chunk: int = 200_000) -> list[list[int]]:
    """Returns a list of input_id lists (no padding). Chunked to bound memory; the fast tokenizer parallelises internally."""
    ids = []
    for i in range(0, len(a), chunk):
        enc = tok(a[i:i + chunk], b[i:i + chunk], truncation=True, max_length=max_len, padding=False,
                  return_attention_mask=False, return_token_type_ids=False)
        ids.extend(enc["input_ids"])
    return ids


def collate(seqs: list[list[int]], pad_id: int, device) -> dict:
    """Dynamic padding to the longest sequence in the batch."""
    L = max(len(s) for s in seqs)
    x = np.full((len(seqs), L), pad_id, dtype=np.int64)
    m = np.zeros((len(seqs), L), dtype=np.int64)
    for j, s in enumerate(seqs):
        x[j, :len(s)] = s; m[j, :len(s)] = 1
    return dict(input_ids=torch.from_numpy(x).to(device, non_blocking=True),
                attention_mask=torch.from_numpy(m).to(device, non_blocking=True))
