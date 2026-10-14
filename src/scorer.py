"""Exact competition metric: macro F0.5 per Source-1 entity, singletons included.
Per entity: F0.5 = 1.25*TP / (1.25*TP + FP + 0.25*FN); empty truth & empty pred → 1.0; empty truth & any pred → 0.0.

Usage as library:
    from scorer import macro_f05, slice_report
    macro_f05(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float
Usage as CLI (both files are TSVs in submission format):
    python src/scorer.py --pred output/val_pred.tsv --truth data/val_truth.tsv [--records data/train_records.parquet]
"""
import argparse, sys
import numpy as np


def entity_f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth); fp = len(pred) - tp; fn = len(truth) - tp
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn) if tp else 0.0


def per_entity_scores(pred: dict, truth: dict) -> dict:
    """truth keys define the evaluation set; missing pred rows count as empty predictions."""
    return {k: entity_f05(pred.get(k, set()), t) for k, t in truth.items()}


def macro_f05(pred: dict, truth: dict) -> float:
    s = per_entity_scores(pred, truth)
    return float(np.mean(list(s.values()))) if s else float("nan")


def read_submission(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header[0] == "source1_entity_id", header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            k = parts[0]; v = parts[1] if len(parts) > 1 else ""
            out[k] = set(x for x in v.split(",") if x) if v else set()
    return out


def slice_report(pred: dict, truth: dict, meta: dict | None = None) -> dict:
    """meta: s1_id -> country (optional). Returns dict of slice -> (n, macro F0.5, precision-ish, recall-ish)."""
    s = per_entity_scores(pred, truth)
    rows = {}
    def add(name, keys):
        if not keys: return
        vals = [s[k] for k in keys]
        tp = sum(len(pred.get(k, set()) & truth[k]) for k in keys)
        npred = sum(len(pred.get(k, set())) for k in keys); ntrue = sum(len(truth[k]) for k in keys)
        rows[name] = dict(n=len(keys), macro_f05=float(np.mean(vals)),
                          pair_precision=tp / npred if npred else float("nan"), pair_recall=tp / ntrue if ntrue else float("nan"),
                          pred_per_entity=npred / len(keys), true_per_entity=ntrue / len(keys))
    keys = list(truth)
    add("ALL", keys)
    add("singletons", [k for k in keys if not truth[k]])
    add("non-singletons", [k for k in keys if truth[k]])
    if meta:
        for c in sorted(set(meta.get(k, "?") for k in keys)):
            add(f"country={c}", [k for k in keys if meta.get(k, "?") == c])
    return rows


def print_report(rows: dict):
    print(f"{'slice':<18}{'n':>9}{'macroF05':>10}{'pairP':>8}{'pairR':>8}{'pred/ent':>9}{'true/ent':>9}")
    for k, r in rows.items():
        print(f"{k:<18}{r['n']:>9}{r['macro_f05']:>10.4f}{r['pair_precision']:>8.3f}{r['pair_recall']:>8.3f}{r['pred_per_entity']:>9.2f}{r['true_per_entity']:>9.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--records", default=None, help="parquet with entity_id,country for country slices")
    a = ap.parse_args()
    pred, truth = read_submission(a.pred), read_submission(a.truth)
    meta = None
    if a.records:
        import polars as pl
        df = pl.read_parquet(a.records, columns=["entity_id", "country"]).filter(pl.col("entity_id").is_in(list(truth)))
        meta = dict(zip(df["entity_id"], df["country"]))
    print_report(slice_report(pred, truth, meta))
