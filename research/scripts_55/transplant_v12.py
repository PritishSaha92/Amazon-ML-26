"""fr12: apply the R49 role maps (research/scripts_49/transplant_fr.py, unchanged; pair-conditioned mode = the recipe that was
trained and passed the R49 gate) to the rebuilt new-fold sets data/v12/cefold2/ce_train_hard_v12f2{A,B}.parquet.
Per-pair checks + revert exactly as R49 (injectivity, digit runs, punctuation, field equality, collisions).
Writes ce_train_hard_v12f2{F}_frmix.parquet (original + transplanted 'ok' copy, column origin) and fr12_transplant_checks_{F}.json."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import sys, os, json, time, hashlib, collections, random
os.chdir(_BER_R); sys.path.insert(0, "research/scripts_49")
from multiprocessing import Pool
import polars as pl
import transplant_fr as T
D = (_BER_S + "/data/v12/cefold2")
def work(args):
    rows, seedtag = args; res = []
    for s1, t, c, a, b in rows:
        rng = random.Random(hashlib.md5(f"{T.SEED}|{seedtag}|{s1}|{t}".encode()).hexdigest())
        try: r = T.transplant(a, b, c, rng)
        except Exception as ex: r = (a, b, 0, "revert:error:" + type(ex).__name__, [], [])
        res.append(r[:4])
    return res
if __name__ == "__main__":
    for F in sys.argv[1:] or ["A", "B"]:
        t0 = time.time(); d = pl.read_parquet(f"{D}/ce_train_hard_v12f2{F}.parquet")
        rows = list(zip(d["s1_id"], d["t_id"], d["country"], d["a"], d["b"])); CH = 20000
        with Pool(int(os.environ.get("NPROC", 11))) as pool:
            res = [x for part in pool.imap(work, [(rows[i:i + CH], f"v12f2{F}") for i in range(0, len(rows), CH)]) for x in part]
        del rows
        A2 = [x[0] for x in res]; B2 = [x[1] for x in res]; NS = [x[2] for x in res]; ST = [x[3] for x in res]; del res
        out = d.with_columns(a_fr=pl.Series(A2), b_fr=pl.Series(B2), tp_nsub=pl.Series(NS, dtype=pl.Int16), tp_status=pl.Series(ST))
        mix = pl.concat([d.with_columns(origin=pl.lit("orig")),
                         out.filter(pl.col("tp_status") == "ok").drop("a", "b").rename({"a_fr": "a", "b_fr": "b"}).select(d.columns).with_columns(origin=pl.lit("fr"))])
        pm = f"{D}/ce_train_hard_v12f2{F}_frmix.parquet"; mix.write_parquet(pm)
        lab = d["label"].to_list(); A0, B0 = d["a"].to_list(), d["b"].to_list(); ok = [i for i, s in enumerate(ST) if s == "ok"]
        rev = collections.Counter(rl for s in ST if s.startswith("revert:") for rl in s[7:].split("+"))
        sig_same = collections.Counter(); smp = random.Random(0).sample(ok, min(200000, len(ok)))
        for i in smp:
            s0, s1_ = T.sig(A0[i], B0[i]), T.sig(A2[i], B2[i])
            for k in s0: sig_same[k] += s0[k] == s1_[k]
            sig_same["ALL_excl_tokdiff_jac"] += all(s0[k] == s1_[k] for k in s0 if k not in ("tokdiff", "jac"))
        chk = dict(fold=F, rows=d.height, status=dict(collections.Counter(ST).most_common()), reverted_by_role=dict(rev.most_common()),
                   status_by_country={f"{c}|{s.split(':')[0]}": n for (c, s), n in sorted(collections.Counter(zip(d["country"].to_list(), ST)).items())},
                   newly_identical_negatives=sum(1 for i in ok if lab[i] == 0 and A2[i] == B2[i] and A0[i] != B0[i]),
                   newly_identical_positives=sum(1 for i in ok if lab[i] == 1 and A2[i] == B2[i] and A0[i] != B0[i]),
                   signature_identical_share_200k={k: round(v / len(smp), 5) for k, v in sig_same.items()},
                   mean_substitutions_ok=sum(NS[i] for i in ok) / max(1, len(ok)), frmix_rows=mix.height,
                   frmix_origin=dict(mix["origin"].value_counts().iter_rows()), frmix_pools={f"{p}|{o}": n for p, o, n in mix.group_by("pool", "origin").len().sort("pool", "origin").iter_rows()},
                   exposure_after_pos_dup1=mix.height + int((mix["pool"] == "pos_hard").sum()),
                   sha256_frmix=hashlib.sha256(open(pm, "rb").read()).hexdigest(), seconds=round(time.time() - t0))
        json.dump(chk, open(f"{D}/fr12_transplant_checks_{F}.json", "w"), indent=1); print(json.dumps(chk), flush=True)
        ch = [i for i in ok if A2[i] != A0[i] or B2[i] != B0[i]]; random.Random(7).shuffle(ch)
        with open(f"{D}/fr12_sample30_{F}.txt", "w") as fh:
            for i in ch[:30]: fh.write(f"[{d['country'][i]} label={lab[i]} pool={d['pool'][i]}]\n  A0: {A0[i]}\n  B0: {B0[i]}\n  A*: {A2[i]}\n  B*: {B2[i]}\n")
        del d, out, mix, A2, B2, A0, B0
