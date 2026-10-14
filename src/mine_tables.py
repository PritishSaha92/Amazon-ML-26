"""Mine substitution tables and the Indic word dictionary from matched train pairs.

Run:  nice -n 10 python src/mine_tables.py [--sample 2000000] [--workers 12]
Writes data/tables/:
  indic_dict.json      folded Indic word -> [[latin, count], ...]   (ALL Indic-name pairs)
  indic_fallback.json  unseen test Indic word -> [[latin, s1_freq]] via romanize + skeleton
  indic_addr.json      folded Indic address component -> [[latin component, count], ...]
  latin_equiv.json     union-find classes of Latin variants sharing an Indic pivot
  subst.json           {country: {field: [{a, b, count, rev, lift, share, kind}, ...]}}
  equiv_classes.json   {country: {field: [[word, ...], ...]}} union-find over kept substitutions
  inserted_words.json  {country: {field: {word: {"t_only": n, "s1_only": n}}}}

Method (research/03_names_addresses.md): tokens are text_views cleaned words (accents stripped,
Indic folded to Devanagari but NOT latinized). For each matched (S1, target) pair we compare the
two token MULTISETS (order-insensitive, so component shuffles don't hide substitutions):
  * exactly one token on each side differs      -> substitution a (S1) -> b (target)
  * the target has exactly one extra token       -> insertion (t_only)
  * the S1 side has exactly one extra token      -> deletion  (s1_only)
Fields: name, addr (words), addr_comp (whole comma components, for state renames / Indic states).
A substitution is kept when count >= MIN_COUNT and either lift = c(a,b)*N / (out(a)*in(b)) >=
MIN_LIFT (random whole-word replacement noise has lift ~ 1) or it dominates one side's
substitutions (share >= MIN_SHARE: needed for very frequent pairs like road->rd whose lift is low
because they ARE most of N). Digit-only pairs (1->001) are skipped (addr_numbers handles them).
Equivalence classes are per country and field; legal-form/honorific/stopword swaps (llc->inc) are
kept in subst.json with kind="legal" but NOT unioned, and plain synonyms are unioned only when
strongly associated and dominant on one side (see _unionable).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_views as tv  # noqa: E402

OUT_DIR = "data/tables"
MIN_COUNT = 20
MIN_LIFT = 20.0
MIN_SHARE = 0.3
SKEL_FREQ_WEIGHT = 20.0  # fallback score = ratio(romanized, cand) + W*log10(freq); tuned 0/2/5/10/20/40
# Name-suffix pool the generator swaps freely (llc->inc, group->llc): tagged kind="legal", not unioned.
SUFFIX_POOL = frozenset(tv.LEGAL_FORMS | tv.HONORIFICS | tv.NAME_STOPWORDS |
                        {"group", "holdings", "lp", "lllp", "com"})
EQUIV_MIN_COUNT = 5      # latin variants of one Indic pivot
EQUIV_MIN_SHARE = 0.05
INDIC_PATTERN = "[\u0900-\u0DFF]"   # any Brahmic block (polars regex)


# ----------------------------------------------------------------------------------------------
# Worker: one chunk of pairs -> counters
# ----------------------------------------------------------------------------------------------

def _name_toks(s: str) -> list[str]:
    return tv._tokens(tv.clean(s, True), True)


def _addr_comps(s: str) -> list[str]:
    out = []
    for comp in tv.clean(s, False).split(","):
        t = tv._tokens(comp, False)
        if t:
            out.append(" ".join(t))
    return out


def _diff(a: list[str], b: list[str], cty: str, field: str, subs: Counter, ins: Counter) -> None:
    ca, cb = Counter(a), Counter(b)
    only_a, only_b = ca - cb, cb - ca
    na, nb = sum(only_a.values()), sum(only_b.values())
    if na == 1 and nb == 1:
        subs[(cty, field, next(iter(only_a)), next(iter(only_b)))] += 1
    elif na == 0 and nb == 1:
        ins[(cty, field, "t_only", next(iter(only_b)))] += 1
    elif na == 1 and nb == 0:
        ins[(cty, field, "s1_only", next(iter(only_a)))] += 1


def _mine_chunk(rows):
    """rows: list of (country, s1_name, s1_addr, t_name, t_addr)."""
    indic = Counter()
    subs, ins = Counter(), Counter()
    stats = Counter()
    for cty, s1n, s1a, tn, ta in rows:
        a_n, b_n = _name_toks(s1n), _name_toks(tn)
        if tv.has_indic(tn):
            stats["indic_pairs"] += 1
            if len(a_n) == len(b_n):
                stats["indic_same_len"] += 1
                for wa, wb in zip(a_n, b_n):
                    if tv.has_indic(wb) and not tv.has_indic(wa):
                        indic[(wb, wa)] += 1
        else:
            stats["name_pairs"] += 1
            _diff(a_n, b_n, cty, "name", subs, ins)
        if not ta:
            continue
        stats["addr_pairs"] += 1
        ca, cb = _addr_comps(s1a), _addr_comps(ta)
        _diff([w for c in ca for w in c.split()], [w for c in cb for w in c.split()],
              cty, "addr", subs, ins)
        _diff(ca, cb, cty, "addr_comp", subs, ins)
    return indic, subs, ins, stats


def _test_indic_words(names):
    """Worker: count Indic name words in test names."""
    c = Counter()
    for n in names:
        if tv.has_indic(n):
            for w in _name_toks(n):
                if tv.has_indic(w):
                    c[w] += 1
    return c


def _s1_vocab(names):
    c = Counter()
    for n in names:
        c.update(_name_toks(n))
    return c


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def classes(self):
        g = defaultdict(list)
        for x in self.p:
            g[self.find(x)].append(x)
        return [sorted(v) for v in g.values() if len(v) > 1]


def _is_subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _levenshtein_le(a: str, b: str, k: int) -> bool:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b, score_cutoff=k) <= k
    except ImportError:  # pragma: no cover
        return False


def pair_kind(a: str, b: str) -> str:
    """abbrev: one is a same-initial subsequence of the other and much shorter; typo: edit <= 2."""
    if tv.has_indic(a) != tv.has_indic(b):
        return "script"
    if a in SUFFIX_POOL and b in SUFFIX_POOL:
        return "legal"
    s, l_ = (a, b) if len(a) <= len(b) else (b, a)
    if s and s[0] == l_[0] and len(s) <= 0.7 * len(l_) and _is_subseq(s, l_):
        return "abbrev"
    if _levenshtein_le(a, b, 2) and min(len(a), len(b)) >= 4:
        return "typo"
    return "synonym"


def _unionable(a, b, kind, lift, share, share_in) -> bool:
    """Which kept substitutions become equivalences (union-find is transitive, so bridges hurt).

    script: always.  abbrev/typo: the target variant must mostly come from this source word
    (share_in >= 0.5; blocks cafe->ce->care).  synonym: both tokens >= 3 chars, lift >= 5 and
    dominant on one side while non-trivial on the other -- keeps street=saint, mumbai=bombay,
    telangana=andhra pradesh; drops 'no'->'bengaluru' and 'delhi'->'east' bridges.
    """
    if kind == "legal" or len(a) < 2 or len(b) < 2:
        return False
    if kind == "script":
        return True
    if kind in ("abbrev", "typo"):
        return share_in >= 0.5
    return (min(len(a), len(b)) >= 3 and lift >= 5 and max(share, share_in) >= 0.5
            and min(share, share_in) >= 0.05)


def _pool(workers):
    return mp.get_context("fork").Pool(workers)


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def load_pairs(sample: int, seed: int = 0):
    """All pairs whose target name contains Indic + a random `sample` of the rest."""
    import polars as pl
    rec = pl.read_parquet("data/train_records.parquet",
                          columns=["entity_id", "business_name", "business_address", "country"])
    pairs = pl.read_parquet("data/train_pairs.parquet")
    s1 = rec.rename({"entity_id": "s1_id", "business_name": "s1_name",
                     "business_address": "s1_addr"})
    tg = rec.select(pl.col("entity_id").alias("t_id"), pl.col("business_name").alias("t_name"),
                    pl.col("business_address").alias("t_addr"))
    df = pairs.join(s1, on="s1_id").join(tg, on="t_id")
    is_ind = df["t_name"].str.contains(INDIC_PATTERN)
    ind = df.filter(is_ind)
    rest = df.filter(~is_ind)
    rest = rest.sample(min(sample, len(rest)), seed=seed)
    out = pl.concat([ind, rest])
    return out, len(ind), len(df), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2_000_000, help="non-Indic pairs to sample")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import polars as pl

    t0 = time.time()
    df, n_ind, n_all, rec = load_pairs(args.sample)
    print(f"[load] {n_all:,} pairs, {n_ind:,} Indic-name pairs (all used), "
          f"{len(df) - n_ind:,} sampled others; {time.time() - t0:.1f}s", flush=True)

    rows = list(zip(df["country"].to_list(), df["s1_name"].to_list(), df["s1_addr"].to_list(),
                    df["t_name"].to_list(), df["t_addr"].fill_null("").to_list()))
    del df
    indic, subs, ins, stats = Counter(), Counter(), Counter(), Counter()
    t1 = time.time()
    with _pool(args.workers) as pool:
        for a, b, c, d in pool.imap_unordered(_mine_chunk, _chunks(rows, 20_000)):
            indic.update(a); subs.update(b); ins.update(c); stats.update(d)
    del rows
    print(f"[mine] {time.time() - t1:.1f}s  stats={dict(stats)}  "
          f"Indic same word count: {stats['indic_same_len'] / max(1, stats['indic_pairs']):.5%}",
          flush=True)

    # ---------------- Indic dictionary ----------------
    idict = defaultdict(list)
    for (w, lat), n in indic.items():
        idict[w].append([lat, n])
    idict = {w: sorted(v, key=lambda x: -x[1]) for w, v in idict.items()}
    tot = sum(n for v in idict.values() for _, n in v)
    top1 = sum(v[0][1] for v in idict.values())
    print(f"[indic] {len(idict):,} distinct Indic words, {tot:,} aligned occurrences, "
          f"top-1 reading share {top1 / tot:.2%}")
    _dump(idict, args.out, "indic_dict.json")

    # Latin equivalence classes through Indic pivots
    uf = UnionFind()
    for w, v in idict.items():
        total = sum(n for _, n in v)
        keep = [lat for lat, n in v if n >= EQUIV_MIN_COUNT and n / total >= EQUIV_MIN_SHARE]
        for lat in keep[1:]:
            uf.union(keep[0], lat)
    lat_freq = Counter()
    for v in idict.values():
        for lat, n in v:
            lat_freq[lat] += n
    lclasses = sorted(uf.classes(), key=lambda c: -sum(lat_freq[x] for x in c))
    lclasses = [sorted(c, key=lambda x: -lat_freq[x]) for c in lclasses]
    _dump(lclasses, args.out, "latin_equiv.json")
    print(f"[latin_equiv] {len(lclasses)} classes; top: " +
          "; ".join("=".join(c[:6]) for c in lclasses[:15]))

    # ---------------- Test coverage + skeleton fallback ----------------
    t2 = time.time()
    test = pl.read_parquet("data/test_records.parquet",
                           columns=["business_name", "country", "source"])
    tnames = test.filter(pl.col("business_name").str.contains(INDIC_PATTERN))[
        "business_name"].to_list()
    s1_ind = pl.concat([
        rec.filter((pl.col("country") == "India") & pl.col("entity_id").str.starts_with("S1-"))
        ["business_name"],
        test.filter((pl.col("country") == "India") & (pl.col("source") == 1))["business_name"]
    ]).to_list()
    with _pool(args.workers) as pool:
        tcount = Counter()
        for c in pool.imap_unordered(_test_indic_words, _chunks(tnames, 20_000)):
            tcount.update(c)
        vocab = Counter()
        for c in pool.imap_unordered(_s1_vocab, _chunks(s1_ind, 50_000)):
            vocab.update(c)
    occ = sum(tcount.values())
    cov_occ = sum(n for w, n in tcount.items() if w in idict)
    cov_dist = sum(1 for w in tcount if w in idict)
    print(f"[coverage] test Indic name words: {len(tcount):,} distinct / {occ:,} occurrences; "
          f"dictionary covers {cov_dist / len(tcount):.2%} distinct, {cov_occ / occ:.2%} occurrences")

    # skeleton -> India S1 words with that skeleton (transductive: train+test S1 names). For an
    # unseen Indic word: romanize, take same-skeleton S1 words, pick the best string match
    # plus a strong frequency prior); no candidate -> keep the raw romanization.
    from rapidfuzz import fuzz
    import math
    skel_words: dict[str, list] = defaultdict(list)
    for w, n in vocab.items():
        if n >= 2 and w.isalpha() and not tv.has_indic(w):
            k = tv.consonant_skeleton(w)
            if k:
                skel_words[k].append((w, n))

    def fallback(w):
        r = tv.romanize_indic(w).lower()
        cands = skel_words.get(tv.consonant_skeleton(r), ())
        best, best_s = None, -1.0
        for c, n in cands:
            sc = fuzz.ratio(r, c) + SKEL_FREQ_WEIGHT * math.log10(n)
            if sc > best_s:
                best, best_s = (c, n), sc
        return best if best is not None else (r, 0)

    # validate the fallback on dictionary words (weighted by occurrences)
    ok = okv = n_eval = 0
    for w, v in idict.items():
        n = sum(c for _, c in v)
        lat, _ = fallback(w)
        n_eval += n
        ok += n * (lat == v[0][0])
        okv += n * (lat in {x for x, _ in v})
    print(f"[fallback] romanize->skeleton->S1 word on dictionary words: top-1 {ok / n_eval:.2%}, "
          f"any-variant {okv / n_eval:.2%} (occurrence-weighted)")
    unseen = {w: n for w, n in tcount.items() if w not in idict}
    fb = {}
    hits = 0
    for w in unseen:
        lat, f = fallback(w)
        fb[w] = [[lat, f]]
        hits += unseen[w] * (f > 0)
    print(f"[fallback] {len(unseen):,} unseen test words ({sum(unseen.values()):,} occ); "
          f"skeleton hit on {hits / max(1, sum(unseen.values())):.2%} of occurrences; examples: " +
          ", ".join(f"{tv.romanize_indic(w)}->{fb[w][0][0]}" for w, _ in
                    Counter(unseen).most_common(15)))
    _dump(fb, args.out, "indic_fallback.json")
    print(f"[coverage+fallback] {time.time() - t2:.1f}s", flush=True)

    # ---------------- Substitutions ----------------
    by_cf = defaultdict(Counter)
    for (cty, field, a, b), n in subs.items():
        by_cf[(cty, field)][(a, b)] += n
    subst_out = defaultdict(dict)
    uf_c = defaultdict(UnionFind)   # key (country, field)
    indic_addr = defaultdict(Counter)
    for (cty, field), cnt in sorted(by_cf.items()):
        N = sum(cnt.values())
        out_a, in_b = Counter(), Counter()
        for (a, b), n in cnt.items():
            out_a[a] += n; in_b[b] += n
        kept = []
        for (a, b), n in cnt.items():
            if field.startswith("addr") and tv.has_indic(b) and not tv.has_indic(a):
                indic_addr[b][a] += n
            if n < MIN_COUNT or (a.isdigit() and b.isdigit()):
                continue
            lift = n * N / (out_a[a] * in_b[b])
            share, share_in = n / out_a[a], n / in_b[b]
            if lift < MIN_LIFT and max(share, share_in) < MIN_SHARE:
                continue
            kind = pair_kind(a, b)
            kept.append({"a": a, "b": b, "count": n, "rev": cnt.get((b, a), 0),
                         "lift": round(lift, 1), "share": round(share, 4),
                         "share_in": round(share_in, 4), "kind": kind})
            if _unionable(a, b, kind, lift, share, share_in):
                uf_c[(cty, field)].union(a, b)
                if field == "addr_comp" and " " not in a and " " not in b:
                    uf_c[(cty, "addr")].union(a, b)
        kept.sort(key=lambda d: -d["count"])
        subst_out[cty][field] = kept
        print(f"\n[subst] {cty}/{field}: {N:,} one-token substitution events, "
              f"{len(cnt):,} distinct, kept {len(kept)} (count>={MIN_COUNT}, lift>={MIN_LIFT})")
        for d in kept[:30]:
            print(f"   {d['a']:>18} -> {d['b']:<18} {d['count']:>7,}  rev {d['rev']:>6,}  "
                  f"lift {d['lift']:>9}  {d['kind']}")
    _dump(subst_out, args.out, "subst.json")

    eq = defaultdict(dict)
    for (cty, field), uf_ in sorted(uf_c.items()):
        cl = uf_.classes()
        cl.sort(key=lambda c: -len(c))
        eq[cty][field] = cl
        print(f"\n[equiv] {cty}/{field}: {len(cl)} classes, largest sizes {[len(c) for c in cl[:5]]}")
        for c in cl[:20]:
            print("   " + " = ".join(c[:12]) + (" ..." if len(c) > 12 else ""))
    _dump(eq, args.out, "equiv_classes.json")

    ia = {b: [[a, n] for a, n in c.most_common(5)] for b, c in indic_addr.items()
          if sum(c.values()) >= 5}
    _dump(ia, args.out, "indic_addr.json")
    print(f"\n[indic_addr] {len(ia)} Indic address tokens/components; e.g. " +
          ", ".join(f"{b}->{v[0][0]}" for b, v in sorted(ia.items(), key=lambda x: -x[1][0][1])[:12]))

    # ---------------- Insertions ----------------
    ins_out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    agg = defaultdict(Counter)
    for (cty, field, side, w), n in ins.items():
        agg[(cty, field, w)][side] += n
    for (cty, field, w), c in agg.items():
        if sum(c.values()) >= MIN_COUNT:
            ins_out[cty][field][w] = {"t_only": c["t_only"], "s1_only": c["s1_only"]}
    for cty in sorted(ins_out):
        for field in sorted(ins_out[cty]):
            items = sorted(ins_out[cty][field].items(), key=lambda x: -x[1]["t_only"])
            print(f"\n[inserted] {cty}/{field}: {len(items)} words; top target-only: " +
                  ", ".join(f"{w}({d['t_only']:,})" for w, d in items[:25] if d["t_only"]))
            items = sorted(ins_out[cty][field].items(), key=lambda x: -x[1]["s1_only"])
            print(f"[deleted]  {cty}/{field}: top S1-only: " +
                  ", ".join(f"{w}({d['s1_only']:,})" for w, d in items[:20] if d["s1_only"]))
    _dump(ins_out, args.out, "inserted_words.json")
    print(f"\n[done] total {time.time() - t0:.1f}s")


def _dump(obj, out_dir, name):
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    main()
