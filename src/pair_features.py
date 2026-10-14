# packaged copy: module docstring made raw (regex text in it; no invalid-escape warning)
r"""Pairwise string / frequency features for the matcher (adds numeric columns to a build_pairs.py pairs file).

Run:
  POLARS_MAX_THREADS=12 nice -n 10 python src/pair_features.py --pairs data/pairs_timing3.parquet --split train \
      --out data/pairs_timing3_feat.parquet [--chunk 2000000]
  POLARS_MAX_THREADS=12 nice -n 10 python src/pair_features.py --pairs data/pairs_timing3.parquet --split train --bench
      (times 1M pairs, tiling the input if it is smaller, and extrapolates to 100M; writes nothing; the stage-time
       line includes the generator-signature stages "shift" and "wordlr"; the word-table build is logged at setup)
  Word-LR only (no pairwise string/token/collision tables are built; ~few GB):
  POLARS_MAX_THREADS=6 nice -n 10 python src/pair_features.py --split test --wordlr-only --wordlr-min-count 5 \
      --wordlr-table-out data/tables/wordlr_test_min5.json                                    (table only)
  ... --wordlr-only --wordlr-min-count 5 --wordlr-table-out data/tables/wordlr_test_min5.json --wordlr-countries France \
      --pairs data/pairs_test_v7_feat.parquet --out data/pairs_test_v7_frv2_feat.parquet
      (--pairs is an existing FEATURES file; the 8 word-LR columns ins_lr_max ... rep_lr_max are recomputed with the
       new table for rows whose `country` is in --wordlr-countries (default: all); every other column/row is copied)

Output = every input column, same row order, plus the feature columns below. Row chunks of --chunk rows are read with
pyarrow and written with a ParquetWriter, so peak memory = per-split tables (~10-15 GB for 12.5M records) + one chunk.
Inputs: data/views_{split}.parquet (src/text_views.py views; must be row-aligned with data/{split}_records.parquet, which
is asserted), data/{split}_records.parquet (raw name for the domain flag, country, source), data/tables/inserted_words.json.

Conventions / sentinels (all features NaN-free):
  * rapidfuzz scores keep their native scale: fuzz.* in [0, 100], JaroWinkler / Levenshtein normalized_similarity in
    [0, 1]. Any similarity is -1 when either side's view string is empty (e.g. empty target address, empty name_core).
  * IDF = ln(N_c / df_c(token)) with N_c = number of records (S1+S2+S3) of country c in the split and df_c = number of
    those records whose view contains the token (test split includes France, so France gets proper statistics).
    IDF sums / maxima are 0 when there is no shared (resp. unshared) token. Jaccard is -1 when either token set is empty.
  * Collision counts: over the split's S1 records of the same country; 0 when the key is empty.
  * Flags are int8 0/1.

Feature columns (S1 = the Source-1 record, T = the target record):
 name (rapidfuzz, S1 name_words vs T name_latin unless noted)
  nm_ratio, nm_tsort, nm_tset, nm_partial      fuzz ratio / token_sort_ratio / token_set_ratio / partial_ratio
  nm_jw                                        Jaro-Winkler normalized similarity
  nc_tset, nc_jw                               token_set_ratio / Jaro-Winkler on name_core (legal forms etc. removed)
  ncomp_lev, ncomp_partial                     normalized Levenshtein similarity / partial_ratio on name_compact
  nleet_lev, nleet_partial                     same on name_leet (leet-folded compact; catches domain forms)
  nskel_ratio                                  fuzz.ratio on name_skel (consonant skeleton)
  eq_name_core, eq_name_sorted, eq_name_compact  exact-equality flags
  s1_nwords, t_nwords, nwords_diff             word counts of name_words (clipped to 127) and S1 - T
 name_core token sets (per-country IDF over all records of the split)
  nc_jacc                                      Jaccard of word sets (-1 if either empty)
  nc_n_shared                                  number of shared words
  nc_idf_shared_sum, nc_idf_shared_max         sum of IDF of shared words / IDF of the rarest shared word
  nc_idf_s1_unshared_max                       max IDF of an S1 word absent from T (rare word the target lacks)
  nc_idf_cov                                   shared-IDF sum / total S1 IDF (0 if S1 has no words)
  nc_shared_all_generic                        1 if >=1 shared word and every shared word is generic (mined
                                               inserted_words.json name words with count >= 50 in any country, legal
                                               forms, honorifics, stopwords)
 address
  aw_tset, aw_tsort, aw_partial                token_set / token_sort / partial_ratio on addr_words
  ac_tset                                      token_set_ratio on addr_core
  ac_jacc, ac_n_shared, ac_idf_shared_sum, ac_idf_shared_max, ac_idf_s1_unshared_max, ac_idf_cov
                                               same token-set features on addr_core tokens
  addr_len_ratio                               min/max char length of addr_words (-1 if either empty)
  s1_addr_empty, t_addr_empty                  addr_words empty flags
 street / house numbers (addr_numbers: leading zeros stripped, hyphen forms expanded by text_views)
  num_any_shared, num_n_shared                 any / count of exactly shared numbers
  num_presuf                                   some S1 number is a prefix/suffix of some T number or vice versa (not
                                               equal, shorter one >= 2 digits; e.g. dropped first digit 4948 -> 948)
  num_one_edit                                 some number pair is one OSA edit apart (substitution, insertion,
                                               deletion or adjacent transposition), both >= 2 digits
  num_s1_empty, num_t_empty                    side has no numbers
  num_conflict                                 both sides have numbers and none is exactly shared
  num_grade                                    3 exact shared, 2 prefix/suffix, 1 one-edit, 0 a side has no number,
                                               -1 conflict with no fuzzy agreement
  num_jacc                                     Jaccard of the two number sets (-1 if either side has no number)
  num_n_s1_only, num_n_t_only                  # numbers on that side absent from the other side
  num_any_subst1                               some cross pair has same length, exactly one digit different, and
                                               neither number of the pair is shared exactly (neighbouring-address
                                               distractor signature: 34234 vs 34237)
  num_max_len_shared                           digit length of the longest exactly shared number (0 if none)
  num_subst1_any, num_indel1_any               split of num_one_edit: some pair (both >= 2 digits) with same length and
                                               Hamming distance 1 / lengths differing by 1 and Indel distance 1
 house number = FIRST number run in the raw address order (regex [0-9]+(?:\s*-\s*[0-9]+)*, leading zeros stripped,
   hyphen parts joined with the text_views rule: 2 parts, shorter <= 2 digits, total <= 6 digits -> "2-43" = 243,
   "18-244" = 18244; otherwise the first part)
  hn_exact                                     equal
  hn_presuf                                    one is a proper prefix/suffix of the other, shorter >= 2 digits
  hn_indel1                                    lengths differ by 1 and one insertion/deletion aligns them (not presuf)
  hn_subst1, hn_subst2                         same length, exactly one / two digits differ
  hn_other_conflict                            both present and none of the above
  hn_missing                                   either side has no number
  hn_grade                                     exact 4, presuf 3, indel1 2, missing 1, subst1 -1, subst2 -2, other -3
                                               (non-negative = consistent with true-match noise: truncation / padding /
                                               dropped digit; negative = substituted digits = neighbouring address)
 frequency / collision (S1 records of the same country in this split)
  s1_name_core_count, s1_name_compact_count, s1_addr_core_count, s1_num_addr_count
                                               # S1 records (incl. this one) sharing this S1's name_core / name_compact /
                                               addr_core / (first number, addr_core)
  t_name_core_s1_count, t_addr_core_s1_count, t_num_addr_s1_count
                                               # S1 records whose name_core / addr_core / (first number, addr_core)
                                               equals the target's
 flags
  t_has_indic                                  target name contains Indic script
  t_is_domain_form                             raw target name matches \.(com|net|org|in|co|io|fr|us)\b (case-insens.)
  (t_source is already in the input)
 generator signature: address-number shift (research/06_synthetic_generator.md F1; reference = S1, target = T).
   Numbers = raw digit runs \d+ of business_address; "stripped" = leading zeros removed ("0" if all zeros).
   D = {1,2,3,4,5,7,9,11,13,21}. to = target raw runs whose stripped form is not among the reference's;
   ro = distinct stripped reference numbers occurring more often in the reference than in the target (multiset R - T);
   int(x) uses the first 15 digits. This is exactly shift_sig2() of the research scratchpad sig2.py (verified).
  shift_sig                                    3 some to == some ro + d, d in D; 2 some to (stripped) equals r+d with its
                                               first or last digit dropped (len(r+d) >= 3), or the raw zero-padded token
                                               ends with (r+d)[1:]; 1 other nonzero |to - ro| <= 25; 0 none (incl. a side
                                               with no unshared number)
  shift_d                                      signed to - so of the closest pair of UNSHARED stripped numbers (set
                                               semantics, min |d|, ties -> positive), clipped to +-999; 0 = no such pair
  shift_d_big                                  some (to, ro) pair has to - ro in {3,4,5,7,9,11,13,21} (1/2 also occur as
                                               true-match jitter)
  shift_neg                                    no forward D-shift, but some pair has ro - to in D (reverse direction, which
                                               the generator does not produce)
 generator signature: inserted / deleted / replaced name words (F2/F3). Per-country log-likelihood-ratio tables,
   learned on THE SPLIT BEING PROCESSED without labels: every target's rank-0 reverse-sparse reference
   (cands_rsparse_{split}, view rfull; fallback rdense rank 0; train split: only references in split=="train");
   pseudo-positives = shift_sig 3, pseudo-negatives = shift_sig 0 and >= 1 exactly shared stripped number.
   Words: name_core, multiset difference. ins = distinct words of T - S1, del = distinct words of S1 - T, rep = ins
   words of pairs whose name_core word counts are equal. LR(w) = log((pos_w+1)/(N_pos+2)) - log((neg_w+1)/(N_neg+2))
   (counts = pairs containing w; N_rep = pairs with equal word counts), 0 if pos_w + neg_w < 20 or unseen.
   Tables written to data/tables/wordlr_{split}.json (not in --bench mode).
  ins_lr_max, ins_lr_sum, ins_lr_min, n_ins   over inserted words (max/min/sum 0 when n_ins == 0)
  del_lr_max, del_lr_sum, n_del               over deleted words (0 when n_del == 0)
  rep_lr_max                                   max rep-table LR over inserted words when word counts are equal, else 0
  dba_flag                                     raw target name contains dba / aka / formerly / fka / nee / née /
                                               "doing business as" / d/b/a / a.k.a / f.k.a (case-insensitive)
"""
import argparse, json, os, resource, sys, time
os.environ.setdefault("POLARS_MAX_THREADS", "12")
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Hamming, Indel, JaroWinkler, Levenshtein, OSA

sys.path.insert(0, os.path.dirname(__file__))
from text_views import LEGAL_FORMS, HONORIFICS, NAME_STOPWORDS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs")
ap.add_argument("--split", required=True, choices=["train", "test"])
ap.add_argument("--out")
ap.add_argument("--chunk", type=int, default=2_000_000)
ap.add_argument("--bench", action="store_true", help="time features on 1M pairs and extrapolate to 100M; no output")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--data", default="data")
ap.add_argument("--generic-min-count", type=int, default=50)
ap.add_argument("--wordlr-min-count", type=int, default=20)
ap.add_argument("--wordlr-table-out", default="", help="word-LR table json path (default <data>/tables/wordlr_<split>.json)")
ap.add_argument("--wordlr-only", action="store_true", help="build the word-LR table only; with --pairs (a features file) and "
                "--out, recompute only the 8 word-LR columns (rows of --wordlr-countries) and copy everything else")
ap.add_argument("--wordlr-countries", default="", help="--wordlr-only: comma-separated countries to recompute (default all)")
ap.add_argument("--wordlr-table-in", default="", help="--wordlr-only with --pairs/--out: use this word-LR table json "
                "(e.g. data/tables/wordlr_train.json; LRs recomputed at full precision from its stored counts, as mined) instead of "
                "mining this split's table; nothing is written to data/tables. --wordlr-countries must be countries of the json")
a = ap.parse_args()
if not a.bench and not a.out and not a.wordlr_only: raise SystemExit("--out required unless --bench / --wordlr-only")
if not a.wordlr_only and not a.pairs: raise SystemExit("--pairs required")
if a.wordlr_only and a.bench: raise SystemExit("--wordlr-only and --bench are exclusive")
if a.wordlr_table_in and not (a.wordlr_only and a.pairs): raise SystemExit("--wordlr-table-in needs --wordlr-only with --pairs/--out")
if a.wordlr_only and bool(a.pairs) != bool(a.out): raise SystemExit("--wordlr-only: give both --pairs and --out, or neither (table only)")
W = a.workers
T0 = time.time()
def log(msg): print(f"[{time.time() - T0:7.1f}s] {msg}  (maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB)", flush=True)
T_MUL = 1_000_000_000

# ============================================ per-split tables ===================================================
VIEW_COLS = ["entity_id", "has_indic", "name_words", "name_latin", "name_core", "name_compact", "name_leet", "name_sorted",
             "name_skel", "addr_words", "addr_core", "addr_numbers"]
V = pl.read_parquet(f"{a.data}/views_{a.split}.parquet", columns=VIEW_COLS)
R = pl.read_parquet(f"{a.data}/{a.split}_records.parquet", columns=["entity_id", "business_name", "business_address", "country", "source"])
assert len(V) == len(R) and (V["entity_id"] == R["entity_id"]).all(), "views not row-aligned with records"
def house_number(addr: pl.Expr) -> pl.Expr:
    """First number run of the raw address; leading zeros stripped; text_views hyphen-join rule (2-43 -> 243)."""
    hp = (addr.fill_null("").str.extract(r"([0-9]+(?:\s*-\s*[0-9]+)*)", 1).str.replace_all(r"\s", "").str.split("-")
          .list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.when(pl.element() == "").then(pl.lit("0")).otherwise(pl.element())))
    p0, p1 = hp.list.get(0, null_on_oob=True), hp.list.get(1, null_on_oob=True)
    join = ((hp.list.len() == 2) & (pl.min_horizontal(p0.str.len_chars(), p1.str.len_chars()) <= 2)
            & ((p0.str.len_chars() + p1.str.len_chars()) <= 6)).fill_null(False)
    return pl.when(join).then(p0 + p1).otherwise(p0)
HN = R.select(hn=house_number(pl.col("business_address")))["hn"]
DBA_RE = r"(?i)(?:\b(?:dba|aka|formerly|fka|nee)\b|\bnée\b|doing business as|\bd\s*/\s*b\s*/\s*a\b|\bd\.b\.a\b|\ba\.k\.a\b|\bf\.k\.a\b)"
V = V.with_columns(rnum=R["business_address"].fill_null("").str.extract_all(r"\d+"),
                   t_dba=R["business_name"].fill_null("").str.contains(DBA_RE).cast(pl.Int8))
V = V.with_columns(country=R["country"], source=R["source"], hn=HN,
                   t_dom=R["business_name"].str.contains(r"(?i)\.(?:com|net|org|in|co|io|fr|us)\b").fill_null(False).cast(pl.Int8))
del R
V = V.with_columns(
    h_sorted=pl.col("name_sorted").hash(), h_core=pl.col("name_core").hash(), h_compact=pl.col("name_compact").hash(),
    nwords=pl.when(pl.col("name_words").str.len_bytes() == 0).then(0)
             .otherwise(pl.col("name_words").str.count_matches(" ") + 1).clip(0, 127).cast(pl.Int8),
    alen=pl.col("addr_words").str.len_chars().cast(pl.Int32),
    has_indic=pl.col("has_indic").cast(pl.Int8),
).drop("name_sorted")
N = len(V)
log(f"loaded views {a.split}: {N:,} records")

# id -> row index via searchsorted on numeric keys (S{src}-{num} -> src*1e9 + num)
def id_key(s: pl.Series) -> np.ndarray:
    return (s.str.slice(1, 1).cast(pl.Int64) * T_MUL + s.str.slice(3).cast(pl.Int64)).to_numpy()
_keys = id_key(V["entity_id"]); _order = np.argsort(_keys, kind="stable"); _sk = _keys[_order]; del _keys
def lookup(ids: pl.Series) -> np.ndarray:
    k = id_key(ids); pos = np.minimum(np.searchsorted(_sk, k), len(_sk) - 1)
    bad = _sk[pos] != k
    if bad.any(): raise SystemExit(f"{int(bad.sum())} ids not found in views, e.g. {ids.filter(pl.Series(bad)).head(3).to_list()}")
    return _order[pos]

# generic name words: mined insertions (any country) + legal forms / honorifics / stopwords
ins = json.load(open(f"{a.data}/tables/inserted_words.json"))
GENERIC = set(LEGAL_FORMS) | set(HONORIFICS) | set(NAME_STOPWORDS)
for c in ins:
    for w, d in ins[c].get("name", {}).items():
        if d.get("t_only", 0) + d.get("s1_only", 0) >= a.generic_min_count: GENERIC.add(w)
log(f"generic name words: {len(GENERIC)}")

def token_table(col: str, generic: set | None):
    """Per-country vocab over all records: returns (List[UInt32] token-id column aligned with V, idf array, generic array)."""
    t = (V.select(idx=pl.int_range(pl.len(), dtype=pl.UInt32), country="country",
                  tok=pl.col(col).str.split(" ").list.unique(maintain_order=True))
          .explode("tok", empty_as_null=True).filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))
    ncty = V.group_by("country").agg(Nc=pl.len())
    vocab = (t.group_by("country", "tok").agg(df=pl.len()).join(ncty, on="country")
              .with_columns(tok_id=pl.int_range(pl.len(), dtype=pl.UInt32),
                            idf=(pl.col("Nc").cast(pl.Float64) / pl.col("df")).log().cast(pl.Float32)))
    idf = vocab["idf"].to_numpy()
    gen = vocab["tok"].is_in(list(generic)).to_numpy() if generic is not None else None
    t = t.join(vocab.select("country", "tok", "tok_id"), on=["country", "tok"], how="left", maintain_order="left").sort("idx", maintain_order=True)
    idx = t["idx"].to_numpy(); counts = np.bincount(idx, minlength=N)
    offsets = np.zeros(N + 1, dtype=np.int32); np.cumsum(counts, out=offsets[1:])
    arr = pa.ListArray.from_arrays(pa.array(offsets), pa.array(t["tok_id"].to_numpy()))
    return pl.Series(f"{col}_tok", arr), idf, gen, len(vocab)

if not a.wordlr_only:
    nc_tok, NC_IDF, NC_GEN, nv = token_table("name_core", GENERIC); V = V.with_columns(nc_tok)
    log(f"name_core vocab {nv:,}")
    ac_tok, AC_IDF, _, nv = token_table("addr_core", None); V = V.with_columns(ac_tok); del nc_tok, ac_tok
    log(f"addr_core vocab {nv:,}")

# name_core word ids (global vocab, multiset order kept) + country index for the word-LR tables
_wt = (V.select(idx=pl.int_range(pl.len(), dtype=pl.UInt32), w=pl.col("name_core").str.split(" "))
         .explode("w", empty_as_null=True).filter(pl.col("w").is_not_null() & (pl.col("w") != "")))
_wv = _wt.select(pl.col("w").unique().sort()).with_columns(wid=pl.int_range(pl.len(), dtype=pl.UInt32))
WORDS = _wv["w"].to_list(); NW = len(WORDS)
_wt = _wt.join(_wv, on="w", how="left", maintain_order="left")
_cnt = np.bincount(_wt["idx"].to_numpy(), minlength=N); _off = np.zeros(N + 1, dtype=np.int32); np.cumsum(_cnt, out=_off[1:])
V = V.with_columns(pl.Series("ncw", pa.ListArray.from_arrays(pa.array(_off), pa.array(_wt["wid"].to_numpy()))))
del _wt, _wv, _cnt, _off
CTRY = sorted(V["country"].unique().to_list())
V = V.with_columns(cidx=pl.col("country").replace_strict({c: i for i, c in enumerate(CTRY)}, return_dtype=pl.Int8))
log(f"name_core word vocab {NW:,}; countries {CTRY}")

# collision counts over S1 records of the same country
if a.wordlr_only:  # only rnum / ncw / cidx / t_dba are needed for the word-LR features
    V = V.select("entity_id", "rnum", "ncw", "cidx", "t_dba", "country")
else:
    V = V.with_columns(k_numaddr=pl.when(pl.col("addr_core").str.len_bytes() == 0).then(pl.lit(""))
                       .otherwise(pl.col("addr_numbers").list.first().fill_null("") + "|" + pl.col("addr_core")))
    for key, out in [("name_core", "c_core"), ("name_compact", "c_compact"), ("addr_core", "c_addr"), ("k_numaddr", "c_numaddr")]:
        cnt = V.filter(pl.col("source") == 1).group_by("country", key).agg(pl.len().cast(pl.Float32).alias(out))
        V = V.join(cnt, on=["country", key], how="left", maintain_order="left").with_columns(
            pl.when(pl.col(key).str.len_bytes() == 0).then(0.0).otherwise(pl.col(out).fill_null(0.0)).cast(pl.Float32).alias(out))
    V = V.drop("k_numaddr", "source")
    assert len(V) == N
    log("collision tables done")

# ============================================ per-chunk features =================================================
def cp(x, y, scorer):
    return process.cpdist(x, y, scorer=scorer, workers=W, dtype=np.float32)

def sim(S, T, col_s, col_t, scorers, names, out):
    xs, xt = S[col_s], T[col_t]
    empty = ((xs.str.len_bytes() == 0) | (xt.str.len_bytes() == 0)).to_numpy()
    ls, lt = xs.to_list(), xt.to_list()
    for sc, nm in zip(scorers, names):
        v = cp(ls, lt, sc); v[empty] = -1.0; out[nm] = v

def flat(s: pl.Series):
    arr = s.rechunk().to_arrow()
    if isinstance(arr, pa.ChunkedArray): arr = arr.combine_chunks()
    return pc.list_parent_indices(arr).to_numpy(), pc.list_flatten(arr)

def token_feats(ls: pl.Series, lt: pl.Series, idf, gen, p, out, n):
    rs, vs = flat(ls); rt, vt = flat(lt)
    vs = vs.to_numpy().astype(np.int64); vt = vt.to_numpy().astype(np.int64)
    ks = (rs.astype(np.int64) << 32) | vs; kt = (rt.astype(np.int64) << 32) | vt
    sh = np.isin(ks, kt)
    w = idf[vs]
    n_s = np.bincount(rs, minlength=n); n_t = np.bincount(rt, minlength=n)
    n_sh = np.bincount(rs, weights=sh.astype(np.float64), minlength=n)
    tot = np.bincount(rs, weights=w, minlength=n)
    sh_sum = np.bincount(rs, weights=w * sh, minlength=n)
    sh_max = np.zeros(n, np.float32); np.maximum.at(sh_max, rs[sh], w[sh])
    un_max = np.zeros(n, np.float32); np.maximum.at(un_max, rs[~sh], w[~sh])
    union = n_s + n_t - n_sh
    out[f"{p}_jacc"] = np.where((n_s == 0) | (n_t == 0), -1.0, n_sh / np.maximum(union, 1)).astype(np.float32)
    out[f"{p}_n_shared"] = np.minimum(n_sh, 127).astype(np.int8)
    out[f"{p}_idf_shared_sum"] = sh_sum.astype(np.float32)
    out[f"{p}_idf_shared_max"] = sh_max
    out[f"{p}_idf_s1_unshared_max"] = un_max
    out[f"{p}_idf_cov"] = np.where(tot > 0, sh_sum / np.maximum(tot, 1e-9), 0.0).astype(np.float32)
    if gen is not None:
        nongen_sh = np.bincount(rs, weights=(sh & ~gen[vs]).astype(np.float64), minlength=n)
        out[f"{p}_shared_all_generic"] = ((n_sh > 0) & (nongen_sh == 0)).astype(np.int8)

def number_feats(ls: pl.Series, lt: pl.Series, out, n):
    rs, vs = flat(ls); rt, vt = flat(lt)
    A = pl.DataFrame({"row": rs, "a": pl.Series(vs, dtype=pl.String)})
    B = pl.DataFrame({"row": rt, "b": pl.Series(vt, dtype=pl.String)})
    x = A.join(B, on="row", how="inner")
    n_s = np.bincount(rs, minlength=n) if len(rs) else np.zeros(n, np.int64)
    n_t = np.bincount(rt, minlength=n) if len(rt) else np.zeros(n, np.int64)
    n_sh = np.zeros(n); pres = np.zeros(n, bool); one = np.zeros(n, bool)
    sub1x = np.zeros(n, bool); sub1 = np.zeros(n, bool); ind1 = np.zeros(n, bool); maxlen = np.zeros(n)
    if len(x):
        la, lb = pl.col("a").str.len_chars(), pl.col("b").str.len_chars()
        x = x.with_columns(eq=pl.col("a") == pl.col("b"),
                           ps=(pl.col("a") != pl.col("b")) & (pl.min_horizontal(la, lb) >= 2)
                              & (pl.col("a").str.ends_with(pl.col("b")) | pl.col("b").str.ends_with(pl.col("a"))
                                 | pl.col("a").str.starts_with(pl.col("b")) | pl.col("b").str.starts_with(pl.col("a"))),
                           ok2=pl.min_horizontal(la, lb) >= 2)
        d = process.cpdist(x["a"].to_list(), x["b"].to_list(), scorer=OSA.distance, workers=W, dtype=np.int32)
        ham = process.cpdist(x["a"].to_list(), x["b"].to_list(), scorer=Hamming.distance, workers=W, dtype=np.int32)
        x = x.with_columns(oe=pl.Series(d == 1) & pl.col("ok2"), ham1=pl.Series(ham == 1) & (la == lb),
                           dl=(la.cast(pl.Int32) - lb.cast(pl.Int32)).abs())
        x = x.with_columns(a_sh=pl.col("eq").any().over("row", "a"), b_sh=pl.col("eq").any().over("row", "b"))
        g = x.group_by("row").agg(nsh=pl.col("a").filter(pl.col("eq")).n_unique(), ps=pl.col("ps").any(), oe=pl.col("oe").any(),
                                  s1x=(pl.col("ham1") & ~pl.col("a_sh") & ~pl.col("b_sh")).any(),
                                  s1=(pl.col("ham1") & pl.col("ok2")).any(),
                                  i1=(pl.col("oe") & (pl.col("dl") == 1)).any(),
                                  ml=pl.col("a").filter(pl.col("eq")).str.len_chars().max().fill_null(0))
        r = g["row"].to_numpy()
        n_sh[r] = g["nsh"].to_numpy(); pres[r] = g["ps"].to_numpy(); one[r] = g["oe"].to_numpy()
        sub1x[r] = g["s1x"].to_numpy(); sub1[r] = g["s1"].to_numpy(); ind1[r] = g["i1"].to_numpy(); maxlen[r] = g["ml"].to_numpy()
    s_e, t_e = n_s == 0, n_t == 0
    anysh = n_sh > 0
    conflict = ~s_e & ~t_e & ~anysh
    out["num_any_shared"] = anysh.astype(np.int8)
    out["num_n_shared"] = np.minimum(n_sh, 127).astype(np.int8)
    out["num_presuf"] = pres.astype(np.int8)
    out["num_one_edit"] = one.astype(np.int8)
    out["num_s1_empty"] = s_e.astype(np.int8)
    out["num_t_empty"] = t_e.astype(np.int8)
    out["num_conflict"] = conflict.astype(np.int8)
    grade = np.where(anysh, 3, np.where(pres, 2, np.where(one, 1, np.where(s_e | t_e, 0, -1))))
    out["num_grade"] = grade.astype(np.int8)
    union = n_s + n_t - n_sh
    out["num_jacc"] = np.where(s_e | t_e, -1.0, n_sh / np.maximum(union, 1)).astype(np.float32)
    out["num_n_s1_only"] = np.minimum(n_s - n_sh, 127).astype(np.int8)
    out["num_n_t_only"] = np.minimum(n_t - n_sh, 127).astype(np.int8)
    out["num_any_subst1"] = sub1x.astype(np.int8)
    out["num_max_len_shared"] = np.minimum(maxlen, 127).astype(np.int8)
    out["num_subst1_any"] = sub1.astype(np.int8)
    out["num_indel1_any"] = ind1.astype(np.int8)

def house_feats(hs: pl.Series, ht: pl.Series, out):
    miss = (hs.is_null() | ht.is_null()).to_numpy()
    a, b = hs.fill_null(""), ht.fill_null("")
    la, lb = a.str.len_chars().to_numpy(), b.str.len_chars().to_numpy()
    eq = (a == b).to_numpy() & ~miss
    ps = (~miss & ~eq & (np.minimum(la, lb) >= 2)
          & (a.str.starts_with(b) | b.str.starts_with(a) | a.str.ends_with(b) | b.str.ends_with(a)).to_numpy())
    al, bl = a.to_list(), b.to_list()
    ham = process.cpdist(al, bl, scorer=Hamming.distance, workers=W, dtype=np.int32)
    ind = process.cpdist(al, bl, scorer=Indel.distance, workers=W, dtype=np.int32)
    same = (la == lb) & ~miss & ~eq
    s1, s2 = same & (ham == 1), same & (ham == 2)
    i1 = ~miss & ~eq & ~ps & (np.abs(la - lb) == 1) & (ind == 1)
    other = ~miss & ~eq & ~ps & ~s1 & ~s2 & ~i1
    for k, v in [("hn_exact", eq), ("hn_presuf", ps), ("hn_indel1", i1), ("hn_subst1", s1), ("hn_subst2", s2),
                 ("hn_other_conflict", other), ("hn_missing", miss)]:
        out[k] = v.astype(np.int8)
    out["hn_grade"] = np.select([eq, ps, i1, miss, s1, s2], [4, 3, 2, 1, -1, -2], -3).astype(np.int8)

D_SHIFT = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]; D_BIG = [3, 4, 5, 7, 9, 11, 13, 21]
def _strip0(e: pl.Expr) -> pl.Expr:
    z = e.str.strip_chars_start("0"); return pl.when(z == "").then(pl.lit("0")).otherwise(z)

def shift_core(lr: pl.Series, lt: pl.Series, n):
    """Vectorised sig2.shift_sig2(target_addr, ref_addr) + extras. lr/lt = raw digit-run lists (reference S1 / target)."""
    rs, vs = flat(lr); rt, vt = flat(lt)
    A = pl.DataFrame({"row": rt, "a": pl.Series(vt, dtype=pl.String)}).with_columns(sa=_strip0(pl.col("a")))
    B = pl.DataFrame({"row": rs, "b": pl.Series(vs, dtype=pl.String)}).with_columns(b=_strip0(pl.col("b")))
    ia, ib = pl.col("sa").str.slice(0, 15).cast(pl.Int64), pl.col("b").str.slice(0, 15).cast(pl.Int64)
    cR = B.group_by("row", "b").agg(cr=pl.len()); cT = A.group_by("row", "sa").agg(ct=pl.len())
    to = A.join(cR, left_on=["row", "sa"], right_on=["row", "b"], how="anti")
    ro = (cR.join(cT, left_on=["row", "b"], right_on=["row", "sa"], how="left")
            .filter(pl.col("cr") > pl.col("ct").fill_null(0)).select("row", "b"))
    X = to.join(ro, on="row").with_columns(d=ia - ib, ibv=ib)
    tr = pl.lit(False)
    for dd in D_SHIFT:
        s = (pl.col("ibv") + dd).cast(pl.String); s_t, s_h = s.str.tail(-1), s.str.head(-1)
        tr = tr | ((s.str.len_chars() >= 3) & ((pl.col("sa") == s_t) | (pl.col("sa") == s_h)
                                               | ((pl.col("a") != pl.col("sa")) & pl.col("a").str.ends_with(s_t))))
    d = pl.col("d")
    g = X.group_by("row").agg(s3=d.is_in(D_SHIFT).any(), s2=tr.any(), s1=((d != 0) & (d.abs() <= 25)).any(),
                              big=d.is_in(D_BIG).any(), neg=(-d).is_in(D_SHIFT).any())
    r = g["row"].to_numpy(); sig = np.zeros(n, np.int8); big = np.zeros(n, np.int8); neg = np.zeros(n, np.int8)
    s3, s2, s1 = g["s3"].to_numpy(), g["s2"].to_numpy(), g["s1"].to_numpy()
    sig[r] = np.where(s3, 3, np.where(s2, 2, np.where(s1, 1, 0))); big[r] = g["big"].to_numpy()
    neg[r] = g["neg"].to_numpy() & ~s3
    # set semantics for the signed closest difference + exact-shared flag
    Ts = A.select("row", "sa").unique(); Ss = B.select("row", "b").unique()
    shared = np.zeros(n, bool)
    sr = Ts.join(Ss, left_on=["row", "sa"], right_on=["row", "b"], how="semi")["row"].to_numpy(); shared[sr] = True
    to2 = Ts.join(Ss, left_on=["row", "sa"], right_on=["row", "b"], how="anti")
    so2 = Ss.join(Ts, left_on=["row", "b"], right_on=["row", "sa"], how="anti")
    X2 = to2.join(so2, on="row").with_columns(d=ia - ib)
    g2 = X2.group_by("row").agg(pl.col("d").sort_by(pl.col("d").abs() * 2 - (pl.col("d") > 0).cast(pl.Int64)).first())
    dmin = np.zeros(n, np.float32); dmin[g2["row"].to_numpy()] = np.clip(g2["d"].to_numpy(), -999, 999)
    return sig, dmin, big, neg, shared

def word_diff(ws: pl.Series, wt: pl.Series, n):
    """Multiset name_core word difference. Returns distinct (row, wid) of T-S (ins) and S-T (del), word counts."""
    rs, vs = flat(ws); rt, vt = flat(wt)
    def keys(r, v):  # (row, word, occurrence#) packed; occurrence = rank among equal (row, word)
        k0 = np.sort((r.astype(np.int64) << 32) | v.to_numpy().astype(np.int64))
        if len(k0) == 0: return k0
        start = np.r_[True, k0[1:] != k0[:-1]]; pos = np.arange(len(k0))
        occ = pos - np.maximum.accumulate(np.where(start, pos, 0))
        return (k0 << 4) | np.minimum(occ, 15)
    ks, kt = keys(rs, vs), keys(rt, vt)  # sorted
    def uniq(x): return x[np.r_[True, x[1:] != x[:-1]]] if len(x) else x
    ik = uniq(kt[~np.isin(kt, ks)] >> 4); dk = uniq(ks[~np.isin(ks, kt)] >> 4)
    m32 = (1 << 32) - 1  # outputs sorted by row
    return (ik >> 32), (ik & m32), (dk >> 32), (dk & m32), np.bincount(rs, minlength=n), np.bincount(rt, minlength=n)

def build_word_tables():
    t0 = time.time()
    c0 = (pl.scan_parquet(f"{a.data}/cands_rsparse_{a.split}.parquet").filter((pl.col("view") == "rfull") & (pl.col("rank") == 0))
            .select("s1_id", "t_id").unique("t_id", keep="first").collect())
    c1 = (pl.scan_parquet(f"{a.data}/cands_rdense_{a.split}.parquet").filter(pl.col("rank") == 0)
            .select("s1_id", "t_id").unique("t_id", keep="first").collect())
    c1 = c1.join(c0, on="t_id", how="anti"); nfb = len(c1); P = pl.concat([c0, c1]); del c0, c1
    if a.split == "train":
        tr_ids = pl.read_parquet(f"{a.data}/split.parquet").filter(pl.col("split") == "train")["s1_id"]
        P = P.filter(pl.col("s1_id").is_in(tr_ids.implode()))
    si, ti = lookup(P["s1_id"]), lookup(P["t_id"]); np_ = len(P); del P
    Vs = V.select("rnum", "ncw", "cidx"); K = len(CTRY)
    cnt = {k: np.zeros(K * NW, np.int64) for k in ["ins_pos", "ins_neg", "del_pos", "del_neg", "rep_pos", "rep_neg"]}
    Np = {k: np.zeros(K, np.int64) for k in ["pos", "neg", "rep_pos", "rep_neg"]}; sig_hist = np.zeros(4, np.int64)
    for lo in range(0, np_, 2_000_000):
        S, T = Vs[si[lo:lo + 2_000_000]], Vs[ti[lo:lo + 2_000_000]]; m = len(S)
        sig, _, _, _, shared = shift_core(S["rnum"], T["rnum"], m); sig_hist += np.bincount(sig, minlength=4)
        pos, neg = sig == 3, (sig == 0) & shared
        ir, iw, dr, dw, ns, nt = word_diff(S["ncw"], T["ncw"], m)
        c = T["cidx"].to_numpy().astype(np.int64); eq = ns == nt
        for lab, msk in [("pos", pos), ("neg", neg)]:
            Np[lab] += np.bincount(c[msk], minlength=K); Np["rep_" + lab] += np.bincount(c[msk & eq], minlength=K)
            cnt["ins_" + lab] += np.bincount(c[ir] * NW + iw, weights=msk[ir], minlength=K * NW).astype(np.int64)
            cnt["del_" + lab] += np.bincount(c[dr] * NW + dw, weights=msk[dr], minlength=K * NW).astype(np.int64)
            cnt["rep_" + lab] += np.bincount(c[ir] * NW + iw, weights=(msk & eq)[ir], minlength=K * NW).astype(np.int64)
    LR, js = {}, {}
    for tab in ["ins", "del", "rep"]:
        cp_, cn_ = cnt[tab + "_pos"].reshape(K, NW), cnt[tab + "_neg"].reshape(K, NW)
        npos = Np["rep_pos" if tab == "rep" else "pos"][:, None]; nneg = Np["rep_neg" if tab == "rep" else "neg"][:, None]
        lr = np.log((cp_ + 1.0) / (npos + 2.0)) - np.log((cn_ + 1.0) / (nneg + 2.0))
        lr = np.where(cp_ + cn_ >= a.wordlr_min_count, lr, 0.0).astype(np.float32); LR[tab] = lr
        for k, cty in enumerate(CTRY):
            idx = np.nonzero(cp_[k] + cn_[k] >= a.wordlr_min_count)[0]; idx = idx[np.argsort(-lr[k, idx], kind="stable")]
            js.setdefault(cty, {})[tab] = {WORDS[i]: [round(float(lr[k, i]), 4), int(cp_[k, i]), int(cn_[k, i])] for i in idx}
    for k, cty in enumerate(CTRY):
        js[cty]["meta"] = dict(n_pos=int(Np["pos"][k]), n_neg=int(Np["neg"][k]), n_rep_pos=int(Np["rep_pos"][k]),
                               n_rep_neg=int(Np["rep_neg"][k]), alpha=1, min_count=a.wordlr_min_count)
        top = list(js[cty]["ins"].items())[:15]
        log(f"wordlr {cty}: N_pos {Np['pos'][k]:,} N_neg {Np['neg'][k]:,}; top ins: " + ", ".join(f"{w} {v[0]:.2f}" for w, v in top))
    js["_meta"] = dict(split=a.split, n_ref_pairs=int(np_), n_rdense_fallback=int(nfb), sig_hist=sig_hist.tolist(),
                       reference="cands_rsparse rfull rank 0 (fallback rdense rank 0)" + (", train-split references only" if a.split == "train" else ""))
    log(f"word-LR tables built in {time.time() - t0:.0f}s on {np_:,} rank-0 pairs (rdense fallback {nfb:,}); sig hist {sig_hist.tolist()}")
    return LR, js

def load_word_tables(path):
    """--wordlr-table-in: LR arrays [country of this split, word of this split's vocab] from a stored table json. Same float64
    arithmetic as build_word_tables on the stored counts, then float32; words absent from the json (or below its min_count) -> 0."""
    js = json.load(open(path)); widx = {w: i for i, w in enumerate(WORDS)}; LR = {}
    for tab in ["ins", "del", "rep"]:
        cp_, cn_, has = np.zeros((len(CTRY), NW), np.int64), np.zeros((len(CTRY), NW), np.int64), np.zeros((len(CTRY), NW), bool)
        npos, nneg = np.zeros((len(CTRY), 1), np.int64), np.zeros((len(CTRY), 1), np.int64)
        for k, cty in enumerate(CTRY):
            if cty not in js: continue
            m = js[cty]["meta"]; npos[k], nneg[k] = (m["n_rep_pos"], m["n_rep_neg"]) if tab == "rep" else (m["n_pos"], m["n_neg"])
            for w, v in js[cty][tab].items():
                i = widx.get(w)
                if i is not None: cp_[k, i], cn_[k, i], has[k, i] = v[1], v[2], True
        lr = np.log((cp_ + 1.0) / (npos + 2.0)) - np.log((cn_ + 1.0) / (nneg + 2.0))
        LR[tab] = np.where(has, lr, 0.0).astype(np.float32)
        log(f"word-LR table {path} [{tab}]: " + ", ".join(f"{c} {int(has[k].sum()):,} words" for k, c in enumerate(CTRY)))
    return LR, [c for c in CTRY if c in js]

if a.wordlr_table_in:
    LR_TAB, TAB_CTRY = load_word_tables(a.wordlr_table_in)
    bad = [c for c in (a.wordlr_countries.split(",") if a.wordlr_countries else CTRY) if c and c not in TAB_CTRY]
    if bad: raise SystemExit(f"--wordlr-table-in {a.wordlr_table_in} has no table for {bad}; restrict --wordlr-countries")
    WORDLR_JSON = None
else:
    LR_TAB, WORDLR_JSON = build_word_tables()
if not a.bench and not a.wordlr_table_in:
    os.makedirs(f"{a.data}/tables", exist_ok=True)
    tab_out = a.wordlr_table_out or f"{a.data}/tables/wordlr_{a.split}.json"
    json.dump(WORDLR_JSON, open(tab_out, "w"), ensure_ascii=False); log(f"wrote word-LR table {tab_out}")
del WORDLR_JSON
V = V.drop("entity_id")

def signature_feats(S, T, out, n):
    sig, dmin, big, neg, _ = shift_core(S["rnum"], T["rnum"], n)
    out["shift_sig"] = sig; out["shift_d"] = dmin; out["shift_d_big"] = big; out["shift_neg"] = neg

def wordlr_feats(S, T, out, n):
    ir, iw, dr, dw, ns, nt = word_diff(S["ncw"], T["ncw"], n)
    c = S["cidx"].to_numpy().astype(np.int64)
    def agg(r, v):  # r sorted
        k = np.bincount(r, minlength=n); sm = np.bincount(r, weights=v, minlength=n)
        mx = np.zeros(n, np.float32); mn = np.zeros(n, np.float32)
        if len(r):
            st = np.r_[0, np.nonzero(r[1:] != r[:-1])[0] + 1]
            mx[r[st]] = np.maximum.reduceat(v, st); mn[r[st]] = np.minimum.reduceat(v, st)
        return k, mx, mn, sm.astype(np.float32)
    k, mx, mn, sm = agg(ir, LR_TAB["ins"][c[ir], iw])
    out["ins_lr_max"] = mx; out["ins_lr_sum"] = sm; out["ins_lr_min"] = mn; out["n_ins"] = np.minimum(k, 127).astype(np.int8)
    k, mx, _, sm = agg(dr, LR_TAB["del"][c[dr], dw])
    out["del_lr_max"] = mx; out["del_lr_sum"] = sm; out["n_del"] = np.minimum(k, 127).astype(np.int8)
    eq = (ns == nt)[ir]
    _, mx, _, _ = agg(ir[eq], LR_TAB["rep"][c[ir[eq]], iw[eq]])
    out["rep_lr_max"] = mx
    out["dba_flag"] = T["t_dba"].to_numpy()

TIMES = {}
def tick(k, t):
    TIMES[k] = TIMES.get(k, 0.0) + time.time() - t; return time.time()

def features(chunk: pl.DataFrame) -> pl.DataFrame:
    n = len(chunk); t = time.time()
    si, ti = lookup(chunk["s1_id"]), lookup(chunk["t_id"])
    S = V[si]; T = V[ti]
    t = tick("gather", t)
    o = {}
    sim(S, T, "name_words", "name_latin", [fuzz.ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio,
                                           JaroWinkler.normalized_similarity],
        ["nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw"], o)
    sim(S, T, "name_core", "name_core", [fuzz.token_set_ratio, JaroWinkler.normalized_similarity], ["nc_tset", "nc_jw"], o)
    sim(S, T, "name_compact", "name_compact", [Levenshtein.normalized_similarity, fuzz.partial_ratio], ["ncomp_lev", "ncomp_partial"], o)
    sim(S, T, "name_leet", "name_leet", [Levenshtein.normalized_similarity, fuzz.partial_ratio], ["nleet_lev", "nleet_partial"], o)
    sim(S, T, "name_skel", "name_skel", [fuzz.ratio], ["nskel_ratio"], o)
    t = tick("name_rf", t)
    sim(S, T, "addr_words", "addr_words", [fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_ratio],
        ["aw_tset", "aw_tsort", "aw_partial"], o)
    sim(S, T, "addr_core", "addr_core", [fuzz.token_set_ratio], ["ac_tset"], o)
    t = tick("addr_rf", t)
    o["eq_name_core"] = ((S["h_core"] == T["h_core"]) & (S["name_core"].str.len_bytes() > 0)).cast(pl.Int8).to_numpy()
    o["eq_name_sorted"] = (S["h_sorted"] == T["h_sorted"]).cast(pl.Int8).to_numpy()
    o["eq_name_compact"] = ((S["h_compact"] == T["h_compact"]) & (S["name_compact"].str.len_bytes() > 0)).cast(pl.Int8).to_numpy()
    sw, tw = S["nwords"].to_numpy(), T["nwords"].to_numpy()
    o["s1_nwords"] = sw; o["t_nwords"] = tw; o["nwords_diff"] = (sw.astype(np.int16) - tw).astype(np.int8)
    token_feats(S["name_core_tok"], T["name_core_tok"], NC_IDF, NC_GEN, "nc", o, n)
    token_feats(S["addr_core_tok"], T["addr_core_tok"], AC_IDF, None, "ac", o, n)
    t = tick("tokens", t)
    sa, ta = S["alen"].to_numpy(), T["alen"].to_numpy()
    o["addr_len_ratio"] = np.where((sa == 0) | (ta == 0), -1.0, np.minimum(sa, ta) / np.maximum(np.maximum(sa, ta), 1)).astype(np.float32)
    o["s1_addr_empty"] = (sa == 0).astype(np.int8); o["t_addr_empty"] = (ta == 0).astype(np.int8)
    number_feats(S["addr_numbers"], T["addr_numbers"], o, n)
    house_feats(S["hn"], T["hn"], o)
    t = tick("numbers", t)
    signature_feats(S, T, o, n)
    t = tick("shift", t)
    wordlr_feats(S, T, o, n)
    t = tick("wordlr", t)
    o["s1_name_core_count"] = S["c_core"].to_numpy(); o["s1_name_compact_count"] = S["c_compact"].to_numpy()
    o["s1_addr_core_count"] = S["c_addr"].to_numpy(); o["s1_num_addr_count"] = S["c_numaddr"].to_numpy()
    o["t_name_core_s1_count"] = T["c_core"].to_numpy(); o["t_addr_core_s1_count"] = T["c_addr"].to_numpy()
    o["t_num_addr_s1_count"] = T["c_numaddr"].to_numpy()
    o["t_has_indic"] = T["has_indic"].to_numpy(); o["t_is_domain_form"] = T["t_dom"].to_numpy()
    feats = pl.DataFrame(o)
    clash = set(feats.columns) & set(chunk.columns)
    if clash: raise SystemExit(f"feature columns already in input: {sorted(clash)}")
    for c, dt in feats.schema.items():
        if dt.is_float(): assert not feats[c].is_nan().any() and not feats[c].is_null().any(), c
    res = chunk.hstack(feats.get_columns())
    tick("assemble", t)
    return res

# ============================================ driver =============================================================
WORDLR_COLS = ["ins_lr_max", "ins_lr_sum", "ins_lr_min", "n_ins", "del_lr_max", "del_lr_sum", "n_del", "rep_lr_max"]
if a.wordlr_only:
    if not a.pairs: log("table only: done"); sys.exit(0)
    pf = pq.ParquetFile(a.pairs); n_in = pf.metadata.num_rows; sch = pf.schema_arrow
    miss = [c for c in WORDLR_COLS + ["s1_id", "t_id", "country"] if c not in sch.names]
    if miss: raise SystemExit(f"--wordlr-only: {a.pairs} lacks {miss} (must be a pair_features output)")
    ctys = [c for c in a.wordlr_countries.split(",") if c] or CTRY
    tmp = a.out + ".tmp"; writer = pq.ParquetWriter(tmp, sch, compression="zstd"); done = n_re = 0; tc = time.time()
    for batch in pf.iter_batches(batch_size=a.chunk):
        tb = pa.Table.from_batches([batch], schema=sch)
        rows = np.nonzero(pc.is_in(tb["country"], value_set=pa.array(ctys, type=sch.field("country").type)).to_numpy(zero_copy_only=False))[0]
        if len(rows):
            sub = pl.from_arrow(tb.select(["s1_id", "t_id"]).take(pa.array(rows)))
            S, T = V[lookup(sub["s1_id"])], V[lookup(sub["t_id"])]; o = {}
            wordlr_feats(S, T, o, len(sub))
            for c in WORDLR_COLS:  # scatter new values into the recomputed rows; all other rows keep their exact values
                col = tb[c].to_numpy(); ty = col.dtype; col = col.copy(); col[rows] = o[c].astype(ty)
                tb = tb.set_column(sch.get_field_index(c), sch.field(c), pa.array(col, type=sch.field(c).type))
            n_re += len(rows)
        writer.write_table(tb, row_group_size=min(a.chunk, 1_000_000)); done += len(tb)
        log(f"{done:,}/{n_in:,} rows, {n_re:,} word-LR recomputed ({(time.time() - tc) / done * 1e6:.1f}s per 1M)")
        del tb, batch
    writer.close(); os.replace(tmp, a.out)
    log(f"wrote {a.out}: {done:,} rows; word-LR columns recomputed for {n_re:,} rows of {ctys}")
    sys.exit(0)
pf = pq.ParquetFile(a.pairs)
n_in = pf.metadata.num_rows
if a.bench:
    target = 1_000_000
    first = pl.from_arrow(next(pf.iter_batches(batch_size=min(target, n_in))))
    reps = -(-target // len(first))
    df = pl.concat([first] * reps).head(target)
    log(f"bench: {len(df):,} pairs (input {n_in:,} rows, tiled x{reps})")
    t = time.time(); f = features(df); dt = time.time() - t
    log(f"bench features: {dt:.1f}s per 1M pairs -> {dt * 100 / 60:.1f} min per 100M (+ setup {t - T0:.0f}s)")
    print("stage times:", {k: round(v, 1) for k, v in TIMES.items()})
    print(f"added {f.width - df.width} columns")
    sys.exit(0)

tmp = a.out + ".tmp"; writer = None; done = 0; tc = time.time()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
for batch in pf.iter_batches(batch_size=a.chunk):
    chunk = pl.from_arrow(batch) if isinstance(batch, pa.RecordBatch) else pl.from_arrow(batch)
    if isinstance(chunk, pl.Series): chunk = chunk.to_frame()
    res = features(chunk).to_arrow()
    if writer is None: writer = pq.ParquetWriter(tmp, res.schema, compression="zstd")
    writer.write_table(res, row_group_size=min(a.chunk, 1_000_000))
    done += len(chunk)
    log(f"{done:,}/{n_in:,} rows ({(time.time() - tc) / done * 1e6:.1f}s per 1M)")
    del res, chunk
writer.close(); os.replace(tmp, a.out)
print("stage times:", {k: round(v, 1) for k, v in TIMES.items()})
log(f"wrote {a.out}: {done:,} rows")
