"""Research 49: label-free role transplant US->France and India->France of the labelled CE fold sets (v7fA / v7fB).
Uses data/cefold_v7fr/role_maps.json (fr_roles.py). ONE substitution set per pair, applied to both records (a, b).
Constraints (astra + coordinator): ALL digits and all punctuation outside role tokens preserved byte-for-byte (no postal
randomisation); role-restricted, case-preserving, token-boundary substitutions; injective within each role across different
source values; aliases of one source value -> aliases of one French value; person-name tokens untouched.
Per-pair checks; a pair failing any is REVERTED (original text, status column says which role failed):
  role relation (equal / subset / superset / overlap / disjoint of the per-record value sets) for every role before vs after;
  new-token collisions (an introduced French token equal to an untouched token of the pair);
  digit-run sequence per record identical; name / address / string equality (exact, casefold, accent-fold) identical.
Outputs data/cefold_v7fr/ce_train_hard_v7f{A,B}_fr.parquet (a,b original + a_fr,b_fr + tp_status + tp_nsub),
ce_train_hard_v7f{A,B}_frmix.parquet (original set + transplanted 'ok' copies, column origin), checks json."""
import re, json, sys, os, random, hashlib, collections, unicodedata, time
from multiprocessing import Pool
import polars as pl
from rapidfuzz import fuzz
sys.path.insert(0, "."); from src.text_views import strip_accents
O = "data/cefold_v7fr"; RM = json.load(open(f"{O}/role_maps.json")); SEED = 49
def key(w): return strip_accents(w).strip()
def fold_char(ch):
    k = strip_accents(ch); return k if len(k) == 1 else ch.lower()
def fold(s): return "".join(fold_char(c) for c in s)                     # same length as s
def norm_city(k): return re.sub(r"\s+", " ", k.replace("st.-", "saint-").replace("st-", "saint-").replace("-", " ")).strip()
def alt(keys): return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
FR = RM["France"]
CITIES = FR["cities"]; DEPT_REG = FR["dept_region"]; DEPT_D = FR["dept_display"]; REG_D = FR["region_display"]
REGIONS = sorted({c["region"] for c in CITIES if c["region"]}); REG_W = {r: FR["region_n"].get(r, 1) for r in REGIONS}
SURF = FR["surfaces"]
def region_alt(disp):  # second alias surface of a region: hyphens <-> spaces
    return disp.replace("-", " ") if "-" in disp else disp.replace(" ", "-")
US_ST = RM["US_ST"]; US_FULL = {v.lower(): k for k, v in US_ST.items()}
IN_ST = RM["IN_ST"]; IN_FULL = {k.lower(): k for k in IN_ST}; IN_AB = {v.lower(): k for k, v in IN_ST.items()}; IN_NAT = RM["IN_NATIVE_ST"]
P_STZIP = re.compile(r"^([A-Za-z]{2})(\s+\d{5}(?:-\d{4})?)$")
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
def casefit(src, rep):
    letters = [c for c in src if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(letters) > 1: return rep.upper()
    if letters and all(c.islower() for c in letters): return rep.lower()
    if len(letters) == 1 and letters[0].isupper(): return rep[:1].upper() + rep[1:]
    return rep[:1].upper() + rep[1:] if rep[:1].islower() else rep
def accent_transfer(src, rep):
    """re-apply the source's NOISE accents (marks not in its accent-free key) to rep at relative positions."""
    marks = []
    for i, ch in enumerate(src):
        d = unicodedata.normalize("NFD", ch)
        if len(d) > 1 and d[0].isascii(): marks.append((i, d[1:]))
    if not marks: return rep
    out = list(rep)
    for i, m in marks:
        p = min(len(out) - 1, round(i * len(out) / max(1, len(src))))
        for q in list(range(p, len(out))) + list(range(p - 1, -1, -1)):
            if out[q].isascii() and out[q].isalpha(): out[q] = unicodedata.normalize("NFC", out[q] + m); break
    return "".join(out)
def render(src, rep): return accent_transfer(src, casefit(src, rep))
def edit_transfer(s, v, t):
    """s = clean source key, v = noisy variant surface, t = French target: replay the s->v char edits on t at scaled positions."""
    vk = fold(v); ops = __import__("difflib").SequenceMatcher(None, s, vk, autojunk=False).get_opcodes(); out = []
    for tag, i1, i2, j1, j2 in ops:
        a, b = round(i1 * len(t) / max(1, len(s))), round(i2 * len(t) / max(1, len(s)))
        if tag == "equal": out.append(t[a:b])
        elif tag in ("replace", "insert"): out.append(vk[j1:j2])
    r = "".join(out)
    if key(r) == key(t): r = t + vk[-1:]
    return render(v, r)

class Ctx:
    def __init__(s, country):
        s.c = country; S = RM[country]
        s.legal = S["legal_vm"]; s.P_LEGAL = re.compile(r"(?<![\w.])([\[\(]*)(" + alt(s.legal) + r")([\]\)]*)[\s.,;:!]*$")
        s.street = S["street_vm"]; s.unit = S["unit_vm"]; s.name_map = S["name_map"]; s.name_role = S["name_role"]
        s.street_group = {v: g for g, vs, _ in S["street_rank"] for v in vs}
        s.P_UNIT = re.compile(r"(?<![\w.])(" + alt(s.unit) + r")\.?(?=\s*(?:#|\d|no\b|n°|[a-z]\b))")
CTX = {c: Ctx(c) for c in ("US", "India")}

def parse_addr(addr, c):
    """components with offsets; state idx list; city idx per state."""
    comps, pos = [], 0
    for part in addr.split(","):
        st = part.strip(); off = pos + (len(part) - len(part.lstrip()))
        comps.append((off, st)); pos += len(part) + 1
    states = {}
    for i, (off, st) in enumerate(comps):
        sid = alias = None; span = (off, off + len(st))
        if c == "US":
            m = P_STZIP.match(st)
            if st.upper() in US_ST and len(st) == 2: sid, alias = st.upper(), "ab"
            elif st.lower() in US_FULL: sid, alias = US_FULL[st.lower()], "full"
            elif m and m.group(1).upper() in US_ST: sid, alias, span = m.group(1).upper(), "ab", (off, off + 2)
        else:
            if st.lower() in IN_FULL: sid, alias = IN_FULL[st.lower()], "full"
            elif st.lower() in IN_AB and len(st) == 2: sid, alias = IN_AB[st.lower()], "ab"
            elif st in IN_NAT: sid, alias = IN_NAT[st], "nat"
        if sid: states[i] = (sid, alias, span, st[span[0] - off:span[1] - off])
    cities = {}
    for i, (sid, _, _, _) in states.items():
        for j in (i - 1, i + 1):
            if 0 <= j < len(comps) and j not in states and comps[j][1] and not any(ch.isdigit() for ch in comps[j][1]) and len(comps[j][1]) < 40:
                cities[j] = sid; break
    return comps, states, cities

def relation(x, y):
    if not x and not y: return "none"
    if x == y: return "eq"
    if x <= y: return "sub"
    if y <= x: return "sup"
    return "ovl" if x & y else "dis"

LAST_UNIT = []
VALUE_KEYED = False   # record-level mode (valfr_rec): every choice is a deterministic function of the SOURCE VALUE, not of the pair,
                      # so one record gets one transformed string on every edge (distinct values may collide; measured, not reverted)
def vr(tag): return random.Random(hashlib.md5(f"{SEED}|VK|{tag}".encode()).hexdigest())
def transplant(a, b, c, rng):
    """returns (a2, b2, nsub, status, roles_used, intro_keys)"""
    X = CTX[c]; recs = [a.partition(" | ") for a in (a, b)]
    names, addrs = [r[0] for r in recs], [r[2] for r in recs]; seps = [r[1] for r in recs]
    spans = [[[], []], [[], []]]  # rec -> field(0 name,1 addr) -> list of (s,e,rep,role,src_id,tgt_id)
    def add(r, f, s, e, rep, role, sid, tid):
        for (s2, e2, *_ ) in spans[r][f]:
            if s < e2 and s2 < e: return False
        spans[r][f].append((s, e, rep, role, sid, tid)); return True
    # ---- states + cities (pair-level allocation) ----
    P = [parse_addr(addrs[r], c) for r in (0, 1)]
    sids = []
    for r in (0, 1):
        for i, (sid, *_ ) in P[r][1].items():
            if sid not in sids: sids.append(sid)
    city_src = []  # (key, sid)
    for r in (0, 1):
        comps, states, cities = P[r]
        for j, sid in cities.items():
            k = key(comps[j][1])
            if not any(k == k2 for k2, _ in city_src): city_src.append((k, sid))
    status = "ok"; st_map = {}; city_map = {}
    if sids:
        if len(sids) > len(REGIONS) and not VALUE_KEYED: return a, b, 0, "revert:state_overflow", [], []
        regs = []; pool = list(REGIONS)
        for _ in sids:
            if VALUE_KEYED: continue
            r_ = rng.choices(pool, [REG_W[x] for x in pool])[0]; regs.append(r_); pool.remove(r_)
        if VALUE_KEYED: regs = [vr(f"reg|{c}|{sid}").choices(REGIONS, [REG_W[x] for x in REGIONS])[0] for sid in sids]
        for sid, r_ in zip(sids, regs):
            depts = [d for d, rr in DEPT_REG.items() if rr == r_]
            need = [k for k, s_ in city_src if s_ == sid]
            # canonical clusters: fuzzy-merge noisy city variants onto the first clean key
            clusters = []
            for k in need:
                for cl in clusters:
                    if fuzz.ratio(k, cl[0]) >= 85 or k.startswith(cl[0] + " ") or cl[0].startswith(k + " "): cl.append(k); break
                else: clusters.append([k])
            ok_d = [d for d in depts if len([x for x in CITIES if x["dept"] == d]) >= len(clusters)]
            if VALUE_KEYED: ok_d = depts
            if not ok_d: return a, b, 0, "revert:city_overflow", [], []
            d = (vr(f"dep|{c}|{sid}") if VALUE_KEYED else rng).choices(ok_d, [FR["dept_n"].get(x, 1) for x in ok_d])[0]
            st_map[sid] = (d, r_)
            cpool = [x for x in CITIES if x["dept"] == d]
            for cl in clusters:
                if VALUE_KEYED:
                    cp = [x for x in CITIES if x["dept"] == d]; fc = vr(f"city|{c}|{sid}|{cl[0]}").choices(cp, [x["n"] for x in cp])[0]
                    vs = fc["variants"]; disp = vr(f"disp|{fc['key']}").choices([v for v, _ in vs], [n for _, n in vs])[0]
                else:
                    fc = rng.choices(cpool, [x["n"] for x in cpool])[0]; cpool.remove(fc)
                    vs = fc["variants"]; disp = rng.choices([v for v, _ in vs], [n for _, n in vs])[0]
                for k in cl: city_map[k] = (cl[0], disp)
    for r in (0, 1):
        comps, states, cities = P[r]
        for i, (sid, alias, (s, e), surf) in states.items():
            d, r_ = st_map[sid]
            if alias == "ab": rep, tid = DEPT_D[d], "d:" + d
            elif alias == "full": rep, tid = REG_D[r_], "r:" + r_
            else: rep, tid = region_alt(REG_D[r_]), "ra:" + r_
            add(r, 1, s, e, render(surf, rep) if alias != "ab" else (rep.upper() if surf.isupper() and len(surf) > 2 else rep), "state", sid + ":" + alias, tid)
        for j, sid in cities.items():
            off, st = comps[j]; k = key(st); clean, disp = city_map[k]
            rep = render(st, disp) if k == clean else edit_transfer(clean, st, disp)
            add(r, 1, off, off + len(st), rep, "city", k, key(rep))
    # city names elsewhere (names and other components), token-bounded, only clean keys
    clean_cities = {cl: disp for k, (cl, disp) in city_map.items() if k == cl and len(cl) >= 4}
    if clean_cities:
        pc = re.compile(r"(?<![\w])(" + alt(clean_cities) + r")(?![\w])")
        for r in (0, 1):
            for f, txt in ((0, names[r]), (1, addrs[r])):
                for m in pc.finditer(fold(txt)):
                    src = txt[m.start():m.end()]; add(r, f, m.start(), m.end(), render(src, clean_cities[m.group(1)]), "city", m.group(1), key(clean_cities[m.group(1)]))
    # ---- legal form (trailing, in the name) ----
    for r in (0, 1):
        fn = fold(names[r]); m = X.P_LEGAL.search(fn)
        if m:
            s, e = m.start(2), m.end(2); v = m.group(2); src = names[r][s:e]
            add(r, 0, s, e, render(src, X.legal[v]), "legal", v, X.legal[v])
    # ---- street types / units (address) ----
    for r in (0, 1):
        comps, states, cities = P[r]; fa = fold(addrs[r])
        for j, (off, st) in enumerate(comps):
            if j in states or j in cities or not st: continue
            ws = [(m.start() + off, m.end() + off) for m in re.finditer(r"\S+", st)]
            for q, (s, e) in enumerate(ws):
                tok = fa[s:e]; dot = tok.endswith("."); tk = tok.rstrip(".")
                if tk in X.street and q > 0 and any(ch.isalpha() for ch in fa[ws[q - 1][0]:ws[q - 1][1]]):
                    rep = X.street[tk] + ("." if dot else ""); src = addrs[r][s:e]
                    last = q == len(ws) - 1
                    body = [(s2, e2) for (s2, e2) in ws[:q] if not any(ch.isdigit() for ch in fa[s2:e2])]
                    lead = [(s2, e2) for (s2, e2) in ws[:q] if any(ch.isdigit() for ch in fa[s2:e2])]
                    # reorder 'NUM NAME TYPE' -> 'NUM TYPE NAME' when the component is exactly that shape
                    if last and body and lead == ws[:len(lead)] and body == ws[len(lead):q]:
                        bs = body[0][0]; seg = addrs[r][bs:e]; name_part = addrs[r][bs:body[-1][1]]
                        add(r, 1, bs, e, render(src, rep) + " " + name_part, "street", tk + ("." if dot else ""), key(rep))
                    else:
                        add(r, 1, s, e, render(src, rep), "street", tk + ("." if dot else ""), key(rep))
            for m in X.P_UNIT.finditer(fa[off:off + len(st)]):
                s, e = m.start(1) + off, m.end(1) + off; v = m.group(1); src = addrs[r][s:e]
                add(r, 1, s, e, render(src, X.unit[v]), "unit", v, X.unit[v])
    # ---- name words (category / generator / noise-insertion / function words) ----
    surf_choice = {}
    def fr_surface(k):
        if k not in surf_choice:
            vs = SURF.get(k)
            surf_choice[k] = (vr(f"surf|{k}") if VALUE_KEYED else rng).choices([v for v, _ in vs], [n for _, n in vs])[0] if vs else k
        return surf_choice[k]
    for r in (0, 1):
        fn = fold(names[r])
        for m in WORD.finditer(fn):
            k = m.group(0)
            if k in X.name_map:
                src = names[r][m.start():m.end()]; t = fr_surface(X.name_map[k])
                add(r, 0, m.start(), m.end(), render(src, t), X.name_role.get(k, "name"), k, key(t))
    # ---- typo'd variants of substituted single-token source words (pair-level, edit transfer) ----
    subs = {}
    for r in (0, 1):
        for f in (0, 1):
            for (s, e, rep, role, sid, tid) in spans[r][f]:
                if role in ("category", "gen_pure", "gen_dual", "gen2", "noise_ins", "street", "legal") and " " not in sid and len(sid) >= 4:
                    subs[sid.rstrip(".")] = (role, key(rep) if role != "street" else tid, rep)
    if subs:
        for r in (0, 1):
            for f, txt in ((0, names[r]), (1, addrs[r])):
                ft = fold(txt)
                for m in WORD.finditer(ft):
                    k = m.group(0)
                    if len(k) < 4 or k in subs or k in X.name_map: continue
                    if any(s <= m.start() < e for (s, e, *_ ) in spans[r][f]): continue
                    best = max(subs, key=lambda x: fuzz.ratio(k, x)); sc = fuzz.ratio(k, best)
                    if sc >= 80:
                        role, tid, rep = subs[best]; tgt = key(rep).rstrip(".")
                        add(r, f, m.start(), m.end(), edit_transfer(best, txt[m.start():m.end()], tgt), "typo_" + role, "~" + k, None)
    # ---- build ----
    out, intro, nsub = [], [], 0; unit_b, unit_a = [], []   # unit_*: every substituted span as ONE token (diagnostic only)
    U = lambda z: re.sub(r"[\s\-]+", "_", z.strip())
    for r in (0, 1):
        parts, pb, pa = [], [], []
        for f, txt in ((0, names[r]), (1, addrs[r])):
            sp = sorted(spans[r][f]); o, ob, oa, p = [], [], [], 0
            for (s, e, rep, *_ ) in sp:
                o.append(txt[p:s]); o.append(rep); ob += [txt[p:s], U(txt[s:e])]; oa += [txt[p:s], U(rep)]; p = e; nsub += 1; intro.append(rep)
            o.append(txt[p:]); ob.append(txt[p:]); oa.append(txt[p:]); parts.append("".join(o)); pb.append("".join(ob)); pa.append("".join(oa))
        out.append(parts[0] + seps[r] + parts[1]); unit_b.append(pb[0] + seps[r] + pb[1]); unit_a.append(pa[0] + seps[r] + pa[1])
    if os.environ.get("TP_UNIT"): LAST_UNIT[:] = [unit_b, unit_a]
    if nsub == 0: return a, b, 0, "nochange", [], []
    # ---- checks ----
    fails = []
    roles = sorted({sp[3] for r in (0, 1) for f in (0, 1) for sp in spans[r][f]})
    for role in roles:
        bs = [set(sp[4] for f in (0, 1) for sp in spans[r][f] if sp[3] == role) for r in (0, 1)]
        as_ = [set((sp[5] if sp[5] is not None else key(sp[2])) for f in (0, 1) for sp in spans[r][f] if sp[3] == role) for r in (0, 1)]
        if relation(*bs) != relation(*as_): fails.append(role)
    # collisions: introduced tokens equal to untouched tokens of the pair (that the source tokens did not equal)
    unt, srck = set(), set()
    for r in (0, 1):
        for f, txt in ((0, names[r]), (1, addrs[r])):
            ft = fold(txt); cov = [(s, e) for (s, e, *_ ) in spans[r][f]]
            for m in WORD.finditer(ft):
                if not any(s <= m.start() < e for s, e in cov): unt.add(m.group(0))
                else: srck.add(m.group(0))
    introk = {w for x in intro for w in WORD.findall(fold(x))}
    coll = {w for w in (introk & unt) - (srck & unt) if len(w) >= 3}
    if os.environ.get("TP_DEBUG") and coll: print("COLL", coll)
    if coll: fails.append("collision")
    for r, (x0, x1) in enumerate(((a, out[0]), (b, out[1]))):
        if re.findall(r"\d+", x0) != re.findall(r"\d+", x1): fails.append("numeric"); break
    def eqsig(p, q):
        pn, _, pa = p.partition(" | "); qn, _, qa = q.partition(" | ")
        return tuple(f(u) == f(v) for u, v in ((pn, qn), (pa, qa), (p, q)) for f in (lambda z: z, str.casefold, key))
    if eqsig(a, b) != eqsig(*out): fails.append("field_eq")
    if fails: return a, b, 0, "revert:" + "+".join(sorted(set(fails))), roles, []
    return out[0], out[1], nsub, "ok", roles, list(introk)

def work(args):
    rows, seedtag = args; res = []
    for s1, t, c, a, b in rows:
        rng = random.Random(hashlib.md5(f"{SEED}|{seedtag}|{s1}|{t}".encode()).hexdigest())
        try: res.append(transplant(a, b, c, rng))
        except Exception as ex: res.append((a, b, 0, "revert:error:" + type(ex).__name__, [], []))
    return res

def sig(a, b):
    na, _, aa = a.partition(" | "); nb, _, ab = b.partition(" | ")
    n1, n2 = re.findall(r"\d+", aa), re.findall(r"\d+", ab)
    d = abs(int(n1[0]) - int(n2[0])) if n1 and n2 and len(n1[0]) < 10 and len(n2[0]) < 10 else -1
    ta, tb = a.split(), b.split(); ka, kb = set(WORD.findall(fold(a))), set(WORD.findall(fold(b)))
    jac = len(ka & kb) / max(1, len(ka | kb))
    return dict(name_eq=na == nb, name_eq_ci=key(na) == key(nb), addr_eq=aa == ab, addr_eq_ci=key(aa) == key(ab), nums_eq=n1 == n2,
                num_diff=(min(d, 30) if d >= 0 else -1), tokdiff=min(abs(len(ta) - len(tb)), 6), jac=round(jac * 10) / 10, string_eq=a == b)

if __name__ == "__main__":
    fr_vocab = collections.Counter()
    te = pl.read_parquet("data/test_records.parquet").filter(pl.col("country") == "France")
    for s in te["business_name"].fill_null("").to_list() + te["business_address"].fill_null("").to_list():
        for w in WORD.findall(fold(s)): fr_vocab[w] += 1
    FRV = {w for w, n in fr_vocab.items() if n >= 5}; del te
    manifest = dict(role_maps_sha256=hashlib.sha256(open(f"{O}/role_maps.json", "rb").read()).hexdigest(), seed=SEED, folds={})
    for fold_name in sys.argv[1:] or ["A", "B"]:
        t0 = time.time(); src = f"data/cefold_v7/ce_train_hard_v7f{fold_name}.parquet"
        d = pl.read_parquet(src); rows = list(zip(d["s1_id"], d["t_id"], d["country"], d["a"], d["b"]))
        CH = 20000; chunks = [(rows[i:i + CH], fold_name) for i in range(0, len(rows), CH)]
        with Pool(7) as pool: res = [x for part in pool.imap(work, chunks) for x in part]
        A2 = [x[0] for x in res]; B2 = [x[1] for x in res]; NS = [x[2] for x in res]; ST_ = [x[3] for x in res]
        out = d.with_columns(a_fr=pl.Series(A2), b_fr=pl.Series(B2), tp_nsub=pl.Series(NS, dtype=pl.Int16), tp_status=pl.Series(ST_))
        p = f"{O}/ce_train_hard_v7f{fold_name}_fr.parquet"; out.write_parquet(p)
        mix = pl.concat([d.with_columns(origin=pl.lit("orig")),
                         out.filter(pl.col("tp_status") == "ok").select(d.columns[:5] + ["a_fr", "b_fr", "split"]).rename({"a_fr": "a", "b_fr": "b"})
                            .select(d.columns).with_columns(origin=pl.lit("fr"))])
        pm = f"{O}/ce_train_hard_v7f{fold_name}_frmix.parquet"; mix.write_parquet(pm)
        # matched-exposure control: original + UNMAPPED duplicate of exactly the rows that got an 'ok' transplanted copy
        dup = pl.concat([d.with_columns(origin=pl.lit("orig")),
                         out.filter(pl.col("tp_status") == "ok").select(d.columns).with_columns(origin=pl.lit("dup"))])
        pdup = f"{O}/ce_train_hard_v7f{fold_name}_dup.parquet"; dup.write_parquet(pdup)
        # ---- checks ----
        lab = d["label"].to_list(); cty = d["country"].to_list(); A0, B0 = d["a"].to_list(), d["b"].to_list()
        stat = collections.Counter(ST_); stat_c = collections.Counter((c, s.split(":")[0]) for c, s in zip(cty, ST_))
        rev_role = collections.Counter()
        for s in ST_:
            if s.startswith("revert:"):
                for rl in s[7:].split("+"): rev_role[rl] += 1
        ok = [i for i, s in enumerate(ST_) if s == "ok"]
        newly_ident_neg = sum(1 for i in ok if lab[i] == 0 and A2[i] == B2[i] and A0[i] != B0[i])
        newly_ident_pos = sum(1 for i in ok if lab[i] == 1 and A2[i] == B2[i] and A0[i] != B0[i])
        sig_same = collections.Counter(); marg = collections.Counter(); rng = random.Random(0)
        smp = ok if len(ok) <= 400000 else rng.sample(ok, 400000)
        for i in smp:
            s0, s1_ = sig(A0[i], B0[i]), sig(A2[i], B2[i])
            for k in s0:
                sig_same[k] += s0[k] == s1_[k]; marg[(k, "before", str(s0[k]))] += 1; marg[(k, "after", str(s1_[k]))] += 1
            sig_same["ALL_exact_signature"] += s0 == s1_
            sig_same["ALL_excl_tokdiff_jac"] += all(s0[k] == s1_[k] for k in s0 if k not in ("tokdiff", "jac"))
        introduced = collections.Counter(); allb = [0, 0]; alla = [0, 0]
        for i in smp[:200000]:
            for w in res[i][5]: introduced[w in FRV] += 1
            for x, acc in ((A0[i] + " " + B0[i], allb), (A2[i] + " " + B2[i], alla)):
                ws = WORD.findall(fold(x)); acc[0] += sum(w in FRV for w in ws); acc[1] += len(ws)
        role_use = collections.Counter(rl for i in ok for rl in res[i][4])
        chk = dict(fold=fold_name, source=src, rows=len(d), status=dict(stat.most_common()), status_by_country={f"{a}|{b}": v for (a, b), v in sorted(stat_c.items())},
                   reverted_pairs_by_role=dict(rev_role.most_common()), ok_pairs_using_role=dict(role_use.most_common()),
                   newly_identical_negatives=newly_ident_neg, newly_identical_positives=newly_ident_pos,
                   signature_sample=len(smp), signature_identical_share={k: v / len(smp) for k, v in sig_same.items()},
                   signature_marginals={f"{a}|{b}|{c}": v for (a, b, c), v in sorted(marg.items())},
                   introduced_token_in_french_vocab_share=introduced[True] / max(1, sum(introduced.values())),
                   all_token_in_french_vocab_share_before=allb[0] / max(1, allb[1]), all_token_in_french_vocab_share_after=alla[0] / max(1, alla[1]),
                   mean_substitutions_ok=sum(NS[i] for i in ok) / max(1, len(ok)), seconds=time.time() - t0,
                   sha256_fr=hashlib.sha256(open(p, "rb").read()).hexdigest(), sha256_frmix=hashlib.sha256(open(pm, "rb").read()).hexdigest(),
                   frmix_rows=mix.height, dup_rows=dup.height, sha256_dup=hashlib.sha256(open(pdup, "rb").read()).hexdigest(), frmix_origin=dict(mix["origin"].value_counts().iter_rows()))
        json.dump(chk, open(f"{O}/checks_v7f{fold_name}.json", "w"), indent=1); manifest["folds"][fold_name] = {k: v for k, v in chk.items() if k != "signature_marginals"}
        print(json.dumps({k: v for k, v in chk.items() if k != "signature_marginals"}, indent=1), flush=True)
        # 30-pair random sample of changed ok pairs
        ch = [i for i in ok if A2[i] != A0[i] or B2[i] != B0[i]]; random.Random(7).shuffle(ch)
        with open(f"{O}/sample30_v7f{fold_name}.txt", "w") as fh:
            for i in ch[:30]:
                fh.write(f"[{cty[i]} label={lab[i]} pool={d['pool'][i]}]\n  A0: {A0[i]}\n  B0: {B0[i]}\n  A*: {A2[i]}\n  B*: {B2[i]}\n")
    old = json.load(open(f"{O}/manifest.json")) if os.path.exists(f"{O}/manifest.json") else {}
    old.setdefault("transplant", {}).update(manifest); json.dump(old, open(f"{O}/manifest.json", "w"), indent=1)
