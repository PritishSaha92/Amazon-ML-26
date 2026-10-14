"""Research 49: LABEL-FREE role tables US->France and India->France (no labels, no model scores).
Frequencies: France from UNLABELLED test records (country=France); US/India from train records (text only).
Role classes are declared lexica (annotation of ROLE, not of labels); mapping = identity-first (same key exists in the French
role list), then frequency rank within role. Injective within every role; aliases of one source value (e.g. LLC / L.L.C.)
map to aliases of ONE French value (variant rank -> variant rank). Output data/cefold_v7fr/role_maps.json."""
import re, json, collections, sys, hashlib, statistics
import polars as pl
sys.path.insert(0, "."); from src.text_views import strip_accents
def key(w): return strip_accents(w).strip()
OUT = "data/cefold_v7fr/role_maps.json"
TOKRE = re.compile(r"[^\s,;:()\[\]{}|/\"<>]+")
def toks(s): return [t.strip("-'’*+!?") for t in TOKRE.findall(s or "")]

te = pl.read_parquet("data/test_records.parquet").filter(pl.col("country") == "France")
tr = pl.read_parquet("data/train_records.parquet")
FRN, FRA = te["business_name"].fill_null("").to_list(), te["business_address"].fill_null("").to_list()
src = {c: tr.filter(pl.col("country") == c).sample(1_500_000, seed=0) for c in ("US", "India")}
SN = {c: d["business_name"].fill_null("").to_list() for c, d in src.items()}
SA = {c: d["business_address"].fill_null("").to_list() for c, d in src.items()}

def name_counts(names):
    k, surf = collections.Counter(), collections.defaultdict(collections.Counter)
    for n in names:
        for t in toks(n):
            if not t: continue
            kk = key(t); k[kk] += 1; surf[kk][t.lower()] += 1
    return k, surf
def addr_counts(addrs):
    k = collections.Counter()
    for a in addrs:
        for t in toks(a):
            if t and not any(ch.isdigit() for ch in t): k[key(t).rstrip(".")] += 1
    return k
FRK, FRSURF = name_counts(FRN); FRAK = addr_counts(FRA)
SK = {c: name_counts(SN[c])[0] for c in SN}; SAK = {c: addr_counts(SA[c]) for c in SA}
# trailing-phrase counts for legal forms
def trail_counts(names, maxw=4):
    k = collections.Counter()
    for n in names:
        ws = [key(t) for t in toks(n) if t]
        for L in range(1, min(maxw, len(ws)) + 1): k[" ".join(ws[-L:])] += 1
    return k
FRT = trail_counts(FRN); ST = {c: trail_counts(SN[c]) for c in SN}
# lead-phrase counts too (France puts legal forms first as well: '(SARL) Solution', 'sas pessac amis')
def lead_counts(names):
    k = collections.Counter()
    for n in names:
        ws = [key(t) for t in toks(n) if t]
        if ws: k[ws[0]] += 1
    return k
FRL = lead_counts(FRN)

# ---------------- declared role lexica (groups = aliases of one value; keys are accent-stripped lowercase) ----------------
LEGAL_G = {
 "France": {"sarl": ["sarl", "s.a.r.l.", "s.a.r.l", "sarl."], "sas": ["sas", "s.a.s.", "s.a.s", "sas."], "eurl": ["eurl", "e.u.r.l.", "e.u.r.l"],
            "sa": ["sa", "s.a.", "s.a"], "sasu": ["sasu", "s.a.s.u.", "s.a.s.u"], "sci": ["sci", "s.c.i.", "s.c.i"], "cie": ["cie", "compagnie", "cie."],
            "ets": ["ets", "etablissements", "ets."], "ei": ["ei", "e.i.", "e.i"], "snc": ["snc", "s.n.c.", "s.n.c"], "selarl": ["selarl"],
            "scp": ["scp"], "scm": ["scm"], "earl": ["earl"], "gaec": ["gaec"], "sca": ["sca"], "sccv": ["sccv"], "sel": ["sel"]},
 "US": {"llc": ["llc", "l.l.c.", "l.l.c"], "inc": ["inc", "inc.", "incorporated"], "corp": ["corp", "corp.", "corporation"],
        "co": ["co", "co.", "company"], "ltd": ["ltd", "ltd.", "limited"], "lp": ["lp", "l.p.", "l.p"], "pc": ["pc", "p.c.", "p.c"],
        "pllc": ["pllc", "p.l.l.c."], "llp": ["llp", "l.l.p."], "pa": ["pa", "p.a."], "plc": ["plc"], "lllp": ["lllp"]},
 "India": {"pvtltd": ["private limited", "pvt ltd", "pvt. ltd.", "pvt. ltd", "pvt ltd.", "private-limited", "pvt.ltd.", "pvt.ltd",
                      "private ltd", "pvt limited", "private", "pvt", "pvt."],
           "ltd": ["limited", "ltd", "ltd."], "llp": ["llp", "l.l.p.", "l.l.p"], "co": ["company", "co", "co."],
           "corp": ["corporation", "corp", "corp."], "inc": ["inc", "inc."]}}
STREET_G = {
 "France": {"rue": ["rue", "r"], "avenue": ["avenue", "av", "ave"], "allee": ["allee", "all"], "boulevard": ["boulevard", "bd", "blvd", "bld"],
            "impasse": ["impasse", "imp"], "route": ["route", "rte"], "chemin": ["chemin", "ch", "chem"], "place": ["place", "pl"],
            "cours": ["cours", "crs"], "quai": ["quai", "q"], "passage": ["passage", "pass"], "square": ["square", "sq"],
            "faubourg": ["faubourg", "fbg"], "promenade": ["promenade", "prom"], "voie": ["voie"], "sentier": ["sentier", "sent"],
            "esplanade": ["esplanade", "espl"], "rond-point": ["rond-point", "rpt"], "mail": ["mail"], "parvis": ["parvis"]},
 "US": {"street": ["street", "st", "str"], "road": ["road", "rd"], "drive": ["drive", "dr"], "avenue": ["avenue", "ave", "av"],
        "lane": ["lane", "ln"], "court": ["court", "ct"], "way": ["way", "wy"], "circle": ["circle", "cir"], "place": ["place", "pl"],
        "boulevard": ["boulevard", "blvd"], "highway": ["highway", "hwy"], "trail": ["trail", "trl"], "loop": ["loop"],
        "terrace": ["terrace", "ter"], "parkway": ["parkway", "pkwy"], "pike": ["pike"], "cove": ["cove", "cv"],
        "square": ["square", "sq"], "plaza": ["plaza", "plz"], "alley": ["alley", "aly"]},
 "India": {"road": ["road", "rd"], "street": ["street", "st"], "main": ["main"], "cross": ["cross"], "lane": ["lane", "ln"],
           "marg": ["marg"], "gali": ["gali"], "avenue": ["avenue", "ave"], "path": ["path"], "chowk": ["chowk"]}}
UNIT_G = {
 "France": {"appartement": ["appartement", "appt", "apt"], "etage": ["etage"], "batiment": ["batiment", "bat", "bt"], "porte": ["porte"],
            "escalier": ["escalier", "esc"], "bp": ["bp"], "cs": ["cs"], "lot": ["lot"], "local": ["local"], "bureau": ["bureau", "bur"],
            "entree": ["entree"], "residence": ["residence", "res"], "cellule": ["cellule"], "box": ["box"]},
 "US": {"unit": ["unit"], "suite": ["suite", "ste"], "apartment": ["apartment", "apt"], "pobox": ["po box", "p.o. box", "box"],
        "pmb": ["pmb"], "floor": ["floor", "fl"], "building": ["building", "bldg"], "room": ["room", "rm"]},
 "India": {"floor": ["floor", "flr", "fl"], "plot": ["plot"], "door": ["door"], "hno": ["h.no", "hn", "h no"], "flat": ["flat"],
           "block": ["block"], "house": ["house"], "shop": ["shop"], "office": ["office"], "building": ["building", "bldg"],
           "room": ["room"], "tower": ["tower"], "wing": ["wing"], "unit": ["unit"]}}
FUNC = {"US": {"and": "et", "of": "de", "the": "la", "for": "pour", "in": "en"}, "India": {"and": "et", "of": "de", "the": "la", "for": "pour", "in": "en"}}
FR_REGIONS = ["hauts-de-france", "nouvelle-aquitaine", "pays de la loire"]; FR_DEPTS = ["nord", "gironde", "loire-atlantique", "pas-de-calais"]
# person-name / place tokens never used as category words (role annotation, both sides)
PERSON = set("""jean pierre marie paul louis joseph jacques georges martin michel anne jeanne robert andre francois henri charles
 claude bernard philippe alain daniel jules victor albert saint sainte st ste notre dame smith johnson williams brown jones miller davis
 wilson moore taylor anderson thomas jackson white harris clark lewis young king wright lee dr mr mrs ms sri shri smt m.d. dds md
 kumar singh sharma gupta patel shah jain agarwal reddy rao dba formerly aka fka id a an k l""".split())
US_ST = dict(AL="Alabama", AK="Alaska", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado", CT="Connecticut", DE="Delaware",
    FL="Florida", GA="Georgia", HI="Hawaii", ID="Idaho", IL="Illinois", IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana",
    ME="Maine", MD="Maryland", MA="Massachusetts", MI="Michigan", MN="Minnesota", MS="Mississippi", MO="Missouri", MT="Montana",
    NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey", NM="New Mexico", NY="New York", NC="North Carolina",
    ND="North Dakota", OH="Ohio", OK="Oklahoma", OR="Oregon", PA="Pennsylvania", RI="Rhode Island", SC="South Carolina",
    SD="South Dakota", TN="Tennessee", TX="Texas", UT="Utah", VT="Vermont", VA="Virginia", WA="Washington", WV="West Virginia",
    WI="Wisconsin", WY="Wyoming", DC="District of Columbia")
IN_ST = {"Maharashtra": "MH", "Delhi": "DL", "Karnataka": "KA", "Tamil Nadu": "TN", "Uttar Pradesh": "UP", "Gujarat": "GJ",
    "West Bengal": "WB", "Telangana": "TG", "Rajasthan": "RJ", "Kerala": "KL", "Haryana": "HR", "Madhya Pradesh": "MP",
    "Andhra Pradesh": "AP", "Bihar": "BR", "Punjab": "PB", "Odisha": "OD", "Jharkhand": "JH", "Assam": "AS", "Chhattisgarh": "CG",
    "Uttarakhand": "UK", "Goa": "GA", "Himachal Pradesh": "HP", "Chandigarh": "CH", "Jammu and Kashmir": "JK"}

def rank_groups(groups, cnt):
    """groups: {canon: [variants]} -> list of (canon, [variants ranked by freq]) ranked by total freq (all variants kept)."""
    out = []
    for g, vs in groups.items():
        vv = sorted(vs, key=lambda v: -cnt.get(v, 0)); out.append((g, vv, sum(cnt.get(v, 0) for v in vs), {v: cnt.get(v, 0) for v in vv}))
    return sorted(out, key=lambda x: -x[2])
def synth_variants(canon, have, n):
    """extra distinct French alias surfaces when a source group has more variants than observed French ones (dotted/spaced forms)."""
    letters = list(canon.replace("-", "")) if len(canon) <= 6 else [canon]
    cands = have + [canon + ".", ".".join(letters) + ".", ".".join(letters), " ".join(letters), ". ".join(letters) + ".", "-".join(letters),
                    canon + " .", canon.upper()[:1] + canon[1:] + ".."]
    seen, out = set(), []
    for c in cands:
        if c not in seen: seen.add(c); out.append(c)
    return out[:max(n, len(have))]
TRANSLATE = {"street": {"street": "rue", "road": "route", "avenue": "avenue", "boulevard": "boulevard", "place": "place", "square": "square",
                         "lane": "allee", "way": "voie", "alley": "passage"},
             "unit": {"pobox": "bp", "apartment": "appartement", "floor": "etage", "building": "batiment", "room": "bureau", "flat": "appartement",
                      "office": "bureau"}}
def map_groups(src_ranked, fr_ranked, tr=None):
    """declared translation-equivalent first, then identity on canon, then frequency rank; variants: identity-first, then rank.
    Injective on canon. Returns {src_variant: fr_variant}, {src_canon: fr_canon}."""
    fr_left = [g for g in fr_ranked]; gm = {}
    fr_by = {g[0]: g for g in fr_ranked}
    for g in src_ranked:
        t = (tr or {}).get(g[0])
        if t in fr_by and t not in gm.values(): gm[g[0]] = t
    for g in src_ranked:
        if g[0] in fr_by and g[0] not in gm and g[0] not in gm.values(): gm[g[0]] = g[0]
    used = set(gm.values()); rest = [g for g in fr_ranked if g[0] not in used]
    for g in src_ranked:
        if g[0] not in gm and rest: gm[g[0]] = rest.pop(0)[0]
    vm = {}
    for g in src_ranked:
        if g[0] not in gm: continue
        fv = synth_variants(gm[g[0]], list(fr_by[gm[g[0]]][1]), len(g[1]))
        left = list(fv); m = {}
        for v in g[1]:
            if v in left: m[v] = v; left.remove(v)
        for v in g[1]:
            if v not in m and left: m[v] = left.pop(0)
        vm.update(m)
    return vm, gm
def rank_map(src_list, fr_list):
    """identity-first then rank, injective. lists are (key, count) ranked."""
    fr_keys = [k for k, _ in fr_list]; m = {}
    for k, _ in src_list:
        if k in fr_keys: m[k] = k
    rest = [k for k in fr_keys if k not in set(m.values())]
    for k, _ in src_list:
        if k not in m and rest: m[k] = rest.pop(0)
    return m

R = {"_note": __doc__, "France": {}, "US": {}, "India": {}}
FR_LEGAL = rank_groups(LEGAL_G["France"], {**{v: FRK.get(v, 0) + FRT.get(v, 0) * 0 for g in LEGAL_G["France"].values() for v in g}})
FR_STREET = rank_groups(STREET_G["France"], FRAK); FR_UNIT = rank_groups(UNIT_G["France"], FRAK)
R["France"].update(legal=[(g, v, n) for g, v, n, _ in FR_LEGAL], street=[(g, v, n) for g, v, n, _ in FR_STREET],
                   unit=[(g, v, n) for g, v, n, _ in FR_UNIT])
# word-table roles
W = json.load(open("data/tables/wordlr_test.json"))
def wt_roles(c):
    ins = W[c]["ins"]; mx = max(v[1] for v in ins.values()); mt = max(v[1] + v[2] for v in ins.values())
    G = {w: v for w, v in ins.items() if v[1] >= 0.25 * mx and v[0] >= 1.0}; med = statistics.median(v[0] for v in G.values())
    dual = {w for w, v in G.items() if 1.0 <= v[0] <= med - 1.5}; pure = set(G) - dual
    gen2 = {w: v for w, v in ins.items() if w not in G and v[0] >= 1.0 and v[1] >= 0.04 * mx and w.isalpha() and len(w) > 2}
    noise = {w: v for w, v in ins.items() if v[0] < 0 and v[1] + v[2] >= 0.05 * mt and w.isalpha() and len(w) > 2 and w not in ("dba", "and", "the", "formerly", "aka", "fka", "nee")}
    rk = lambda s: sorted(((w, ins[w][1]) for w in s), key=lambda x: -x[1])
    return dict(gen_pure=rk(pure), gen_dual=rk(dual), gen2=rk(gen2), noise_ins=sorted(((w, ins[w][2]) for w in noise), key=lambda x: -x[1]))
WR = {c: wt_roles(c) for c in ("France", "US", "India")}
R["France"]["wordtable_roles"] = WR["France"]
# French category words: frequent name tokens not in any other role
fr_legal_keys = {v for g in LEGAL_G["France"].values() for v in g}
fr_wt = {w for r in WR["France"].values() for w, _ in r}
# French place tokens (cities/regions/depts) from address tails
comp_cnt = collections.Counter(); surf = collections.defaultdict(collections.Counter)
for a in FRA:
    for p in a.split(","):
        p = p.strip()
        if p and not any(ch.isdigit() for ch in p): comp_cnt[key(p)] += 1; surf[key(p)][p] += 1
def norm_city(k): return re.sub(r"\s+", " ", k.replace("st.-", "saint-").replace("st-", "saint-").replace("-", " ")).strip()
REG_N = {norm_city(x) for x in FR_REGIONS + FR_DEPTS}
cityv = collections.defaultdict(collections.Counter)
for k, n in comp_cnt.items():
    if n >= 1000 and norm_city(k) not in REG_N: cityv[norm_city(k)][k] += n
# co-occurrence city -> dept, city -> region
co_d, co_r = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
for a in FRA:
    ps = [norm_city(key(p.strip())) for p in a.split(",")]
    cs = [p for p in ps if p in cityv]; ds = [p for p in ps if p in {norm_city(x) for x in FR_DEPTS}]; rs = [p for p in ps if p in {norm_city(x) for x in FR_REGIONS}]
    for c in cs:
        for d in ds: co_d[c][d] += 1
        for r in rs: co_r[c][r] += 1
def display(k):
    best = [s for s, _ in surf[k].most_common() if not s.isupper() and not s.islower()]
    return best[0] if best else surf[k].most_common(1)[0][0]
cities = []
for c, vs in sorted(cityv.items(), key=lambda x: -sum(x[1].values())):
    tot = sum(vs.values())
    variants = [(display(k), n) for k, n in vs.most_common() if n >= 0.05 * tot]
    if not (co_d[c] and co_r[c]): continue
    cities.append(dict(key=c, n=tot, variants=variants, dept=co_d[c].most_common(1)[0][0] if co_d[c] else None,
                       region=co_r[c].most_common(1)[0][0] if co_r[c] else None))
dispreg = {norm_city(key(x)): display(key(x)) for x in FR_REGIONS + FR_DEPTS}
cities = [c for c in cities if c["dept"] and c["region"] and (c["dept"] in ("nord", "pas de calais")) == (c["region"] == "hauts de france")
          and not (c["dept"] == "gironde") != (c["region"] == "nouvelle aquitaine")]
R["France"]["cities"] = cities; R["France"]["dept_display"] = {norm_city(x): dispreg[norm_city(x)] for x in FR_DEPTS}
R["France"]["region_display"] = {norm_city(x): dispreg[norm_city(x)] for x in FR_REGIONS}
dept_region = collections.defaultdict(collections.Counter)
for c in cities:
    if c["dept"] and c["region"]: dept_region[c["dept"]][c["region"]] += c["n"]
R["France"]["dept_region"] = {d: r.most_common(1)[0][0] for d, r in dept_region.items()}
R["France"]["dept_n"] = {norm_city(x): comp_cnt[key(x)] for x in FR_DEPTS}; R["France"]["region_n"] = {norm_city(x): comp_cnt[key(x)] for x in FR_REGIONS}
fr_place = set()
for c in cities:
    for v, _ in c["variants"]: fr_place |= {key(t) for t in re.split(r"[\s-]+", v)}
for x in FR_REGIONS + FR_DEPTS: fr_place |= set(re.split(r"[\s-]+", x))
FUNC_FR = {"de", "du", "des", "la", "le", "les", "et", "en", "pour", "au", "aux", "l", "d"}
def fr_cat_ok(k): return (k.isalpha() and len(k) >= 3 and k not in fr_legal_keys and k not in fr_wt and k not in fr_place and k not in PERSON
                         and k not in FUNC_FR and k.isascii())
# merge accent-noise variants: FRK keys are accent-stripped already
FR_CAT = [(k, n) for k, n in FRK.most_common() if fr_cat_ok(k) and n >= 300]
R["France"]["category"] = FR_CAT
R["France"]["surfaces"] = {k: [(s, n) for s, n in FRSURF[k].most_common() if n >= 0.1 * sum(FRSURF[k].values())]
                           for k in set([k for k, _ in FR_CAT] + [w for r in WR["France"].values() for w, _ in r])}
R["France"]["vocab_n"] = len([1 for k, n in (FRK + FRAK).items() if n >= 5])
# ---------------- source countries ----------------
for c in ("US", "India"):
    S = R[c]; cnt = SK[c]
    legal_cnt = {v: ST[c].get(v, 0) for g in LEGAL_G[c].values() for v in g}
    SL = rank_groups(LEGAL_G[c], legal_cnt); FL = rank_groups(LEGAL_G["France"], {v: FRK.get(v, 0) for g in LEGAL_G["France"].values() for v in g})
    S["legal_vm"], S["legal_gm"] = map_groups(SL, FL); S["legal_rank"] = [(g, v, n) for g, v, n, _ in SL]
    SS = rank_groups(STREET_G[c], SAK[c]); S["street_vm"], S["street_gm"] = map_groups(SS, FR_STREET, TRANSLATE["street"]); S["street_rank"] = [(g, v, n) for g, v, n, _ in SS]
    SU = rank_groups(UNIT_G[c], SAK[c]); S["unit_vm"], S["unit_gm"] = map_groups(SU, FR_UNIT, TRANSLATE["unit"]); S["unit_rank"] = [(g, v, n) for g, v, n, _ in SU]
    w = WR[c]; f = WR["France"]; used_fr = set(); nm = {}
    def assign(src_list, pools):
        for k, _ in src_list:
            if k in nm: continue
            for pool in pools:
                if k in [x for x, _ in pool] and k not in used_fr: nm[k] = k; used_fr.add(k); break
        for k, _ in src_list:
            if k in nm: continue
            for pool in pools:
                cand = [x for x, _ in pool if x not in used_fr]
                if cand: nm[k] = cand[0]; used_fr.add(cand[0]); break
    role_of = {}
    for role, src_l, pools in [("gen_pure", w["gen_pure"], [f["gen_pure"], f["gen_dual"]]), ("gen_dual", w["gen_dual"], [f["gen_dual"], f["gen_pure"]]),
                               ("gen2", w["gen2"], [f["gen2"]]), ("noise_ins", w["noise_ins"], [f["noise_ins"]])]:
        before = set(nm); assign(src_l, pools)
        for k in set(nm) - before: role_of[k] = role
    src_legal_keys = {v for g in LEGAL_G[c].values() for v in g for v in v.split()}
    src_place = set()
    for a in SA[c][:300000]:
        for p in a.split(","):
            p = p.strip()
            if p and not any(ch.isdigit() for ch in p): src_place |= {key(t) for t in p.split()}
    cat_src = [(k, n) for k, n in cnt.most_common(3000) if k.isalpha() and k.isascii() and len(k) >= 3 and k not in nm and k not in src_legal_keys
               and k not in PERSON and k not in FUNC[c] and n >= 300 and not (k in src_place and SAK[c].get(k, 0) > 3 * n)]
    fr_cat_left = [(k, n) for k, n in FR_CAT if k not in used_fr]
    cm = rank_map(cat_src, fr_cat_left)
    for k, v in cm.items(): nm[k] = v; role_of[k] = "category"
    for k, v in FUNC[c].items(): nm[k] = v; role_of[k] = "function"
    S["name_map"] = nm; S["name_role"] = role_of
    S["wordtable_roles"] = w

R["US_ST"] = US_ST; R["IN_ST"] = IN_ST
# India native-script state -> English state, by co-occurrence of the preceding city component (records only, no labels)
tr_in = tr.filter(pl.col("country") == "India")["business_address"].drop_nulls().to_list()
city_st = collections.defaultdict(collections.Counter); nat = collections.defaultdict(collections.Counter)
INL = {k.lower(): k for k in IN_ST}
for a in tr_in:
    ps = [p.strip() for p in a.split(",")]
    if len(ps) >= 2 and ps[-1].lower() in INL: city_st[ps[-2].lower()][INL[ps[-1].lower()]] += 1
for a in tr_in:
    ps = [p.strip() for p in a.split(",")]
    if len(ps) >= 2 and ps[-1] and not ps[-1].isascii() and ps[-2].lower() in city_st: nat[ps[-1]][city_st[ps[-2].lower()].most_common(1)[0][0]] += 1
R["IN_NATIVE_ST"] = {k: v.most_common(1)[0][0] for k, v in nat.items() if sum(v.values()) >= 50 and v.most_common(1)[0][1] >= 0.8 * sum(v.values())}
s = json.dumps(R, ensure_ascii=False, indent=1); open(OUT, "w").write(s)
print("sha256", hashlib.sha256(s.encode()).hexdigest()); print("written", OUT)
