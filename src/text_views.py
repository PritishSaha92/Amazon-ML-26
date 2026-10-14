"""License-safe multi-view text normalizer for business names and addresses.

Pure Python + stdlib (`re`, `unicodedata`) — no Unidecode / anyascii / external data.
Supersedes the throwaway src/norm.py (which used GPL Unidecode).

Design (see research/03_names_addresses.md and research/04_text_views_report.md):
  * strip_accents  : NFKC -> casefold -> NFKD -> drop *Latin* combining marks -> NFC.
                     Indic vowel signs / viramas are combining marks too, so we only drop marks
                     in the generic combining-diacritics blocks, never Brahmic marks.
  * fold_indic     : every Brahmic block (Bengali..Malayalam) is folded onto Devanagari by the
                     0x80-step alignment; romanize_indic then needs a single Devanagari table.
  * clean          : accents + junk wrappers + filler tokens (keeps punctuation so the address
                     number / comma-component logic still works downstream).
  * name_* / addr_*: string views; apply_views() adds them all to a polars frame in parallel.

Learned tables (data/tables/*.json, produced by src/mine_tables.py) are optional: when present,
Indic words are latinized by the learned dictionary first and romanize_indic() is the fallback.

Run `python src/text_views.py` for self-tests; `python src/text_views.py --bench [train|test]
[--write]` to time apply_views on a records file (and write data/views_{split}.parquet).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

# ----------------------------------------------------------------------------------------------
# Character-level primitives
# ----------------------------------------------------------------------------------------------

# Combining-mark blocks that carry Latin/Greek/Cyrillic diacritics. Brahmic vowel signs, nukta,
# virama, anusvara etc. live inside their own script blocks and are deliberately NOT here.
_LATIN_MARK_RANGES = ((0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF),
                      (0x20D0, 0x20FF), (0xFE20, 0xFE2F))
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")


def _is_latin_mark(ch: str) -> bool:
    o = ord(ch)
    for lo, hi in _LATIN_MARK_RANGES:
        if lo <= o <= hi:
            return True
    return False


def strip_accents(s: str) -> str:
    """NFKC -> casefold -> NFKD -> drop Latin combining marks (Mn) -> NFC; zero-width chars removed.

    >>> strip_accents("Çoncept Frèrês límited")
    'concept freres limited'
    >>> strip_accents("लिमिटेड") == "लिमिटेड"      # Indic marks are preserved
    True
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not (unicodedata.category(c) == "Mn" and _is_latin_mark(c)))
    s = _ZERO_WIDTH.sub("", s)
    # Re-compose so that split vowel signs (e.g. Tamil o = e-sign + aa-sign) become one code point.
    return unicodedata.normalize("NFC", s)


_INDIC_LO, _INDIC_HI = 0x0900, 0x0DFF
_INDIC_RE = re.compile("[\u0900-\u0DFF]")

# A few script-specific letters whose 0x80-aligned slot in Devanagari means something else.
# Applied before the arithmetic fold.
_PRE_FOLD = {
    "\u0a70": "\u0902",  # Gurmukhi tippi -> anusvara
    "\u0a71": "",        # Gurmukhi addak (gemination mark) -> drop
    "ੲ": "इ",  # Gurmukhi iri -> i
    "ੳ": "उ",  # Gurmukhi ura -> u
    "ୱ": "व",  # Odia wa -> va
    "\u0bd7": "",        # Tamil au length mark
    "\u0c55": "", "\u0c56": "",  # Telugu length marks
    "\u0cd5": "", "\u0cd6": "",  # Kannada length marks
    "\u0d57": "",        # Malayalam au length mark
}
_PRE_FOLD_RE = re.compile("[" + "".join(_PRE_FOLD) + "]")


def has_indic(s: str) -> bool:
    return bool(_INDIC_RE.search(s))


def _fold_char(c: str) -> str:
    o = ord(c)
    if 0x0980 <= o <= _INDIC_HI:
        return chr(0x900 + (o - 0x900) % 0x80)
    return c


def fold_indic(s: str) -> str:
    """Fold Bengali/Gurmukhi/Gujarati/Odia/Tamil/Telugu/Kannada/Malayalam onto Devanagari.

    >>> fold_indic("లిమిటెడ్") == fold_indic("ಲಿಮಿಟೆಡ್") == "लिमिटॆड्"
    True
    >>> fold_indic("abc Çé")
    'abc Çé'
    """
    if not s or not _INDIC_RE.search(s):
        return s
    s = _PRE_FOLD_RE.sub(lambda m: _PRE_FOLD[m.group(0)], s)
    return "".join(_fold_char(c) for c in s)


# ---- Devanagari -> Latin table (~90 entries). Keys are Devanagari code points (post-fold). ----
_IND_VOWELS = {  # independent vowels
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ॠ": "ri",
    "ऌ": "li", "ऍ": "e", "ऎ": "e", "ए": "e", "ऐ": "ai", "ऑ": "o", "ऒ": "o", "ओ": "o",
    "औ": "au", "ॲ": "e",
}
_MATRAS = {  # dependent vowel signs
    "\u093e": "a", "\u093f": "i", "\u0940": "i", "\u0941": "u", "\u0942": "u", "\u0943": "ri",
    "\u0944": "ri", "\u0945": "e", "\u0946": "e", "\u0947": "e", "\u0948": "ai", "\u0949": "o",
    "\u094a": "o", "\u094b": "o", "\u094c": "au", "\u0962": "li", "\u0963": "li",
    "\u093a": "e", "\u093b": "e", "\u094e": "e", "\u094f": "au",
}
_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "zh", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # precomposed nukta forms (NFC keeps them decomposed, but be safe)
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}
_NUKTA_CONS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_NUKTA, _VIRAMA = "\u093c", "\u094d"
_ANUSVARA, _CHANDRABINDU, _VISARGA = "\u0902", "\u0901", "\u0903"
_OTHER = {  # misc signs; Malayalam chillus fold into 0x97A..0x97F
    "।": " ", "॥": " ", "ऽ": "", "ॐ": "om", "॰": "", "ॱ": "",
    "ॺ": "n", "ॻ": "n", "ॼ": "r", "ॽ": "l", "ॾ": "l", "ॿ": "k",
    "\u0900": "n", "\u0955": "", "\u0956": "", "\u0957": "",
}
_INDIC_RUN = re.compile("[\u0900-\u0DFF]+")


def _romanize_run(run: str, north: bool = True, final_m: bool = False) -> str:
    """Romanize one maximal run of (folded) Devanagari code points, i.e. one word.

    north: apply Hindi-style word-final schwa deletion (scripts that write final consonants
    without a virama: Devanagari/Bengali/Gurmukhi/Gujarati/Odia). Southern scripts mark
    consonant endings with an explicit virama, so their final inherent 'a' is real.
    final_m: word-final anusvara reads as 'm' (Malayalam).
    """
    units: list[str] = []       # romanized pieces
    inherent: list[bool] = []   # piece ends in an implicit schwa (candidate for deletion)
    i, n = 0, len(run)
    while i < n:
        c = run[i]
        if c in _CONS:
            base = _CONS[c]
            j = i + 1
            if j < n and run[j] == _NUKTA:
                base = _NUKTA_CONS.get(c, base)
                j += 1
            if j < n and run[j] == _VIRAMA:
                units.append(base); inherent.append(False); i = j + 1
            elif j < n and run[j] in _MATRAS:
                units.append(base + _MATRAS[run[j]]); inherent.append(False); i = j + 1
            else:
                units.append(base + "a"); inherent.append(True); i = j
            continue
        if c in _IND_VOWELS:
            units.append(_IND_VOWELS[c]); inherent.append(False)
        elif c in (_ANUSVARA, _CHANDRABINDU):
            nxt = run[i + 1] if i + 1 < n else ""
            labial = _CONS.get(nxt, "")[:1] in ("p", "b", "m") or (final_m and not nxt)
            units.append("m" if labial else "n")
            inherent.append(False)
        elif c == _VISARGA:
            units.append("h"); inherent.append(False)
        elif c in _MATRAS:          # stray vowel sign (after a vowel): keep its vowel
            units.append(_MATRAS[c]); inherent.append(False)
        elif c.isdigit():
            units.append(str(unicodedata.digit(c))); inherent.append(False)
        else:
            units.append(_OTHER.get(c, "")); inherent.append(False)
        i += 1
    # Word-final schwa deletion ("limiteda" -> "limited"), multi-syllable words only. Kept after
    # a conjunct ending in r/y/v ("maharashtra", "surya", "mitra"); loanword clusters such as
    # "impeks" still lose it.
    if north and len(units) > 1 and inherent[-1]:
        last = len(run) - 1 - (run[-1] == _NUKTA)
        if not (last >= 1 and run[last - 1] == _VIRAMA and run[last] in "\u0930\u092f\u0935"):
            units[-1] = units[-1][:-1]
    return "".join(units)


def romanize_indic(s: str) -> str:
    """Phonetic Latin rendering of Indic text (any Brahmic script). Latin text passes through.

    >>> romanize_indic("प्राइवेट लिमिटेड")
    'praivet limited'
    >>> romanize_indic("ప్రైవేట్ లిమిటెడ్")
    'praivet limited'
    >>> romanize_indic("Sun Constructions")
    'Sun Constructions'
    """
    if not s or not _INDIC_RE.search(s):
        return s
    s = _ZERO_WIDTH.sub("", unicodedata.normalize("NFC", s))

    def one(m):
        run = m.group(0)
        o = ord(run[0])
        return _romanize_run(fold_indic(run), north=o < 0x0B80, final_m=o >= 0x0D00)
    # Runs are romanized in their original script so the script-specific rules can apply;
    # already-folded (Devanagari) text is treated as northern.
    return _INDIC_RUN.sub(one, s)


_VOWELS = set("aeiouy")


def consonant_skeleton(word: str) -> str:
    """Consonant skeleton for fuzzy matching of transliterations.

    Drop vowels (keep a leading vowel), drop aspiration 'h', merge voiced/unvoiced pairs and
    w/v, c/k/q, z/s, j/ch, then collapse doubled letters.

    >>> consonant_skeleton("limited"), consonant_skeleton("limitad")
    ('lmt', 'lmt')
    >>> consonant_skeleton("private") == consonant_skeleton("praivet")
    True
    >>> consonant_skeleton("developers") == consonant_skeleton("tevalappars")
    True
    >>> consonant_skeleton("lakshmi") == consonant_skeleton("laxmi")
    True
    """
    w = word.lower()
    if not w:
        return ""
    w = w.replace("ph", "f").replace("ck", "k").replace("x", "ks")
    w = w.replace("ch", "C").replace("sh", "s")
    w = re.sub(r"g(?=[eiy])", "j", w)                # soft g ~ j
    w = re.sub(r"(?<=[^aeiouy])h", "", w)            # aspiration: kh->k, bh->b, th->t ...
    w = w.translate(_SKEL_MAP)
    out = [w[0]] if w[0] in _VOWELS else []
    for ch in w[len(out):]:
        if ch in _VOWELS or not ch.isalnum():
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


_SKEL_MAP = str.maketrans({"b": "p", "d": "t", "g": "k", "c": "k", "q": "k", "z": "s",
                           "w": "v", "j": "C"})

# ----------------------------------------------------------------------------------------------
# clean(): accents + junk wrappers + fillers (punctuation kept)
# ----------------------------------------------------------------------------------------------

LEGAL_FORMS = frozenset("""llc inc corp corporation co company ltd limited pvt private llp plc
    sarl sas sasu eurl sci sa snc ets cie gmbh pc lc incorporated trust pllc
    ei selarl scp scm earl gaec sca lp lllp""".split())
HONORIFICS = frozenset("sri shri smt mr mrs dr m mme ms".split())
NAME_STOPWORDS = frozenset("the of and & de du des la le les d l".split())
FILLERS = frozenset("null n/a na none divreportingcircle nan".split())
DOMAIN_TLDS = ("com", "net", "org", "in", "co", "io", "fr", "us")

_W = r"\w\u0900-\u0DFF"                               # word chars incl. Indic marks
_LEAD_JUNK = re.compile(rf"^[^{_W}]+")
_TRAIL_JUNK = re.compile(rf"[^{_W})\].]+$")
_WWW_FRAG = re.compile(r"\s*\|\s*(?:https?://)?www\.\S*")
_LEGAL_TAG = re.compile(r"[\[\(]\s*(?:" + "|".join(sorted(LEGAL_FORMS | {"group"}, key=len, reverse=True))
                        + r")\.?\s*[\]\)]")
_TRAIL_HASHNUM = re.compile(r"\s*#\s*\d+\s*$")
_TRAIL_PHONE = re.compile(r"\s*[-–|,]\s*\+?\d[\d ]{6,}\d\s*$")
_FILLER_RE = re.compile(rf"(?<![{_W}/])(?:null|n/a|none|divreportingcircle)(?![{_W}/])")
_WS = re.compile(r"\s+")


def clean(s: str, is_name: bool = True) -> str:
    """Accent-fold + remove junk wrappers and filler tokens; keeps punctuation and case-folds.

    >>> clean("*** Çoncept Frèrês [INC] #39256")
    'concept freres'
    >>> clean(">> Sri Indian Private Limited | www.indiansft.com")
    'sri indian private limited'
    >>> clean("3923 Willetta Street, N/A, Phoenix, <NULL>", is_name=False)
    '3923 willetta street, , phoenix, < >'
    """
    if not s:
        return ""
    s = strip_accents(s)
    s = _WWW_FRAG.sub(" ", s)
    s = _LEGAL_TAG.sub(" ", s)
    s = _LEGAL_TAG.sub(" ", s)   # nested "([private])"
    if is_name:
        s = _TRAIL_PHONE.sub(" ", s)
        s = _TRAIL_HASHNUM.sub(" ", s)
        s = _LEAD_JUNK.sub("", s)
    else:   # addresses: leading "#12" is a real number, only drop leading junk before it
        s = re.sub(rf"^[^{_W}#]+", "", s)
    s = _FILLER_RE.sub(" ", s)
    return _WS.sub(" ", s).strip()


# ----------------------------------------------------------------------------------------------
# Tokenization shared by name / address views
# ----------------------------------------------------------------------------------------------

_DOTTED_ACR = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b\.?)?")   # l.l.c. / s.a.s / e.u.r.l.
_ELISION = re.compile(r"\b(qu|[ldjnsctm])['’](?=\w)")          # l'alma -> l alma
_APOS = re.compile(r"['’`´]")
_NUMSIGN = re.compile(r"\bn\s*[°º]\s*(?=\d)|\bn[°º]")          # N° 7 -> no 7
_NON_WORD = re.compile(rf"[^{_W}\s]+|_")


def _tokens(s: str, is_name: bool) -> list[str]:
    """clean() output -> list of lowercase tokens (letters/digits/Indic), Indic folded."""
    if is_name:
        s = s.replace("&", " and ").replace(" + ", " and ").replace("m/s", "ms")
    s = _DOTTED_ACR.sub(lambda m: m.group(0).replace(".", "") + " ", s)
    s = _NUMSIGN.sub("no ", s)
    s = _ELISION.sub(r"\1 ", s)
    s = _APOS.sub("", s)
    s = _NON_WORD.sub(" ", s)
    s = fold_indic(s)
    return s.split()


def name_words(s: str) -> str:
    """Cleaned name words; Indic script kept (folded to Devanagari).

    >>> name_words("-- Sbc (India) Ifntaech Pvt. Ltd. #86134")
    'sbc india ifntaech pvt ltd'
    >>> name_words("Poetes & Cie Sàrl")
    'poetes and cie sarl'
    >>> name_words("E.U.R.L. Calais Groupement")
    'eurl calais groupement'
    """
    return " ".join(_tokens(clean(s, True), True))


def addr_words(s: str) -> str:
    """Cleaned address words (elisions split, fillers dropped). Indic folded, not latinized.

    >>> addr_words("71 Rue de l'Alma, Roubaix, Hauts-de-France")
    '71 rue de l alma roubaix hauts de france'
    """
    return " ".join(_tokens(clean(s, False), False))


# ----------------------------------------------------------------------------------------------
# Indic latinization with optional learned tables
# ----------------------------------------------------------------------------------------------

class Tables:
    """Learned lookup tables (all optional). Built by src/mine_tables.py."""

    def __init__(self, indic_word: dict | None = None, indic_comp: dict | None = None):
        self.indic_word = indic_word or {}   # folded Indic word -> Latin word
        self.indic_comp = indic_comp or {}   # folded Indic address component -> Latin string

    @classmethod
    def load(cls, tables_dir: str = "data/tables") -> "Tables":
        def top(path):
            p = os.path.join(tables_dir, path)
            if not os.path.exists(p):
                return {}
            with open(p) as f:
                d = json.load(f)
            return {k: v[0][0] for k, v in d.items() if v}
        word = top("indic_dict.json")
        word.update({k: v for k, v in top("indic_fallback.json").items() if k not in word})
        return cls(word, top("indic_addr.json"))


_EMPTY_TABLES = Tables()


def latinize_word(w: str, tables: Tables = _EMPTY_TABLES) -> str:
    """Indic word -> Latin via learned dictionary, else own romanization. Latin passes through."""
    if not _INDIC_RE.search(w):
        return w
    hit = tables.indic_word.get(w)
    return hit if hit is not None else romanize_indic(w).lower()


def name_latin(s: str, tables: Tables = _EMPTY_TABLES) -> str:
    return " ".join(latinize_word(w, tables) for w in name_words(s).split())


def _latin_tokens_from_words(words: list[str], tables: Tables) -> list[str]:
    out = []
    for w in words:
        out.extend(latinize_word(w, tables).split())
    return out


# ----------------------------------------------------------------------------------------------
# Name views
# ----------------------------------------------------------------------------------------------

_DOMAIN_LIKE = re.compile(r"[a-z0-9]\.(?:" + "|".join(DOMAIN_TLDS) + r")\b")
_LEET = {"0": "o", "1": "l", "5": "s", "3": "e"}


def _looks_like_domain(raw: str) -> bool:
    return bool(_DOMAIN_LIKE.search(raw.lower()))


def _drop_tld(words: list[str], raw: str) -> list[str]:
    if len(words) > 1 and words[-1] in DOMAIN_TLDS and _looks_like_domain(raw):
        return words[:-1]
    return words


def _core_words(words: list[str]) -> list[str]:
    """Drop legal forms / honorifics / stopwords and repeated words (token duplication is a
    generator noise: "Global Alphlalbet Alphlalbet Group Group")."""
    seen: set[str] = set()
    out = []
    for w in words:
        if w in LEGAL_FORMS or w in HONORIFICS or w in NAME_STOPWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def leet_word(w: str) -> str:
    """Fold 0/1/5/3 only when the digit sits inside an alphabetic run.

    >>> [leet_word(x) for x in ["inn0vative", "pla5tic", "b1aser", "0mega", "1st", "2024", "3m"]]
    ['innovative', 'plastic', 'blaser', 'omega', '1st', '2024', '3m']
    """
    if w.isdigit() or not any(ch in _LEET for ch in w):
        return w
    n_alpha = sum(ch.isalpha() for ch in w)
    n_digit = sum(ch.isdigit() for ch in w)
    out = []
    for i, ch in enumerate(w):
        if ch in _LEET:
            left = w[i - 1].isalpha() if i > 0 else False
            right = w[i + 1].isalpha() if i + 1 < len(w) else False
            if (left and right) or ((left or right) and n_alpha >= 3 and n_digit == 1):
                out.append(_LEET[ch]); continue
        out.append(ch)
    return "".join(out)


def _name_views(raw: str, tables: Tables) -> dict:
    words = _tokens(clean(raw, True), True)
    lat = _latin_tokens_from_words(words, tables)
    lat_nd = _drop_tld(lat, raw)
    core = _core_words(lat_nd)
    return {
        "name_words": " ".join(words),
        "name_latin": " ".join(lat),
        "name_core": " ".join(core),
        "name_compact": "".join(lat_nd),
        "name_leet": "".join(leet_word(w) for w in lat_nd),
        "name_sorted": " ".join(sorted(core)),
        "name_skel": " ".join(consonant_skeleton(w) for w in core),
    }


def name_core(s: str, tables: Tables = _EMPTY_TABLES) -> str:
    """Name words minus legal forms, honorifics, stopwords, repeats (Indic words latinized).

    >>> name_core("The Sri Laxmi Traders Pvt Ltd")
    'laxmi traders'
    >>> name_core("Global Alphlalbet Alphlalbet Group Group LP")
    'global alphlalbet group'
    """
    return _name_views(s, tables)["name_core"]


def name_compact(s: str, tables: Tables = _EMPTY_TABLES) -> str:
    """All alphanumerics, no spaces; trailing TLD dropped for domain-like names.

    >>> name_compact("cardiologysafecare.com"), name_compact("Cardiology Safe Care")
    ('cardiologysafecare', 'cardiologysafecare')
    """
    return _name_views(s, tables)["name_compact"]


def name_leet(s: str, tables: Tables = _EMPTY_TABLES) -> str:
    """
    >>> name_leet("Inn0vative Foundation 2020"), name_leet("pimyie1d.com - 8541734086")
    ('innovativefoundation2020', 'pimyield')
    """
    return _name_views(s, tables)["name_leet"]


def name_sorted(s: str, tables: Tables = _EMPTY_TABLES) -> str:
    """
    >>> name_sorted("Traders Laxmi Pvt Ltd") == name_sorted("Laxmi Traders Private Limited")
    True
    """
    return _name_views(s, tables)["name_sorted"]


# ----------------------------------------------------------------------------------------------
# Address views
# ----------------------------------------------------------------------------------------------

# State / region stoplist (rules-in-code, not external data): used to drop whole comma
# components that are only a state/region name or code.
US_STATES = """alabama al alaska ak arizona az arkansas ar california ca colorado co connecticut ct
delaware de florida fl georgia ga hawaii hi idaho id illinois il indiana in iowa ia kansas ks
kentucky ky louisiana la maine me maryland md massachusetts ma michigan mi minnesota mn
mississippi ms missouri mo montana mt nebraska ne nevada nv ohio oh oklahoma ok oregon or
pennsylvania pa tennessee tn texas tx utah ut vermont vt virginia va washington wa wisconsin wi
wyoming wy nh nj nm ny nc nd ri sc sd wv dc""".split()
US_STATES_MULTI = ["new hampshire", "new jersey", "new mexico", "new york", "north carolina",
                   "north dakota", "rhode island", "south carolina", "south dakota",
                   "west virginia", "district of columbia", "puerto rico"]
IN_STATES = """andhra pradesh|ap|arunachal pradesh|ar|assam|as|bihar|br|chhattisgarh|cg|ct|goa|ga|
gujarat|gj|haryana|hr|himachal pradesh|hp|jharkhand|jh|karnataka|ka|kerala|keralam|kl|
madhya pradesh|mp|maharashtra|mh|manipur|mn|meghalaya|ml|mizoram|mz|nagaland|nl|odisha|orissa|od|
or|punjab|pb|rajasthan|rj|sikkim|sk|tamil nadu|tn|telangana|tg|ts|tripura|tr|uttar pradesh|up|
uttarakhand|uttaranchal|uk|ut|west bengal|wb|delhi|dl|new delhi|jammu and kashmir|jammu kashmir|
jk|ladakh|la|puducherry|pondicherry|py|chandigarh|ch|andaman and nicobar islands|an|
dadra and nagar haveli and daman and diu|dn|dd|lakshadweep|ld""".replace("\n", "").split("|")
FR_REGIONS = ["hauts de france", "nouvelle aquitaine", "pays de la loire", "ile de france",
              "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire",
              "corse", "grand est", "normandie", "occitanie", "provence alpes cote d azur",
              "france"]
DEFAULT_STATE_STOP = frozenset(US_STATES + US_STATES_MULTI + IN_STATES + FR_REGIONS)
ADDR_FILLERS = frozenset("null none na nan n a divreportingcircle".split())

_NUM_RUN = re.compile(r"\d+(?:\s*-\s*\d+)*")


def addr_numbers(s: str) -> list[str]:
    """Sorted number tokens, leading zeros stripped; '-' and '/' split; short hyphen joins kept.

    >>> addr_numbers("2-43 Main St"), addr_numbers("17982-17986 Hwy"), addr_numbers("#3639/A., 05 Rd")
    (['2', '43', '243'], ['17982', '17986'], ['5', '3639'])
    >>> addr_numbers("")
    []
    """
    if not s:
        return []
    s = strip_accents(s)
    out: set[str] = set()
    for m in _NUM_RUN.finditer(s):
        parts = [p.strip() for p in m.group(0).split("-")]
        parts = [str(int(p)) for p in parts if p]
        out.update(parts)
        # the generator inserts hyphens into numbers (243 -> 2-43): keep the joined form
        if len(parts) == 2 and min(len(p) for p in parts) <= 2 and len(parts[0] + parts[1]) <= 6:
            out.add(parts[0] + parts[1])
    return sorted(out, key=lambda x: (len(x), x))


def _addr_components(s: str, tables: Tables) -> list[list[str]]:
    """Comma components -> list of latinized token lists (empty components dropped)."""
    comps = []
    for comp in clean(s, False).split(","):
        toks = _tokens(comp, False)
        if not toks:
            continue
        key = " ".join(toks)
        if _INDIC_RE.search(key):
            hit = tables.indic_comp.get(key)
            toks = hit.split() if hit is not None else _latin_tokens_from_words(toks, tables)
        comps.append(toks)
    return comps


def _addr_views(raw: str, tables: Tables, state_stop: frozenset = DEFAULT_STATE_STOP) -> dict:
    comps = _addr_components(raw, tables)
    words = [t for c in comps for t in c]
    core = []
    for c in comps:
        if " ".join(c) in state_stop:
            continue
        core.extend(t for t in c if not t.isdigit() and t not in ADDR_FILLERS)
    return {
        "addr_words": " ".join(words),
        "addr_core": " ".join(core),
        "addr_numbers": addr_numbers(raw),
        "addr_tokens": " ".join(sorted(set(words))),
    }


def addr_core(s: str, state_stop: frozenset = DEFAULT_STATE_STOP,
              tables: Tables = _EMPTY_TABLES) -> str:
    """Address words minus state/region-only components, fillers and pure numbers.

    >>> addr_core("3923 Willetta Street, N/A, Phoenix, Arizona")
    'willetta street phoenix'
    >>> addr_core("55 Rue Cugnot, Roubaix, Nord, Hauts-de-France")
    'rue cugnot roubaix nord'
    """
    return _addr_views(s, tables, state_stop)["addr_core"]


def addr_tokens_set(s: str, tables: Tables = _EMPTY_TABLES) -> frozenset:
    return frozenset(_addr_views(s, tables)["addr_tokens"].split())


# ----------------------------------------------------------------------------------------------
# All views for one record / a whole frame
# ----------------------------------------------------------------------------------------------

VIEW_COLUMNS = ["has_indic", "name_words", "name_latin", "name_core", "name_compact", "name_leet",
                "name_sorted", "name_skel", "addr_words", "addr_core", "addr_numbers",
                "addr_tokens"]


def record_views(name: str, addr: str, tables: Tables = _EMPTY_TABLES) -> dict:
    d = {"has_indic": bool(_INDIC_RE.search(name or "") or _INDIC_RE.search(addr or ""))}
    d.update(_name_views(name or "", tables))
    d.update(_addr_views(addr or "", tables))
    return d


_WORKER_TABLES: Tables = _EMPTY_TABLES


def _init_worker(tables: Tables) -> None:
    global _WORKER_TABLES
    _WORKER_TABLES = tables


def _views_chunk(args):
    """Worker: (names, addrs) -> polars DataFrame of view columns (Arrow-pickled back)."""
    import polars as pl
    names, addrs = args
    cols: dict[str, list] = {c: [] for c in VIEW_COLUMNS}
    t = _WORKER_TABLES
    for n, a in zip(names, addrs):
        d = record_views(n, a, t)
        for c in VIEW_COLUMNS:
            cols[c].append(d[c])
    return pl.DataFrame(cols, schema={c: (pl.Boolean if c == "has_indic" else
                                          pl.List(pl.String) if c == "addr_numbers" else pl.String)
                                      for c in VIEW_COLUMNS})


def apply_views(df, tables: Tables | str | None = "data/tables", workers: int = 12,
                chunk: int = 100_000, name_col: str = "business_name",
                addr_col: str = "business_address"):
    """Add all VIEW_COLUMNS to a records frame (row order preserved).

    `tables`: a Tables object, a directory to load learned tables from (missing files are fine),
    or None for pure rule-based views. Uses a process pool of `workers` over row chunks.
    """
    import multiprocessing as mp
    import polars as pl

    if isinstance(tables, str):
        tables = Tables.load(tables)
    tables = tables or _EMPTY_TABLES
    names = df[name_col].fill_null("").to_list()
    addrs = df[addr_col].fill_null("").to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    del names, addrs
    if workers <= 1 or len(jobs) <= 1:
        _init_worker(tables)
        parts = [_views_chunk(j) for j in jobs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(tables,)) as pool:
            parts = pool.map(_views_chunk, jobs, chunksize=1)
    if not parts:
        return df.with_columns([pl.Series(c, [], dtype=pl.String) for c in VIEW_COLUMNS])
    views = pl.concat(parts, how="vertical")
    return df.hstack(views)


# ----------------------------------------------------------------------------------------------
# Self-tests / benchmark
# ----------------------------------------------------------------------------------------------

def _selftest() -> None:
    import doctest
    import sys
    res = doctest.testmod(sys.modules[__name__], verbose=False)
    assert res.failed == 0, res
    assert strip_accents("Çoncept") == "concept"
    assert strip_accents("Frèrês") == "freres"
    assert strip_accents("límited") == "limited"
    assert strip_accents("") == "" and name_words("") == "" and addr_words("") == ""
    # Indic: same English word in several scripts -> same / close romanization
    assert romanize_indic("प्राइवेट लिमिटेड") == "praivet limited"
    assert romanize_indic("லிமிடெட்") in ("limitet", "limited")
    assert romanize_indic("ಲಿಮಿಟೆಡ್") == "limited"
    assert romanize_indic("লিমিটেড") == "limited"
    assert romanize_indic("શ્રી") == "shri"
    assert romanize_indic("ज़ू") == "zu" and romanize_indic("फ़ूड") == "fud"
    assert romanize_indic("संपत") == "sampat"
    assert romanize_indic("ఇన్వెస్ట్\u200cమెంట్స్".replace("\u200c", "")) == "investments"
    assert consonant_skeleton(romanize_indic("லிமிடெட்")) == consonant_skeleton("limited")
    assert name_words("Sri Upgrade Trading  Private Ltd | www.upgradetr.com") == \
        "sri upgrade trading private ltd"
    assert name_core("M/s Al Pvt Ltd [Services]") == "al services"
    assert addr_numbers("C-6-B-256 APURPA COLONY") == ["6", "256"]
    assert addr_numbers("N° 13 RUE DU BAS") == ["13"]
    v = record_views("गोल्डन इम्पेक्स प्राइवेट लिमिटेड", "Pune City, 86 Mumbai-pune Road, Pune, महाराष्ट्र")
    assert v["has_indic"] and v["name_latin"] == "goldan impeks praivet limited", v
    print(f"text_views self-test OK ({res.attempted} doctests)")


def _bench(split: str = "train", write: bool = False) -> None:
    """Time apply_views on data/{split}_records.parquet; optionally write data/views_{split}.parquet
    (entity_id + view columns)."""
    import time
    import polars as pl
    t0 = time.time()
    df = pl.read_parquet(f"data/{split}_records.parquet")
    t1 = time.time()
    out = apply_views(df)
    t2 = time.time()
    print(f"[{split}] rows={len(df):,} read={t1 - t0:.1f}s views={t2 - t1:.1f}s "
          f"({(t2 - t1) / len(df) * 1e6:.1f} us/row wall, 12 workers)", flush=True)
    if write:
        out.select(["entity_id"] + VIEW_COLUMNS).write_parquet(f"data/views_{split}.parquet")
        print(f"[{split}] wrote data/views_{split}.parquet in {time.time() - t2:.1f}s")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:   # python src/text_views.py --bench [train|test] [--write]
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        _bench(args[0] if args else "train", "--write" in sys.argv)
    else:
        _selftest()
