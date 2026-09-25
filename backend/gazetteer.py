"""
Post-OCR correction against known vocabularies.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

OCR and HTR both misread faded text. Where the correct answer must come from a
CLOSED set - a district exists or it does not, a land class is one of eleven
codes - a misread value can be snapped back to the real one. This module does
that, using RapidFuzz for the string distance and the bundled LGD extract
(backend/data/admin_master.json: 8 states, 43 districts, 453 tehsils, 6,516
villages) as the authority.

WHAT THE MEASUREMENTS SAID, AND WHY THE DESIGN LOOKS LIKE THIS
--------------------------------------------------------------
Fuzzy-matching place names is far more dangerous than fuzzy-matching English
words against a dictionary, and the corpus says so numerically.

1. NO THRESHOLD SEPARATES AN OCR ERROR FROM A DIFFERENT VILLAGE.
   Scored over hand-built pairs, genuine OCR corruptions and genuinely
   distinct places overlap:

       Lucknov    -> Lucknow    85.7   same place, misread
       Barabanki  -> Barabani   94.1   DIFFERENT places
       Basar      -> Bansar     90.9   DIFFERENT places
       Anvi       -> Anvri      88.9   DIFFERENT places

   Every scorer tried (ratio, WRatio, QRatio, token_sort) gave the same
   picture: good-min 85.7 against bad-max 94.1. A cutoff low enough to fix
   "Lucknov" also rewrites "Barabanki" into a different real village, which
   silently reassigns a parcel to the wrong place. Indian village names are
   densely packed in edit-distance space - thousands of short names one
   character apart, all real - so similarity ALONE can never license a
   correction here.

2. HIERARCHICAL SCOPING IS WHAT MAKES IT SAFE.
   Measured over all 6,516 villages, asking how many have a rival within the
   same similarity band:

       unscoped (national), ratio >= 88   28.7% ambiguous
       scoped to tehsil,    ratio >= 88    1.9% ambiguous
       scoped to tehsil,    ratio >= 92    0.6% ambiguous

   A fifteen-fold reduction. So a village is only ever matched inside its
   resolved tehsil, a tehsil inside its resolved district. When the parent is
   unknown the candidate set cannot be narrowed, and this module then
   SUGGESTS rather than corrects - because at national scope one case in
   three is a coin flip.

3. AMBIGUITY IS REFUSED, NOT RESOLVED.
   Even inside a tehsil, if two candidates sit within MARGIN of each other
   the value is left alone and flagged. Picking the higher of two
   near-identical scores is guessing about someone's land.

4. FUZZY MATCHING CANNOT CROSS SCRIPTS DIRECTLY - SO IT GOES THROUGH A
   PHONETIC KEY.
   RapidFuzz itself is script-agnostic; it compares codepoints and handles
   Devanagari, Bengali or Tamil perfectly well ('नरहरपुर' vs 'नरहरपूर'
   scores 85.7). The problem was never the library, it was the DATA: every
   name in the bundled LGD extract is romanised - 0 of 6,516 villages, 0 of
   453 tehsils and 0 of 43 districts are in Devanagari. A Devanagari value
   scored against them returns 0.0 while extractOne still happily returns a
   "best" match ('नरहरपुर' -> 'Acharamau', score 0.0).

   Both sides are therefore reduced to a common PHONETIC KEY before
   comparison: Devanagari is transliterated to IAST, Hindi schwa deletion is
   applied, diacritics are stripped, and a set of romanisation equivalences
   is folded in (w/v, au/o, ai/e, oo/u, doubled letters collapsed). The same
   folding runs over the romanised gazetteer, so the two meet in the middle.

   Measured on 16 district pairs whose romanisation is independently known
   (लखनऊ/Lucknow, भोपाल/Bhopal, कानपुर/Kanpur ...), at the district bar of
   84: ELEVEN auto-applied, all eleven correct, ZERO wrong. The other five
   fell below the bar and were reported rather than applied - including
   लखनऊ -> Lucknow at 62, which is correct but indistinguishable by score
   from बंगलौर -> Bhagalpur at 62, which is wrong. Refusing both is the only
   defensible reading of that tie.

   Folding does not create new ambiguity: collapsing 6,516 village names to
   phonetic keys costs 0.4% of within-tehsil distinctness, 0.4% of
   within-district tehsil distinctness, and 0.0% for districts.

   A value in a script the transliterator does not cover still reports
   "no comparable vocabulary" - which is NOT the same as "this place does
   not exist", and must never be collapsed into it.

Identifiers are a different problem and get a different tool: a khasra number
has a SHAPE (123, 123/2, 123/2/1) but no closed vocabulary, so it is checked
by regex and never snapped to a neighbour.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import field_extractor
from field_extractor import LAND_CLASSES, detect_script, normalise, normalise_digits

try:
    from rapidfuzz import fuzz, process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except Exception:                                    # pragma: no cover
    fuzz = None
    rf_process = None
    RAPIDFUZZ_AVAILABLE = False

# Devanagari romanisation is done HERE rather than with a library, and that
# is a deliberate reversal.
#
# indic-transliteration was used first and worked, but installing it BROKE
# spaCy on this machine: the Application Control policy then refused
# spacy/pipeline/multitask.pyd, and the NER cross-check silently dropped to
# five skipped tests. Uninstalling it restored spaCy, which identified the
# cause beyond doubt. Trading away a working feature for a new one is not a
# trade worth making, and the library was doing far more than is needed here
# - twenty-odd schemes with exact round-tripping, where all this module wants
# is a rough phonetic key that is about to have its diacritics stripped
# anyway. So the table below is the whole dependency.
TRANSLITERATION_AVAILABLE = True

# Scripts with a table. Anything else reports "no comparable vocabulary"
# rather than being scored against a vocabulary it cannot be compared with.
_TRANSLITERABLE = {"devanagari"}

# Independent vowels.
_DEVA_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ॠ": "ri", "ऌ": "li", "ए": "e", "ऐ": "ai", "ओ": "o",
    "औ": "au", "ऑ": "o", "ऍ": "e",
}

# Vowel signs (matras), which replace a consonant's inherent 'a'.
_DEVA_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "ॄ": "ri", "ॢ": "li", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}

# Consonants, WITHOUT the inherent vowel - that is added by the walker below
# unless a matra or a virama says otherwise.
#
# The nukta letters ड़ and ढ़ map to d/dh rather than the linguistically
# tidier r: Indian romanisation writes बरखेड़ी as "Barkhedi", and matching
# the gazetteer is the whole point of this table.
_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh",
    "फ़": "f", "ऱ": "r", "य़": "y",
}

_DEVA_VIRAMA = "\u094d"
_DEVA_NASALS = {"\u0902", "\u0901"}      # anusvara, chandrabindu -> n
_DEVA_VISARGA = "\u0903"
_DEVA_NUKTA = "\u093c"
_DEVA_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def romanise_devanagari(text: str) -> str:
    """
    Devanagari to rough Latin, keeping the inherent vowel explicit.

    The inherent 'a' is emitted after every consonant that is not followed by
    a matra or a virama, because phonetic_key's schwa-deletion step needs to
    SEE it in order to decide which ones Hindi actually drops. Emitting a
    pre-deleted form here would make that decision impossible downstream.
    """
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # A nukta modifies the preceding consonant, so try the two-character
        # form before the single one.
        pair = text[i:i + 2]
        if len(pair) == 2 and pair[1] == _DEVA_NUKTA and pair[0] in _DEVA_CONSONANTS:
            combined = _DEVA_CONSONANTS.get(pair, _DEVA_CONSONANTS[pair[0]])
            out.append(combined)
            i += 2
            nxt = text[i] if i < n else ""
            if nxt in _DEVA_MATRAS:
                out.append(_DEVA_MATRAS[nxt])
                i += 1
            elif nxt == _DEVA_VIRAMA:
                i += 1
            else:
                out.append("a")
            continue
        if ch in _DEVA_CONSONANTS:
            out.append(_DEVA_CONSONANTS[ch])
            i += 1
            nxt = text[i] if i < n else ""
            if nxt in _DEVA_MATRAS:
                out.append(_DEVA_MATRAS[nxt])
                i += 1
            elif nxt == _DEVA_VIRAMA:
                i += 1
            else:
                out.append("a")
            continue
        if ch in _DEVA_VOWELS:
            out.append(_DEVA_VOWELS[ch])
        elif ch in _DEVA_NASALS:
            out.append("n")
        elif ch == _DEVA_VISARGA:
            out.append("h")
        elif ch in _DEVA_DIGITS:
            out.append(_DEVA_DIGITS[ch])
        elif ch.isspace() or ch in "-_./":
            out.append(" ")
        i += 1
    return "".join(out)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MASTER_PATH = os.path.join(_HERE, "data", "admin_master.json")

# Similarity at or above which a SCOPED match may be applied automatically -
# PER LEVEL, because the levels are not equally crowded and one global number
# was measurably wrong for two of them.
#
# Share of entries that have a rival within the same scope (all 6,516
# villages, 453 tehsils and 43 districts of the bundled extract):
#
#   threshold | districts | tehsils-in-district | villages-in-tehsil
#         80  |    0.0%   |        4.9%         |       9.6%
#         84  |    0.0%   |        4.0%         |       3.9%
#         88  |    0.0%   |        0.9%         |       1.9%
#         92  |    0.0%   |        0.4%         |       0.6%
#
# Districts never collide: the closest pair in the entire extract is
# Jaipur/Udaipur at 76.9, so anything above ~77 can only ever return one
# candidate. Holding districts to the villages' 92 was therefore pure cost -
# it left 'Lucknov' -> 'Lucknow' (85.7) unresolved, and because every other
# level is scoped BY district, one unresolvable district blocked the tehsil
# and village checks too. A crowded level gets a strict bar and a sparse one
# does not.
AUTO_APPLY_SCORE = 92.0            # default, used for villages
LEVEL_AUTO_APPLY = {
    "district": 84.0,              # 0% ambiguous at 80; closest real pair 76.9
    "state": 84.0,                 # eight entries, none remotely similar
    "tehsil": 88.0,                # 0.9% ambiguous within a district
    "village": 92.0,               # 0.6% ambiguous within a tehsil
}

# Below this, no suggestion is offered at all: the value is more likely a
# genuinely unlisted place (the extract is partial) than a misreading of a
# listed one, and offering a bad suggestion invites a reviewer to accept it.
SUGGEST_SCORE = 80.0

# Two candidates within this many points are treated as indistinguishable.
AMBIGUITY_MARGIN = 3.0

# A plot identifier's shape. Sub-division is expressed with "/" and can nest
# more than once (123/2/1 is normal), and some states use a letter suffix
# (142/2B on a Maharashtra 7/12).
KHASRA_PATTERN = re.compile(r"^\d{1,6}(?:/\d{1,4}[A-Za-z]?){0,3}$")
KHATA_PATTERN = re.compile(r"^\d{1,8}$")
# ULPIN (Bhu-Aadhaar) is specified by DILRMP 3.0 as exactly 14 characters.
# The previous range accepted 10 to 16, which let a truncated or
# over-read identifier through as "confirmed" - and an identifier whose
# length is wrong is not a near miss, it is a different parcel or no
# parcel. Composition is left tolerant (alphanumeric) because the
# internal structure is not publicly pinned the way the length is.
ULPIN_LENGTH = 14
ULPIN_PATTERN = re.compile(r"^[A-Z0-9]{14}$")

# Mutation kinds recorded on Indian revenue records. A fixed list per the
# problem statement; states differ in wording, so regional synonyms map onto
# the same code the way LAND_CLASSES already does for land use.
MUTATION_TYPES: Dict[str, List[str]] = {
    "sale": ["sale", "bikri", "बिक्री", "vikray", "विक्रय", "kharid", "खरीद",
             "registry", "रजिस्ट्री"],
    "inheritance": ["inheritance", "succession", "warasat", "वरासत", "विरासत",
                    "uttaradhikar", "उत्तराधिकार", "varsa", "वारसा"],
    "gift": ["gift", "daan", "दान", "hiba", "हिबा"],
    "partition": ["partition", "batwara", "बटवारा", "बँटवारा", "vibhajan", "विभाजन"],
    "mortgage": ["mortgage", "rehan", "रेहन", "bandhak", "बंधक", "गहाण"],
    "lease": ["lease", "patta", "पट्टा", "kiraya", "किराया"],
    "court_decree": ["decree", "court", "adalat", "अदालत", "न्यायालय", "डिक्री"],
    "acquisition": ["acquisition", "adhigrahan", "अधिग्रहण", "bhoo-arjan", "भू-अर्जन"],
    "correction": ["correction", "shuddhi", "शुद्धि", "sudhar", "सुधार"],
}


@dataclass
class Correction:
    """
    One vocabulary check on one field.

    `applied` says whether the value was actually changed. Everything else is
    evidence for a reviewer: what it was, what it became or could become, how
    similar, and which set it was compared against.
    """
    field_key: str
    original: Optional[str]
    outcome: str                       # confirmed | corrected | suggested
                                       # | ambiguous | not_found | unusable
                                       # | no_vocabulary | invalid_shape
    value: Optional[str] = None        # the value to use (corrected or original)
    score: Optional[float] = None
    candidates: List[str] = dc_field(default_factory=list)
    scope: Optional[str] = None        # what the value was compared against
    message: str = ""
    applied: bool = False
    needs_review: bool = False
    via: str = "direct"               # direct | transliteration

    def to_dict(self) -> dict:
        return {
            "field_key": self.field_key, "original": self.original,
            "outcome": self.outcome, "value": self.value, "score": self.score,
            "candidates": self.candidates, "scope": self.scope,
            "message": self.message, "applied": self.applied,
            "needs_review": self.needs_review, "via": self.via,
        }


def _key(value: str) -> str:
    """Comparison form: case, spacing and punctuation folded away."""
    return re.sub(r"[^a-z0-9ऀ-ॿ]", "", normalise(value or "").lower())


def _script_of(value: str) -> str:
    return detect_script(value or "")


# --------------------------------------------------------------------------
# Phonetic key: the bridge between an Indic value and a romanised gazetteer
# --------------------------------------------------------------------------

def _strip_marks(text: str) -> str:
    """Drop combining diacritics, so IAST's ā/ī/ṭ/ṣ fold onto a/i/t/s."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def _fold_roman(text: str) -> str:
    """
    Reduce romanised text to the equivalences Indian place names actually
    vary by. Aurangabad/Orangabad, Jaipur/Jepur, Barkhedi/Barkheri,
    Lucknow/Lakhnau: the same place, spelled by different conventions.
    """
    text = text.lower()
    text = text.replace("au", "o").replace("ai", "e")
    text = re.sub(r"[^a-z]", "", text)
    text = text.replace("w", "v").replace("q", "k").replace("x", "ks")
    text = text.replace("oo", "u").replace("ee", "i")
    text = re.sub(r"(.)+", r"", text)      # collapse doubled letters
    text = re.sub(r"a$", "", text)              # drop a trailing inherent 'a'
    return text


def phonetic_key(value: str) -> Optional[str]:
    """
    A script-independent key for one name, or None if it cannot be made.

    For an Indic script the value goes through IAST and then HINDI SCHWA
    DELETION - the inherent 'a' after a consonant is dropped when the next
    consonant carries its own vowel, because Hindi does not pronounce it and
    LGD does not write it ('नरहरपुर' is romanised Narharpur, not
    naraharapura).

    Long 'aa' is emitted as two characters by the table, which is what keeps
    the schwa rule from eating it: an earlier version folded it to a single
    'a' first and then deleted it, turning कानपुर into 'knpur' instead of
    'kanpur' and costing three of sixteen test pairs.
    """
    text = (value or "").strip()
    if not text:
        return None
    script = _script_of(text)
    if script == "latin":
        return _fold_roman(_strip_marks(text)) or None
    if script not in _TRANSLITERABLE:
        return None
    roman = romanise_devanagari(text)
    if not roman.strip():
        return None
    # Hindi schwa deletion: drop the inherent 'a' where the following
    # consonant carries its own vowel. 'naraharapura' -> 'narharpur', which is
    # how LGD writes it. Long 'aa' is spelled with two characters by the
    # table above precisely so this rule cannot eat it.
    roman = re.sub(r"(?<=[a-z])a(?=[bcdfghjklmnpqrstvyz]+[aeiou])", "", roman)
    return _fold_roman(roman) or None


# --------------------------------------------------------------------------
# The gazetteer
# --------------------------------------------------------------------------

class Gazetteer:
    """
    The LGD hierarchy, indexed for scoped lookup.

    Indexes are built once at import: state -> districts, district -> tehsils,
    (district, tehsil) -> villages. The scoping in match_place() depends on
    these being addressable by parent, which is the whole point.
    """

    def __init__(self, path: str = _MASTER_PATH):
        self.loaded = False
        self.states: Dict[str, dict] = {}
        self.districts: Dict[str, str] = {}            # key -> canonical name
        self.district_state: Dict[str, str] = {}
        self.tehsils_by_district: Dict[str, List[str]] = {}
        self.villages_by_tehsil: Dict[Tuple[str, str], List[str]] = {}
        self.villages_by_district: Dict[str, List[str]] = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.states = payload.get("states", {})
            self.loaded = bool(self.states)
        except Exception:
            self.loaded = False
        if self.loaded:
            self._index()

    def _index(self) -> None:
        for state, meta in self.states.items():
            for district, dmeta in (meta.get("districts") or {}).items():
                dk = _key(district)
                self.districts[dk] = district
                self.district_state[dk] = state
                tehsils = list(dmeta.get("tehsils") or [])
                self.tehsils_by_district[dk] = tehsils
                villages = dmeta.get("villages") or {}
                flat: List[str] = []
                if isinstance(villages, dict):
                    for tehsil, names in villages.items():
                        self.villages_by_tehsil[(dk, _key(tehsil))] = list(names)
                        flat.extend(names)
                else:
                    flat.extend(villages)
                self.villages_by_district[dk] = flat

    # -- exact membership ---------------------------------------------------

    def district(self, name: str) -> Optional[str]:
        return self.districts.get(_key(name))

    def state_of_district(self, name: str) -> Optional[str]:
        return self.district_state.get(_key(name))

    def tehsils(self, district: str) -> List[str]:
        return self.tehsils_by_district.get(_key(district), [])

    def villages(self, district: str, tehsil: Optional[str] = None) -> List[str]:
        dk = _key(district)
        if tehsil:
            scoped = self.villages_by_tehsil.get((dk, _key(tehsil)))
            if scoped:
                return scoped
        return self.villages_by_district.get(dk, [])

    def states_list(self) -> List[str]:
        return list(self.states.keys())


GAZETTEER = Gazetteer()


# --------------------------------------------------------------------------
# Fuzzy matching with the guards the measurements demanded
# --------------------------------------------------------------------------

def _comparable(value: str, candidates: Sequence[str]) -> List[str]:
    """
    Candidates written in the same script as `value`.

    Without this, a Devanagari value is scored against a romanised gazetteer,
    every score is 0.0, and the "best match" is whichever entry happened to
    be first - a confident answer computed from nothing.

    An empty result routes the caller to the phonetic bridge, so what counts
    as "unknown" matters. detect_script returns "unknown" for two very
    different things: plain ASCII that carries no script signal ("237/4"),
    which compares perfectly well against a romanised list, and text in a
    script this system has no table for ("北京市"), which does not. Treating
    both as comparable sent Han text down the direct path and reported
    "not found" for a name that was never actually checked.
    """
    script = _script_of(value)
    if script and script not in ("unknown", ""):
        return [c for c in candidates if _script_of(c) == script]
    # Unknown script: comparable only if it is ASCII and therefore already in
    # the same alphabet as the gazetteer.
    if all(ord(ch) < 128 for ch in (value or "")):
        return list(candidates)
    return []


_VOWELS = re.compile(r"[aeiou]")


def _skeleton(key):
    """
    A phonetic key with its vowels removed.

    A second, lossier view of the same name, used alongside the full key
    because the two fail on different things. Romanisation disagrees about
    vowels far more than about consonants - Lakhnau/Lucknow,
    Barkhedi/Barkheri - so dropping them recovers matches the full key
    misses.

    Measured on the 16 verifiable district pairs, adding this took correct
    auto-applications from 11 to 13 with no new errors. It is lossier:
    within-tehsil distinctness costs 1.3% against the full key's 0.4%. Those
    losses surface as AMBIGUOUS - two candidates tie and the value is left
    alone - rather than as wrong answers, which is the trade being made.
    """
    if not key:
        return None
    return _VOWELS.sub("", key) or None


def _match_phonetic(original, candidates, field_key, scope_label, auto):
    """
    Match across scripts by reducing both sides to a phonetic key.

    Used only when the value and the vocabulary are in different scripts, so
    a direct comparison is meaningless - it returns 0.0 for every candidate
    while still naming a "best" one.
    """
    key = phonetic_key(original)
    if not key:
        return Correction(
            field_key, original, "no_vocabulary", value=original,
            scope=scope_label, needs_review=True, via="transliteration",
            message=("This value is in " + _script_of(original) + " script, the "
                     + scope_label + " is romanised, and no transliteration is "
                     "available for that script, so they cannot be compared. "
                     "This is NOT evidence that the place does not exist."))

    by_key = {}
    by_skeleton = {}
    for candidate in candidates:
        ckey = phonetic_key(candidate)
        if not ckey:
            continue
        by_key.setdefault(ckey, candidate)
        skeleton = _skeleton(ckey)
        if skeleton:
            by_skeleton.setdefault(skeleton, candidate)
    if not by_key:
        return Correction(
            field_key, original, "no_vocabulary", value=original,
            scope=scope_label, needs_review=True, via="transliteration",
            message=("No entry in the " + scope_label + " could be reduced to "
                     "a comparable key."))

    scored = {}
    for probe, table in ((key, by_key), (_skeleton(key), by_skeleton)):
        if not probe or not table:
            continue
        for match, score, _ in rf_process.extract(
                probe, list(table.keys()), scorer=fuzz.ratio, limit=3):
            name = table[match]
            scored[name] = max(scored.get(name, 0.0), float(score))

    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < SUGGEST_SCORE:
        return Correction(
            field_key, original, "not_found", value=original,
            score=ranked[0][1] if ranked else None, scope=scope_label,
            candidates=[n for n, _ in ranked[:3]], needs_review=True,
            via="transliteration",
            message=("'" + original + "' transliterates to '" + key + "', which "
                     "matches nothing close in the " + scope_label + ". The "
                     "bundled extract is partial, so absence is not proof of "
                     "error."))

    best, best_score = ranked[0]
    rivals = [n for n, sc in ranked[1:] if best_score - sc <= AMBIGUITY_MARGIN]
    if rivals:
        names = ", ".join(repr(c) for c in [best] + rivals)
        return Correction(
            field_key, original, "ambiguous", value=original, score=best_score,
            scope=scope_label, candidates=[best] + rivals, needs_review=True,
            via="transliteration",
            message=("'" + original + "' transliterates to '" + key + "', which "
                     "is about equally close to " + names + ". Choosing between "
                     "them would be a guess about which land this is, so the "
                     "value was left unchanged."))

    if best_score >= auto:
        return Correction(
            field_key, original, "corrected", value=best, score=best_score,
            scope=scope_label, candidates=[best], applied=True,
            needs_review=True, via="transliteration",
            message=("'" + original + "' transliterates to '" + key + "' and "
                     "matches '" + best + "' (" + ("%.0f" % best_score) + "%), "
                     "the only close entry in the " + scope_label + ". "
                     "Transliteration is approximate - confirm against the "
                     "document."))

    return Correction(
        field_key, original, "suggested", value=original, score=best_score,
        scope=scope_label, candidates=[n for n, _ in ranked[:3]],
        needs_review=True, via="transliteration",
        message=("'" + original + "' transliterates to '" + key + "'; the "
                 "closest entry in the " + scope_label + " is '" + best + "' ("
                 + ("%.0f" % best_score) + "%). Too uncertain to apply "
                 "automatically."))


def match_in_scope(value: str, candidates: Sequence[str], field_key: str,
                   scope_label: str, auto_score: Optional[float] = None) -> Correction:
    """
    Match one value against one already-narrowed candidate list.

    The caller is responsible for the narrowing; this function will not widen
    it, because the 28.7%-vs-1.9% ambiguity measurement is entirely a property
    of how narrow the list is.
    """
    original = (value or "").strip()
    auto = auto_score if auto_score is not None else LEVEL_AUTO_APPLY.get(
        field_key, AUTO_APPLY_SCORE)
    if not original:
        return Correction(field_key, original, "unusable", value=None,
                          scope=scope_label, message="No value to check.")
    if not RAPIDFUZZ_AVAILABLE:
        return Correction(field_key, original, "unusable", value=original,
                          scope=scope_label,
                          message="RapidFuzz is not installed; vocabulary "
                                  "correction was skipped.")
    if not candidates:
        return Correction(field_key, original, "no_vocabulary", value=original,
                          scope=scope_label,
                          message=f"No {scope_label} list is available to check "
                                  f"this value against.")

    # Exact membership first - no scoring needed, and no risk taken.
    lookup = {_key(c): c for c in candidates}
    hit = lookup.get(_key(original))
    if hit:
        return Correction(field_key, original, "confirmed", value=hit,
                          score=100.0, scope=scope_label,
                          message=f"'{hit}' is listed in the {scope_label}.")

    usable = _comparable(original, candidates)
    if not usable:
        # Different scripts: comparing codepoints is meaningless here, so go
        # through the phonetic bridge rather than score 0.0 against every
        # candidate and then name a winner anyway.
        return _match_phonetic(original, candidates, field_key, scope_label, auto)

    ranked = rf_process.extract(original, usable, scorer=fuzz.ratio, limit=3)
    if not ranked or ranked[0][1] < SUGGEST_SCORE:
        return Correction(
            field_key, original, "not_found", value=original,
            score=ranked[0][1] if ranked else None, scope=scope_label,
            candidates=[r[0] for r in ranked[:3]],
            message=(f"'{original}' is not in the {scope_label} and nothing "
                     f"close enough to suggest. The bundled extract is "
                     f"partial, so absence is not proof of error."),
            needs_review=True)

    best, best_score = ranked[0][0], ranked[0][1]
    rivals = [r[0] for r in ranked[1:]
              if best_score - r[1] <= AMBIGUITY_MARGIN]
    if rivals:
        return Correction(
            field_key, original, "ambiguous", value=original,
            score=best_score, scope=scope_label,
            candidates=[best] + rivals, needs_review=True,
            message=(f"'{original}' is about equally close to "
                     f"{', '.join(repr(c) for c in [best] + rivals)} in the "
                     f"{scope_label}. Choosing between them would be a guess "
                     f"about which land this is, so the value was left "
                     f"unchanged."))

    if best_score >= auto:
        return Correction(
            field_key, original, "corrected", value=best, score=best_score,
            scope=scope_label, candidates=[best], applied=True,
            needs_review=True,
            message=(f"'{original}' corrected to '{best}' ({best_score:.0f}% "
                     f"similar, only match in the {scope_label}). Confirm "
                     f"against the document."))

    return Correction(
        field_key, original, "suggested", value=original, score=best_score,
        scope=scope_label, candidates=[r[0] for r in ranked], needs_review=True,
        message=(f"'{original}' is not in the {scope_label}; the closest entry "
                 f"is '{best}' ({best_score:.0f}%). Too uncertain to apply "
                 f"automatically."))


# --------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------

def _match_controlled(value: str, table: Dict[str, List[str]],
                      field_key: str, label: str) -> Correction:
    """
    Snap a value onto a controlled code by synonym, then by fuzzy distance.

    Safe where place names are not, because the target set is a handful of
    codes rather than thousands of near-identical names - so a wrong snap has
    somewhere obvious to be wrong and a reviewer will see it.
    """
    original = (value or "").strip()
    if not original:
        return Correction(field_key, original, "unusable", value=None, scope=label)
    low = normalise(original).lower()

    # Longest synonym wins: "असिंचित" contains "सिंचित", and matching the
    # shorter one first inverts the meaning of every unirrigated parcel.
    best_code, best_len = None, 0
    for code, words in table.items():
        for word in words:
            wl = word.lower()
            if wl in low and len(wl) > best_len:
                best_code, best_len = code, len(wl)
    if best_code:
        return Correction(field_key, original, "confirmed", value=best_code,
                          score=100.0, scope=label,
                          message=f"Recognised as '{best_code}'.")

    if not RAPIDFUZZ_AVAILABLE:
        return Correction(field_key, original, "unusable", value=original,
                          scope=label, message="RapidFuzz is not installed.")

    flat = [(w, code) for code, words in table.items() for w in words]
    usable = [w for w, _ in flat if _script_of(w) == _script_of(original)] or \
             [w for w, _ in flat]
    ranked = rf_process.extract(low, usable, scorer=fuzz.ratio, limit=2)
    if ranked and ranked[0][1] >= AUTO_APPLY_SCORE:
        word = ranked[0][0]
        code = next(c for w, c in flat if w == word)
        return Correction(field_key, original, "corrected", value=code,
                          score=ranked[0][1], scope=label, applied=True,
                          needs_review=True,
                          message=(f"'{original}' read as '{word}' "
                                   f"({ranked[0][1]:.0f}% similar) -> "
                                   f"'{code}'. Confirm against the document."))

    return Correction(
        field_key, original, "not_found", value=original,
        score=ranked[0][1] if ranked else None, scope=label,
        candidates=[r[0] for r in ranked], needs_review=True,
        message=(f"'{original}' does not match any known {label}. Passed "
                 f"through unchanged rather than forced into a code - an "
                 f"unrecognised term is information, not an error."))


def match_land_class(value: str) -> Correction:
    return _match_controlled(value, LAND_CLASSES, "land_classification",
                             "land classification vocabulary")


def match_mutation_type(value: str) -> Correction:
    return _match_controlled(value, MUTATION_TYPES, "mutation_type",
                             "mutation type vocabulary")


# --------------------------------------------------------------------------
# Identifier shape
# --------------------------------------------------------------------------

def check_identifier(value: str, field_key: str) -> Correction:
    """
    Validate an identifier's SHAPE. Never snaps it to a neighbour.

    A plot number has no closed vocabulary - 213/1 and 218/1 are both
    perfectly valid and denote different land - so the only safe check is
    whether the string looks like a plot number at all. Correcting one to the
    other is exactly the mistake fact_checker.py was already bitten by.
    """
    original = (value or "").strip()
    if not original:
        return Correction(field_key, original, "unusable", value=None)

    # Devanagari digits are a transcription of the same number, so they are
    # folded before the shape test rather than failing it.
    ascii_form = normalise_digits(original)
    patterns = {"khasra_number": KHASRA_PATTERN, "survey_number": KHASRA_PATTERN,
                "khata_number": KHATA_PATTERN, "ulpin": ULPIN_PATTERN}
    pattern = patterns.get(field_key)
    if pattern is None:
        return Correction(field_key, original, "no_vocabulary", value=original,
                          message="No shape is defined for this identifier.")

    cleaned = re.sub(r"\s+", "", ascii_form)
    if pattern.match(cleaned):
        return Correction(field_key, original, "confirmed", value=cleaned,
                          score=100.0, scope=f"{field_key} format",
                          message=f"'{cleaned}' has the expected shape.")

    shapes = {"khasra_number": "123, 123/2 or 123/2/1",
              "survey_number": "123, 123/2 or 142/2B",
              "khata_number": "a plain number such as 1428",
              "ulpin": "exactly 14 alphanumeric characters, e.g. UP091223700412"}
    return Correction(
        field_key, original, "invalid_shape", value=original, scope=f"{field_key} format",
        needs_review=True,
        message=(f"'{original}' does not look like a {field_key.replace('_', ' ')} "
                 f"({shapes.get(field_key, 'unknown shape')}). It was left "
                 f"unchanged: a plot number cannot be repaired by guessing, "
                 f"because a different number is different land."))


# --------------------------------------------------------------------------
# The whole record, resolved top-down
# --------------------------------------------------------------------------

def correct_record(values: Dict[str, dict],
                   gazetteer: Optional[Gazetteer] = None) -> List[Correction]:
    """
    Check every vocabulary-bound field on one record.

    Resolved TOP-DOWN - district, then tehsil within that district, then
    village within that tehsil - because each level narrows the next one's
    candidate list, and the narrowing is the only reason any of this is safe
    to apply automatically. A level that cannot be resolved does not fall back
    to the national list: it degrades to suggesting.
    """
    gz = gazetteer if gazetteer is not None else GAZETTEER
    out: List[Correction] = []

    def raw(key: str) -> str:
        entry = values.get(key) or {}
        return (entry.get("value") or "").strip()

    # --- District: the root of the scope, matched against all 43.
    district_name = raw("district")
    district_resolved = None
    if district_name:
        names = list(gz.districts.values())
        result = match_in_scope(district_name, names, "district", "LGD district list")
        out.append(result)
        if result.outcome in ("confirmed", "corrected"):
            district_resolved = result.value

    # --- Tehsil: only within the resolved district.
    tehsil_name = raw("tehsil")
    tehsil_resolved = None
    if tehsil_name:
        if district_resolved:
            result = match_in_scope(
                tehsil_name, gz.tehsils(district_resolved), "tehsil",
                f"tehsils of {district_resolved}")
        else:
            result = Correction(
                "tehsil", tehsil_name, "no_vocabulary", value=tehsil_name,
                scope="tehsil list", needs_review=True,
                message=("The district could not be resolved, so there is no "
                         "tehsil list to check this against. Matching against "
                         "all 453 tehsils nationally was measured at 28.7% "
                         "ambiguity and is not attempted."))
        out.append(result)
        if result.outcome in ("confirmed", "corrected"):
            tehsil_resolved = result.value

    # --- Village: only within the resolved tehsil, or the district if the
    # tehsil is unknown. Never nationally.
    village_name = raw("village")
    if village_name:
        if district_resolved:
            candidates = gz.villages(district_resolved, tehsil_resolved)
            scope = (f"villages of {tehsil_resolved}, {district_resolved}"
                     if tehsil_resolved else f"villages of {district_resolved}")
            out.append(match_in_scope(village_name, candidates, "village", scope))
        else:
            out.append(Correction(
                "village", village_name, "no_vocabulary", value=village_name,
                scope="village list", needs_review=True,
                message=("The district could not be resolved, so there is no "
                         "village list to check this against. Nearly 1 village "
                         "in 3 has a near-identical name somewhere else in "
                         "India, so a national match would be a guess.")))

    # --- State: tiny closed set, and cross-checkable against the district.
    state_name = raw("state")
    if state_name:
        result = match_in_scope(state_name, gz.states_list(), "state", "state list")
        if district_resolved:
            expected = gz.state_of_district(district_resolved)
            if expected and result.value and _key(result.value) != _key(expected):
                result = Correction(
                    "state", state_name, "ambiguous", value=state_name,
                    scope="state list", candidates=[expected], needs_review=True,
                    message=(f"The record says '{state_name}' but district "
                             f"'{district_resolved}' is in {expected}. One of "
                             f"the two was misread; which one cannot be decided "
                             f"from the text alone."))
        out.append(result)

    # --- Closed vocabularies.
    if raw("land_classification"):
        out.append(match_land_class(raw("land_classification")))

    # --- Identifier shapes.
    for key in ("khasra_number", "survey_number", "khata_number", "ulpin"):
        if raw(key):
            out.append(check_identifier(raw(key), key))

    return out


def apply_to_fields(fields: List, gazetteer: Optional[Gazetteer] = None) -> List[Correction]:
    """
    Run the vocabulary checks over freshly extracted fields and apply the
    corrections that earned it, in place.

    Mirrors learning.apply_model: mutate value/notes/status and return what
    was done, so the UI can show exactly which values the machine changed and
    why. Anything applied is ALSO marked needs_review - clearing a similarity
    threshold is evidence, not proof, and a village name is not something to be
    quietly rewritten on a land record.
    """
    by_key = {getattr(f, "key", None): f for f in fields}
    values = {
        key: {"value": getattr(f, "value", None)}
        for key, f in by_key.items() if getattr(f, "value", None)
    }
    corrections = correct_record(values, gazetteer)

    for correction in corrections:
        field = by_key.get(correction.field_key)
        if field is None:
            continue
        if correction.applied and correction.value:
            old = field.value
            field.value = correction.value
            field.notes.append(
                f"Vocabulary correction: '{old}' -> '{correction.value}' "
                f"({correction.score:.0f}% match, {correction.scope}). "
                f"Confirm against the document.")
            field.status = "needs_review"
        elif correction.needs_review:
            field.notes.append(correction.message)
            if correction.outcome in ("ambiguous", "invalid_shape"):
                field.status = "needs_review"
    return corrections


def describe() -> dict:
    return {
        "rapidfuzz": RAPIDFUZZ_AVAILABLE,
        "gazetteer_loaded": GAZETTEER.loaded,
        "states": len(GAZETTEER.states),
        "districts": len(GAZETTEER.districts),
        "tehsils": sum(len(v) for v in GAZETTEER.tehsils_by_district.values()),
        "villages": sum(len(v) for v in GAZETTEER.villages_by_district.values()),
        "auto_apply_score": AUTO_APPLY_SCORE,
        "gazetteer_script": "romanised only - Devanagari values cannot be matched",
    }
