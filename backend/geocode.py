"""
Approximate geotagging from a record's place names.

WHAT THIS IS FOR, AND WHAT IT IS NOT

A land record's true location comes from its geometry: a cadastral parcel
matched by khasra number, a world file or GeoTIFF, at least three ground
control points, or a shapefile. Those paths live in cadastral.py and
georeference.py and they yield parcel-grade coordinates.

This module is the fallback for the common case where a document carries
none of that - a deed, an attorney grant, a mutation order - and the only
locational evidence in it is written place names. Measured on a real Delhi
general power of attorney, the extracted village field read

    "VILLAGE NARELA, SABOLI ROAD, DELHI"

and nothing on the sheet tied any pixel to the earth. A name is still worth
something: it puts the record in the right part of the country, which is
enough to cluster records, route them to the right revenue office, or drop a
pin on a dashboard.

It is NOT enough to do anything else, and the design reflects that:

  * The result always carries an accuracy_m, and it is kilometres, not
    metres. A district coordinate locates the DISTRICT.
  * It is emitted under its own rule name, never the parcel-grade
    GEO_TAGGED, so the two can never be confused downstream.
  * It is informational only. It never validates an area, never contradicts
    a boundary, and never blocks a record. A coordinate derived from a name
    cannot disagree with anything, because it contains no new information
    about the parcel.
  * Parcel geometry always wins. This runs only when no parcel matched.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import geocode_online
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import gazetteer as _gz

_DATA = os.path.join(os.path.dirname(__file__), "data", "place_coords.json")

# Broadest first so a more specific level can always override a coarser one.
_LEVELS = ("state", "district", "locality")
_SPECIFICITY = {name: i for i, name in enumerate(_LEVELS)}

# Fuzzy matching is deliberately tight. A place name is used to position a
# record on a map, and a plausible-looking wrong village is worse than no
# coordinate at all - the blank is visibly missing, the wrong pin is not.
_FUZZY_MIN_RATIO = 0.88
_FUZZY_MIN_LEN = 5

# Words that describe the STRUCTURE of an address rather than naming a place.
# Without stripping these, "VILLAGE NARELA" fuzzy-matches nothing and the
# segment is lost; with them stripped the token "narela" is left standing.
_NOISE = frozenset({
    "village", "vill", "gram", "gaon", "mauza", "mouza", "revenue",
    "tehsil", "tahsil", "taluk", "taluka", "mandal", "block", "pargana",
    "district", "distt", "dist", "state", "division", "circle",
    "road", "rd", "street", "marg", "lane", "gali", "colony", "extn",
    "extension", "sector", "phase", "po", "ps", "near", "opp", "opposite",
    "behind", "part", "no", "plot", "khasra", "khata", "area", "land",
    "property", "situated", "resident", "residing",
    # Devanagari equivalents - the same address furniture in Hindi.
    "गांव", "ग्राम",
    "मौजा", "तहसील",
    "जिला", "राज्य",
    "निवासी", "रोड",
    "मार्ग", "गली",
    "खसरा", "खाता",
})

_SPLIT_RE = re.compile(r"[,;/|\n]+")
_WORD_RE = re.compile(r"[A-Za-zऀ-ॿ]+")


@dataclass
class PlaceMatch:
    """One resolved place, with its precision stated in metres."""
    name: str
    level: str                      # "locality" | "district" | "state"
    lat: float
    lon: float
    accuracy_m: int
    matched_from: str               # the record field the name came from
    matched_text: str               # the token that actually matched
    exact: bool                     # False means a fuzzy match
    state: Optional[str] = None
    district: Optional[str] = None
    corroborated: bool = False      # a coarser level agreed with this one
    conflicts: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["conflicts"] = list(self.conflicts)
        return d


_VOWEL_RUN_RE = re.compile(r"([aeiou])\1+")

# Phonetic keys are lossy, so they are only consulted after an exact lookup
# fails, and only for names long enough that a collision is unlikely.
_PHON_MIN_LEN = 5


def _fold(text: str) -> str:
    """
    Case-fold, and collapse runs of the same vowel.

    Both halves are load-bearing, and both were found by measurement.

    gazetteer.normalise() deliberately does NOT change case. Without folding
    here, the token "NARELA" lifted from an all-caps deed never matches the
    table entry "Narela" - that single omission made the real Delhi GPA
    resolve to nothing at all while the tidier probe "Narela" resolved fine.

    The vowel run is the transliteration half. romanise_devanagari() writes
    long vowels doubled, so नरेला becomes "narelaa" and भोपाल "bhopaala".
    Collapsing the runs lands them on "narela" and "bhopala", which is close
    enough for the phonetic key to bridge; without it, bhopaala/bhopal scored
    0.857 and fell under the fuzzy threshold.
    """
    return _VOWEL_RUN_RE.sub(r"\1", (text or "").casefold())


def _norm_keys(text: str) -> List[str]:
    """
    Normalised forms of a name to try, most faithful first.

    A Devanagari name keeps its own spelling AND gains a romanised form, so
    it can match either a Devanagari alias in the table or a Latin entry.
    """
    out: List[str] = []
    base = _fold(_gz.normalise(text or ""))
    if base:
        out.append(base)
    if _gz.detect_script(text or "") == "devanagari":
        roman = _fold(_gz.normalise(_gz.romanise_devanagari(text)))
        if roman and roman not in out:
            out.append(roman)
    return out


def _phon_keys(text: str) -> List[str]:
    """Phonetic keys for a name. Only used once exact lookup has failed."""
    out: List[str] = []
    for key in _norm_keys(text):
        phon = _gz.phonetic_key(key)
        if phon and len(phon) >= _PHON_MIN_LEN and phon not in out:
            out.append(phon)
    return out


class PlaceIndex:
    """Loaded coordinate tables, keyed by normalised name."""

    def __init__(self) -> None:
        self.loaded = False
        self.accuracy: Dict[str, int] = {"locality": 5000, "district": 40000,
                                         "state": 300000}
        # normalised name -> list of records (a name can repeat across states)
        self.by_level: Dict[str, Dict[str, List[dict]]] = {
            lvl: {} for lvl in _LEVELS}
        self.by_phon: Dict[str, Dict[str, List[dict]]] = {
            lvl: {} for lvl in _LEVELS}
        self._load()

    def _add(self, level: str, name: str, rec: dict,
             aliases: Sequence[str] = ()) -> None:
        """
        Index one place under every spelling it is known by.

        Aliases matter more than they look. Land records are written across
        decades and use whichever name was official at the time, so a deed
        may say Allahabad where the table says Prayagraj, or Bombay where it
        says Mumbai City. They also carry the Devanagari spelling, which is
        the only reliable bridge for names the romaniser mangles: दिल्ली
        romanises to "dillii" and लखनऊ to "lakhanou", neither of which any
        phonetic key will ever join to "Delhi" or "Lucknow".
        """
        for spelling in (name, *aliases):
            for key in _norm_keys(spelling):
                self.by_level[level].setdefault(key, []).append(rec)
            for key in _phon_keys(spelling):
                self.by_phon[level].setdefault(key, []).append(rec)

    def _load(self) -> None:
        try:
            with open(_DATA, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            # Missing or corrupt reference data is not fatal. Approximate
            # geotagging is an optional enrichment, so it degrades to "no
            # coordinate" exactly like every other optional layer here.
            return

        self.accuracy.update(raw.get("accuracy_m") or {})

        for name, v in (raw.get("states") or {}).items():
            self._add("state", name, {"name": name, "level": "state",
                                      "lat": v["lat"], "lon": v["lon"],
                                      "state": name, "district": None},
                      v.get("aliases") or ())
        for state, ds in (raw.get("districts") or {}).items():
            for name, v in ds.items():
                self._add("district", name, {"name": name, "level": "district",
                                             "lat": v["lat"], "lon": v["lon"],
                                             "state": state, "district": name},
                          v.get("aliases") or ())
        for name, v in (raw.get("localities") or {}).items():
            self._add("locality", name, {"name": name, "level": "locality",
                                         "lat": v["lat"], "lon": v["lon"],
                                         "state": v.get("state"),
                                         "district": v.get("district")},
                      v.get("aliases") or ())
        self.loaded = any(self.by_level[lvl] for lvl in _LEVELS)

    def lookup(self, token: str, level: str) -> Tuple[List[dict], bool]:
        """
        Resolve one token at one level. Returns (records, exact).

        Three passes, strictest first, because the cost of a wrong match is
        asymmetric: a missing coordinate is visibly missing, while a
        confident pin on the wrong village is not.

          1. exact, on the normalised spelling or any indexed alias
          2. phonetic, for a Devanagari spelling the romaniser distorted
          3. fuzzy, for OCR damage in an otherwise Latin name

        Only pass 1 reports exact=True; 2 and 3 mark the match inexact so a
        reader can see the coordinate rests on an inference.
        """
        table = self.by_level.get(level) or {}
        if not table:
            return [], False
        keys = _norm_keys(token)
        for key in keys:
            if key in table:
                return table[key], True

        phon_table = self.by_phon.get(level) or {}
        for key in _phon_keys(token):
            if key in phon_table:
                return phon_table[key], False

        for key in keys:
            if len(key) < _FUZZY_MIN_LEN:
                continue
            near = difflib.get_close_matches(key, list(table),
                                             n=1, cutoff=_FUZZY_MIN_RATIO)
            if near:
                return table[near[0]], False
        return [], False

    def describe(self) -> dict:
        return {
            "loaded": self.loaded,
            "source": _DATA if self.loaded else None,
            "counts": {lvl: len({r["name"] for rs in self.by_level[lvl].values()
                                 for r in rs}) for lvl in _LEVELS},
            "accuracy_m": dict(self.accuracy),
        }


def _candidate_tokens(value: str) -> List[str]:
    """
    Place-name candidates from one free-text field value.

    Comma segments are tried whole before individual words, because a real
    name is often two words ("North West Delhi", "Greater Noida") and word
    tokens alone would match only its last one.
    """
    if not value:
        return []
    cands: List[str] = []
    for seg in _SPLIT_RE.split(value):
        words = [w for w in _WORD_RE.findall(seg)
                 if _fold(_gz.normalise(w)) not in _NOISE]
        if not words:
            continue
        whole = " ".join(words)
        if whole not in cands:
            cands.append(whole)
        # Sliding n-grams, longest first, so "New Delhi" is tried before the
        # bare "Delhi" that would otherwise shadow it.
        for n in (3, 2):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if gram not in cands:
                    cands.append(gram)
        for w in words:
            if w not in cands:
                cands.append(w)
    return cands


# Which record fields can name which level. A village field naming a state
# is normal in practice ("VILLAGE NARELA, SABOLI ROAD, DELHI"), so the
# village field is allowed to supply any level.
_FIELD_LEVELS = {
    "village": ("locality", "district", "state"),
    "tehsil": ("locality", "district", "state"),
    "district": ("district", "state"),
    "state": ("state",),
}


def _resolve_online(values: Dict[str, dict]) -> Optional[PlaceMatch]:
    """Ask the configured online gazetteer. Never raises."""
    def get(key):
        v = (values.get(key) or {}).get("value")
        return v.strip() if isinstance(v, str) and v.strip() else None
    try:
        hit = geocode_online.lookup(get("village"), get("tehsil"),
                                    get("district"), get("state"))
    except Exception:
        return None
    if not hit:
        return None
    return PlaceMatch(
        name=hit.get("matched") or "online result",
        level=hit["level"], lat=hit["lat"], lon=hit["lon"],
        accuracy_m=int(hit["accuracy_m"]),
        matched_from="online gazetteer",
        matched_text=(get("village") or get("district") or get("state") or ""),
        exact=False,                     # a gazetteer match is not a table hit
        state=get("state"), district=get("district"))


def resolve(values: Dict[str, dict]) -> Optional[PlaceMatch]:
    """
    Best approximate coordinate for a record, or None.

    `values` is the extractor's field map ({key: {"value": ...}}). The most
    specific level that matches wins, and a coarser match that agrees with
    it is recorded as corroboration rather than discarded - a document that
    says both "Narela" and "Delhi" is more trustworthy than one that says
    only "Narela", and one that says "Narela" and "Kerala" is flagged.
    """
    found: List[PlaceMatch] = []
    for field, levels in _FIELD_LEVELS.items():
        value = (values.get(field) or {}).get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        for token in _candidate_tokens(value):
            for level in levels:
                recs, exact = INDEX.lookup(token, level)
                if not recs:
                    continue
                for rec in recs:
                    found.append(PlaceMatch(
                        name=rec["name"], level=level,
                        lat=rec["lat"], lon=rec["lon"],
                        accuracy_m=int(INDEX.accuracy.get(level, 300000)),
                        matched_from=field, matched_text=token, exact=exact,
                        state=rec.get("state"), district=rec.get("district")))
                break        # this token resolved; do not also match it coarser

    # An online gazetteer, when one is configured, before settling for what
    # the bundled table can reach.
    #
    # The table is a demo extract: measured on the real corpus it answered
    # "Jaunpur" with the centroid of Uttar Pradesh, 300 km away. Asking a
    # real gazetteer first turns that into the district, 40 km. Village level
    # is NOT reached this way and the code does not pretend otherwise -
    # OpenStreetMap has no entry for the revenue villages in this corpus
    # (Amari, Bikapur, Narharpur all return nothing), so an installation
    # wanting village centroids must point GEOCODER_URL at a gazetteer that
    # carries them, such as Bhuvan.
    #
    # It runs only when the offline path did WORSE, so enabling it can raise
    # accuracy and cannot lower it.
    best_offline = min((m.accuracy_m for m in found), default=None)
    if geocode_online.available():
        online = _resolve_online(values)
        if online is not None and (best_offline is None
                                   or online.accuracy_m < best_offline):
            found.append(online)

    if not found:
        return None

    states = {m.name for m in found if m.level == "state"}
    districts = {m.name for m in found if m.level == "district"}

    def rank(m: PlaceMatch) -> tuple:
        agrees = (m.state in states) if (states and m.state) else False
        return (_SPECIFICITY[m.level], m.exact, agrees, -len(m.matched_text))

    # Drop candidates whose parent state contradicts a state named outright.
    # The record's own state field is the more reliable of the two, so a
    # village that disagrees with it loses - but it is REPORTED rather than
    # silently discarded. A document naming both Narela and Kerala has
    # something wrong with it, and hiding that behind a confident Kerala pin
    # would be the worst of the available outcomes.
    consistent = [m for m in found
                  if not states or not m.state or m.state in states]
    rejected = [m for m in found if m not in consistent]
    pool = consistent or found
    best = max(pool, key=rank)

    best.corroborated = bool(
        (best.level != "state" and best.state and best.state in states)
        or (best.level == "locality" and best.district
            and best.district in districts))

    named = ", ".join(sorted(states))
    conflicts: List[str] = []
    for name, state in sorted({(m.name, m.state) for m in rejected}):
        conflicts.append(f"{name} is in {state}, but the record names {named}")
    if not consistent and states:
        conflicts.append(
            "no place named in this record agrees with its stated state")
    best.conflicts = tuple(conflicts)
    return best


INDEX = PlaceIndex()


def describe() -> dict:
    return INDEX.describe()
