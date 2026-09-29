"""
ML-based external fact-checking.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

validator.py checks a record against itself (business rules) and against other
already-ingested documents (duplicate detection). This module adds the third
leg the problem statement asks for: checking a record against an *external*
authority.

In production that authority is a DILRMP/LGD parcel-level registry reached
over an API. No hackathon team has state credentials for that, so this ships
a bundled JSON extract with the same shape (`backend/data/registry_master.json`)
- identical to how `AdminMaster` in validator.py stands in for the LGD
directory. Only the transport layer changes when a real registry is wired in.

Why ML and not another lookup table: OCR-extracted identity fields (khasra,
khata, village, district) rarely match the registry's spelling character for
character - "Narharpur" vs "नरहरपुर" transliterated, a dropped hyphen, a
misread digit. Exact-key lookup would silently return "no match" for a large
fraction of genuine hits. This module fits a scikit-learn character-n-gram
TF-IDF vectoriser over the registry's identity strings and finds the nearest
one by cosine similarity, so near-miss spellings still resolve to the right
parcel. The result is still fully explainable: every match reports which
registry record it picked and the similarity score that picked it, and every
mismatch names the exact field and the two values in conflict, rather than
an opaque "fraud score".

Degradation is honest, matching the rest of the system: if scikit-learn is
not installed, or the registry file cannot be read, fact-checking is skipped
with an explicit `info` issue rather than silently reporting nothing to check.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import field_extractor
from field_extractor import normalise

_HERE = os.path.dirname(os.path.abspath(__file__))
_REGISTRY_PATH = os.path.join(_HERE, "data", "registry_master.json")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

SKLEARN_AVAILABLE = _SKLEARN_OK

# A parcel's identity (khasra + khata + village + district) must be at least
# this similar to a registry entry's identity for the two to be treated as
# the same parcel.
MATCH_THRESHOLD = 0.55
# Below this string-similarity, two owner names are treated as different
# people rather than an OCR/transliteration variant of the same name.
OWNER_SIMILARITY_THRESHOLD = 0.72
AREA_WARN_TOLERANCE = 0.15    # >=15% area difference from the registry: warn
AREA_ERROR_TOLERANCE = 0.50   # >=50% area difference from the registry: block


@dataclass
class Issue:
    rule: str
    severity: str
    field: Optional[str]
    message: str
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _get(values: Dict[str, dict], key: str) -> Optional[str]:
    entry = values.get(key) or {}
    v = entry.get("value")
    return v if v not in (None, "") else None


def _identity_string(khasra: Optional[str], khata: Optional[str],
                      village: Optional[str], district: Optional[str]) -> str:
    parts = [khasra or "", khata or "", village or "", district or ""]
    cleaned = [re.sub(r"[^a-z0-9ऀ-ॿ]", "", normalise(p).lower()) for p in parts]
    return " ".join(cleaned)


def _normalise_identifier(value: Optional[str]) -> str:
    """
    A plot identifier reduced to the characters that carry its identity.

    Indic digits become ASCII, and separators/spacing are dropped, so the
    noise OCR genuinely introduces - '237/4' read as '2374', '२३७/४' for the
    same number - still compares equal. Nothing else is forgiven: the digits
    themselves must agree.
    """
    text = field_extractor.normalise_digits(str(value or ""))
    return re.sub(r"[^0-9a-z]", "", text.casefold())


def _identifiers_agree(extracted: Optional[str], registry: Optional[str]) -> bool:
    a, b = _normalise_identifier(extracted), _normalise_identifier(registry)
    return bool(a) and a == b


def _name_similarity(a: str, b: str) -> float:
    a2, b2 = normalise(a or "").lower().strip(), normalise(b or "").lower().strip()
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


class RegistryIndex:
    """
    Cross-database verification source: an authoritative parcel-level
    registry, retrieved by fuzzy identity match rather than an exact key,
    so OCR noise in the identity fields does not defeat the lookup.
    """

    def __init__(self, path: str = _REGISTRY_PATH):
        self.records: List[dict] = []
        self.loaded = False
        self._vectorizer = None
        self._matrix = None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.records = payload.get("records", [])
            self.loaded = True
        except Exception:
            self.loaded = False
        if self.loaded and _SKLEARN_OK and self.records:
            self._build_index()

    def _build_index(self) -> None:
        docs = [
            _identity_string(r.get("khasra_number"), r.get("khata_number"),
                             r.get("village"), r.get("district"))
            for r in self.records
        ]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self._matrix = self._vectorizer.fit_transform(docs)

    def best_match(self, khasra: Optional[str], khata: Optional[str],
                   village: Optional[str], district: Optional[str]
                   ) -> Tuple[Optional[dict], float]:
        if not (self.loaded and _SKLEARN_OK and self._vectorizer is not None):
            return None, 0.0
        query = _identity_string(khasra, khata, village, district)
        if not query.strip():
            return None, 0.0
        qvec = self._vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix)[0]
        idx = int(sims.argmax())
        return self.records[idx], float(sims[idx])


_REGISTRY = RegistryIndex()


def check(values: Dict[str, dict]) -> List[Issue]:
    """
    Fact-check one record against the external registry. Returns issues; an
    empty list means either "verified with nothing to flag" is not
    distinguished from "nothing to check" at the field level - the caller
    should read the FACT_CHECK_* rule ids for that distinction.
    """
    if not _SKLEARN_OK:
        return [Issue(
            "FACT_CHECK_UNAVAILABLE", "info", None,
            "This record was not cross-checked against the external registry, "
            "because that check is not available on this installation.",
            "Verify the owner name and plot identifier against the registry "
            "manually before approving.",
        )]
    if not _REGISTRY.loaded:
        return [Issue(
            "FACT_CHECK_NO_REGISTRY", "info", None,
            "External registry data could not be loaded; fact-check was skipped.",
            None,
        )]

    khasra = _get(values, "khasra_number")
    village = _get(values, "village")
    if not (khasra and village):
        return []          # not enough identity to look anything up

    khata = _get(values, "khata_number")
    district = _get(values, "district")
    match, score = _REGISTRY.best_match(khasra, khata, village, district)
    if not match or score < MATCH_THRESHOLD:
        return [Issue(
            "FACT_CHECK_NOT_FOUND", "info", None,
            "No matching entry found in the external registry for this parcel.",
            "The bundled registry is a partial extract; absence here is not "
            "proof of a problem.",
        )]

    # The similarity score alone is NOT sufficient to say "this is the same
    # parcel", and trusting it was a real bug: the identity string blends the
    # khasra and khata numbers with the village and district, so a record from
    # the right village scores well above MATCH_THRESHOLD even when its plot
    # number is completely different. A record for khasra 213/1 in नरहरपुर was
    # matched to registry entry 237/4 in नरहरपुर - a different plot - and then
    # reported as FACT_CHECK_OWNER_MISMATCH at *error* severity, blocking a
    # perfectly valid document because someone else's parcel had a different
    # owner. Confidently wrong, and exactly the failure this project exists to
    # avoid.
    #
    # Fuzzy matching is right for names, which carry OCR and transliteration
    # noise. It is wrong for a plot identifier: 213/1 and 218/1 differ by one
    # character and are different pieces of land. So the identifier must agree
    # exactly once separators and scripts are normalised away - which still
    # tolerates the noise that actually occurs (a dropped slash, Devanagari
    # digits, stray spacing) without ever equating two different plots.
    if not _identifiers_agree(khasra, match.get("khasra_number")):
        return [Issue(
            "FACT_CHECK_NOT_FOUND", "info", None,
            f"No registry entry for khasra {khasra} in this village. The closest "
            f"entry ({match.get('source_id', '?')}, khasra "
            f"{match.get('khasra_number')}) is a different parcel and was not "
            f"used for comparison.",
            "The bundled registry is a partial extract; absence here is not "
            "proof of a problem.",
        )]

    issues: List[Issue] = []
    source = match.get("source_id", "?")

    owner = _get(values, "owner_name")
    reg_owner = match.get("owner_name")
    if owner and reg_owner and _name_similarity(owner, reg_owner) < OWNER_SIMILARITY_THRESHOLD:
        issues.append(Issue(
            "FACT_CHECK_OWNER_MISMATCH", "error", "owner_name",
            f"Registry record {source} lists the owner as '{reg_owner}', which "
            f"does not match the extracted owner '{owner}'.",
            "Escalate to the revenue authority before approving; this may "
            "reflect an unrecorded transfer or a data-entry error.",
        ))

    extracted_sqm = (((values.get("area") or {}).get("extra") or {}).get("area") or {}).get("sqm")
    reg_area = match.get("area_sqm")
    if extracted_sqm and reg_area:
        rel_diff = abs(extracted_sqm - reg_area) / reg_area
        if rel_diff >= AREA_ERROR_TOLERANCE:
            issues.append(Issue(
                "FACT_CHECK_AREA_MISMATCH", "error", "area",
                f"Registry record {source} lists the area as {reg_area:.2f} sq.m, "
                f"but the extracted area is {extracted_sqm:.2f} sq.m "
                f"({rel_diff * 100:.0f}% difference).",
                "Verify against the source document before approving.",
            ))
        elif rel_diff >= AREA_WARN_TOLERANCE:
            issues.append(Issue(
                "FACT_CHECK_AREA_MISMATCH", "warning", "area",
                f"Registry record {source} lists the area as {reg_area:.2f} sq.m; "
                f"the extracted area ({extracted_sqm:.2f} sq.m) differs by "
                f"{rel_diff * 100:.0f}%.",
                "Could reflect a genuine subdivision or mutation - confirm.",
            ))

    classification = _get(values, "land_classification")
    reg_class = match.get("land_classification")
    if classification and reg_class and classification != reg_class:
        issues.append(Issue(
            "FACT_CHECK_CLASS_MISMATCH", "warning", "land_classification",
            f"Registry record {source} classifies this parcel as '{reg_class}', "
            f"but the document reads '{classification}'.",
            "Confirm whether the land use has genuinely changed.",
        ))

    if not issues:
        issues.append(Issue(
            "FACT_CHECK_VERIFIED", "info", None,
            f"Owner, area and classification agree with external registry "
            f"record {source} (identity match confidence {score * 100:.0f}%).",
            None,
        ))
    return issues
