"""
Validation engine.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The problem statement asks for "automated validation using business rules,
cross-database verification, and duplicate detection". This module implements
all three as explicit, auditable rules.

Every rule returns an Issue with a severity:

  error   -> the record cannot be accepted as-is (blocks approval)
  warning -> accepted, but a human should look (does not block)
  info    -> an assumption the system made and is disclosing

Design intent: in land administration a wrong record is far more expensive than
a slow one. So rules never silently "fix" data. They correct only where the
correction is provably safe, and otherwise they raise the issue to a verifier.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from field_extractor import (
    AREA_UNITS, FIELD_BY_KEY, LAND_CLASSES, normalise, parse_area,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MASTER_PATH = os.path.join(_HERE, "data", "admin_master.json")


# --------------------------------------------------------------------------
# Issue model
# --------------------------------------------------------------------------

@dataclass
class Issue:
    rule: str                 # stable rule id, e.g. "AREA_RANGE"
    severity: str             # error | warning | info
    field: Optional[str]      # field key, or None for document-level
    message: str
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Administrative master data (stands in for the LGD / DILRMP directory)
# --------------------------------------------------------------------------

class AdminMaster:
    """
    Cross-database verification source. In production this would be the Local
    Government Directory (LGD) and the state DILRMP village master, reached over
    an API. Here it is a bundled JSON extract with the same shape, so the rule
    logic is identical and only the transport changes.
    """

    def __init__(self, path: str = _MASTER_PATH):
        self.states: Dict[str, dict] = {}
        self.loaded = False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.states = payload.get("states", {})
            self.loaded = True
        except Exception:
            self.loaded = False

    @staticmethod
    def _norm(value: str) -> str:
        return re.sub(r"[^a-z\u0900-\u097f]", "", normalise(value or "").lower())

    def _all_districts(self) -> Dict[str, str]:
        out = {}
        for state, meta in self.states.items():
            for district in meta.get("districts", {}):
                out[self._norm(district)] = state
        return out

    def district_exists(self, district: str) -> Optional[str]:
        """Returns the state name if the district is known."""
        return self._all_districts().get(self._norm(district))

    def tehsil_in_district(self, tehsil: str, district: str) -> Optional[bool]:
        """True/False if the district is known, None if we cannot tell."""
        for meta in self.states.values():
            for dname, dmeta in meta.get("districts", {}).items():
                if self._norm(dname) == self._norm(district):
                    tehsils = [self._norm(t) for t in dmeta.get("tehsils", [])]
                    if not tehsils:
                        return None
                    return self._norm(tehsil) in tehsils
        return None

    def suggest_district(self, district: str) -> Optional[str]:
        """Nearest known district name, for OCR-mangled spellings."""
        target = self._norm(district)
        if not target:
            return None
        best, best_score = None, 0.0
        for state, meta in self.states.items():
            for dname in meta.get("districts", {}):
                cand = self._norm(dname)
                score = _dice(target, cand)
                if score > best_score:
                    best, best_score = dname, score
        return best if best_score >= 0.62 else None


def _dice(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ab = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    bb = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    inter = len(ab & bb)
    return 2.0 * inter / (len(ab) + len(bb)) if inter else 0.0


_MASTER = AdminMaster()


# --------------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------------

# Plausible parcel area bounds in square metres. Below the floor is almost
# always a unit error; above the ceiling is almost always an estate or an
# extra digit from OCR.
AREA_MIN_SQM = 5.0
AREA_MAX_SQM = 4_000_000.0        # 400 hectares


def _get(values: Dict[str, dict], key: str) -> Optional[str]:
    entry = values.get(key) or {}
    v = entry.get("value")
    return v if v not in (None, "") else None


def _conf(values: Dict[str, dict], key: str) -> float:
    return float((values.get(key) or {}).get("confidence") or 0.0)


def rule_required_fields(values: Dict[str, dict]) -> List[Issue]:
    issues = []
    for key, spec in FIELD_BY_KEY.items():
        if spec.required and _get(values, key) is None:
            issues.append(Issue(
                rule="REQUIRED_MISSING", severity="error", field=key,
                message=f"{spec.display} is mandatory but was not extracted.",
                suggestion="Enter the value manually from the source document.",
            ))
    return issues


def rule_area_sanity(values: Dict[str, dict]) -> List[Issue]:
    issues: List[Issue] = []
    entry = values.get("area") or {}
    raw = entry.get("value")
    if not raw:
        return issues

    parsed = (entry.get("extra") or {}).get("area") or parse_area(raw)
    if not parsed:
        issues.append(Issue("AREA_UNPARSED", "error", "area",
                            "Plot area could not be interpreted.",
                            "Re-enter as a number plus unit, e.g. '0.4820 hectare'."))
        return issues

    if parsed.get("unit_missing"):
        issues.append(Issue("AREA_UNIT_MISSING", "error", "area",
                            "Plot area has no unit, so it cannot be standardised.",
                            "Add the unit recorded on the document (hectare, bigha, sq.m)."))
        return issues

    sqm = parsed.get("sqm")
    if sqm is None:
        return issues

    if sqm <= 0:
        issues.append(Issue("AREA_NON_POSITIVE", "error", "area",
                            "Plot area is zero or negative.", "Correct the area value."))
    elif sqm < AREA_MIN_SQM:
        issues.append(Issue("AREA_RANGE", "warning", "area",
                            f"Plot area is implausibly small ({sqm:.2f} sq.m).",
                            "Check whether the unit was misread."))
    elif sqm > AREA_MAX_SQM:
        issues.append(Issue("AREA_RANGE", "warning", "area",
                            f"Plot area is implausibly large ({sqm / 10000:.2f} hectare).",
                            "Check for an extra digit introduced by OCR."))

    if parsed.get("regional_unit"):
        issues.append(Issue(
            "AREA_REGIONAL_UNIT", "info", "area",
            "Area uses a regional unit whose size varies by state; "
            f"standardised using the default factor ({sqm:.2f} sq.m).",
            "Confirm the state-specific conversion factor before publishing."))
    return issues


def rule_identifier_formats(values: Dict[str, dict]) -> List[Issue]:
    issues: List[Issue] = []
    for key in ("khasra_number", "khata_number", "survey_number",
                "mutation_number", "registration_number", "ulpin"):
        value = _get(values, key)
        spec = FIELD_BY_KEY[key]
        if not value or not spec.pattern:
            continue
        if not re.match(spec.pattern, value):
            issues.append(Issue(
                "FORMAT_INVALID", "warning", key,
                f"{spec.display} '{value}' does not match the expected format.",
                "Verify the value against the source document."))
        if re.fullmatch(r"0+", re.sub(r"[^\d]", "", value) or "x"):
            issues.append(Issue("IDENTIFIER_ZERO", "error", key,
                                f"{spec.display} is all zeros.", "Re-enter the identifier."))
    return issues


def rule_dates(values: Dict[str, dict]) -> List[Issue]:
    issues: List[Issue] = []
    today = _dt.date.today()

    parsed: Dict[str, Optional[_dt.date]] = {}
    for key in ("registration_date", "mutation_date"):
        value = _get(values, key)
        if not value:
            parsed[key] = None
            continue
        try:
            parsed[key] = _dt.date.fromisoformat(value)
        except Exception:
            parsed[key] = None
            issues.append(Issue("DATE_UNPARSED", "warning", key,
                                f"{FIELD_BY_KEY[key].display} could not be read as a date.",
                                "Enter the date as DD/MM/YYYY."))

    for key, d in parsed.items():
        if d is None:
            continue
        if d > today:
            issues.append(Issue("DATE_FUTURE", "error", key,
                                f"{FIELD_BY_KEY[key].display} ({d.isoformat()}) is in the future.",
                                "Correct the year - a common OCR digit error."))
        elif d.year < 1850:
            issues.append(Issue("DATE_TOO_OLD", "warning", key,
                                f"{FIELD_BY_KEY[key].display} ({d.isoformat()}) predates "
                                "modern land records.",
                                "Confirm the year on the document."))

    reg, mut = parsed.get("registration_date"), parsed.get("mutation_date")
    if reg and mut and mut < reg:
        issues.append(Issue(
            "DATE_ORDER", "error", "mutation_date",
            f"Mutation date ({mut.isoformat()}) is before the registration date "
            f"({reg.isoformat()}), which is not possible.",
            "Check whether the two dates were swapped."))
    return issues


def rule_administrative_hierarchy(values: Dict[str, dict]) -> List[Issue]:
    """Cross-database verification against the administrative master."""
    issues: List[Issue] = []
    if not _MASTER.loaded:
        issues.append(Issue("MASTER_UNAVAILABLE", "info", None,
                            "Administrative master data not loaded; "
                            "hierarchy checks were skipped.", None))
        return issues

    district = _get(values, "district")
    tehsil = _get(values, "tehsil")
    state = _get(values, "state")

    if district:
        found_state = _MASTER.district_exists(district)
        if not found_state:
            suggestion = _MASTER.suggest_district(district)
            issues.append(Issue(
                "DISTRICT_UNKNOWN", "warning", "district",
                f"District '{district}' was not found in the administrative master.",
                f"Closest known district: '{suggestion}'." if suggestion
                else "Verify the district spelling."))
        else:
            if state and _MASTER._norm(state) != _MASTER._norm(found_state):
                issues.append(Issue(
                    "STATE_MISMATCH", "error", "state",
                    f"District '{district}' belongs to {found_state}, "
                    f"but the record says '{state}'.",
                    f"Set state to '{found_state}' or re-check the district."))
            elif not state:
                issues.append(Issue(
                    "STATE_INFERRED", "info", "state",
                    f"State was not on the document; inferred as '{found_state}' "
                    f"from district '{district}'.",
                    f"Accept '{found_state}' if correct."))

            if tehsil:
                ok = _MASTER.tehsil_in_district(tehsil, district)
                if ok is False:
                    issues.append(Issue(
                        "TEHSIL_MISMATCH", "warning", "tehsil",
                        f"Tehsil '{tehsil}' is not listed under district '{district}'.",
                        "Confirm the tehsil, or update the master directory."))
    return issues


def rule_land_classification(values: Dict[str, dict]) -> List[Issue]:
    issues: List[Issue] = []
    value = _get(values, "land_classification")
    if not value:
        return issues
    if value not in LAND_CLASSES:
        issues.append(Issue(
            "CLASS_UNMAPPED", "warning", "land_classification",
            f"Land classification '{value}' is not in the controlled vocabulary.",
            "Map it to one of: " + ", ".join(sorted(LAND_CLASSES)[:6]) + ", ..."))
    return issues


def rule_ownership_consistency(values: Dict[str, dict]) -> List[Issue]:
    issues: List[Issue] = []
    owner = _get(values, "owner_name")
    father = _get(values, "father_name")
    share = _get(values, "share")

    if owner and father and normalise(owner).lower() == normalise(father).lower():
        issues.append(Issue(
            "OWNER_FATHER_SAME", "warning", "father_name",
            "Landowner and father/husband name are identical.",
            "Likely a column bleed during extraction - verify both names."))

    if share:
        m = re.match(r"^(\d{1,4})\s*/\s*(\d{1,4})$", share)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den == 0:
                issues.append(Issue("SHARE_INVALID", "error", "share",
                                    "Ownership share has a zero denominator.",
                                    "Re-enter the share fraction."))
            elif num > den:
                issues.append(Issue(
                    "SHARE_OVER_UNITY", "error", "share",
                    f"Ownership share {num}/{den} exceeds the whole parcel.",
                    "Check whether numerator and denominator were swapped."))
    return issues


def rule_low_confidence(values: Dict[str, dict], threshold: float = 0.80) -> List[Issue]:
    """Turns the confidence model into actionable review items."""
    issues: List[Issue] = []
    for key, entry in values.items():
        if entry.get("value") is None:
            continue
        conf = float(entry.get("confidence") or 0.0)
        if conf < threshold:
            spec = FIELD_BY_KEY.get(key)
            display = spec.display if spec else key
            severity = "error" if (spec and spec.required and conf < 0.55) else "warning"
            issues.append(Issue(
                "LOW_CONFIDENCE", severity, key,
                f"{display} extracted with {conf * 100:.0f}% confidence.",
                "Confirm against the highlighted region of the document."))
    return issues


# --------------------------------------------------------------------------
# Duplicate detection
# --------------------------------------------------------------------------

def parcel_signature(values: Dict[str, dict]) -> Optional[str]:
    """
    Canonical parcel key. Two documents describing the same parcel should
    produce the same signature even if spelling and spacing differ.
    """
    khasra = _get(values, "khasra_number")
    village = _get(values, "village")
    district = _get(values, "district")
    if not (khasra and village):
        return None
    norm = lambda s: re.sub(r"[^a-z0-9\u0900-\u097f]", "", normalise(s or "").lower())
    return f"{norm(khasra)}|{norm(village)}|{norm(district or '')}"


def rule_duplicates(values: Dict[str, dict], existing: List[dict]) -> List[Issue]:
    """
    `existing` is a list of {document_id, filename, signature, owner_name}
    for already-ingested records. Exact signature match = same parcel.
    """
    issues: List[Issue] = []
    sig = parcel_signature(values)
    if not sig:
        return issues

    owner = normalise(_get(values, "owner_name") or "").lower()
    for row in existing:
        if row.get("signature") != sig:
            continue
        other_owner = normalise(row.get("owner_name") or "").lower()
        if other_owner and owner and other_owner != owner:
            issues.append(Issue(
                "DUPLICATE_CONFLICT", "error", "khasra_number",
                f"Parcel already digitised as document #{row.get('document_id')} "
                f"({row.get('filename')}) but with a different owner "
                f"('{row.get('owner_name')}').",
                "Possible ownership conflict or an unrecorded mutation - "
                "escalate before approving."))
        else:
            issues.append(Issue(
                "DUPLICATE", "warning", "khasra_number",
                f"This parcel is already digitised as document "
                f"#{row.get('document_id')} ({row.get('filename')}).",
                "Confirm whether this is a re-scan or a newer mutation record."))
    return issues


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

ALL_RULES = [
    ("REQUIRED_MISSING", rule_required_fields),
    ("AREA", rule_area_sanity),
    ("FORMATS", rule_identifier_formats),
    ("DATES", rule_dates),
    ("HIERARCHY", rule_administrative_hierarchy),
    ("CLASSIFICATION", rule_land_classification),
    ("OWNERSHIP", rule_ownership_consistency),
]


def validate(values: Dict[str, dict], existing: Optional[List[dict]] = None,
             confidence_threshold: float = 0.80) -> dict:
    """
    Run every rule. Returns issues plus a routing decision.

    Decision values:
      auto_approved    -> no errors, no warnings, all confidences healthy
      needs_review     -> warnings or low confidence; a verifier must confirm
      blocked          -> at least one error; cannot be published as-is
    """
    issues: List[Issue] = []
    for _, fn in ALL_RULES:
        try:
            issues.extend(fn(values))
        except Exception as exc:                       # a rule must never crash ingestion
            issues.append(Issue("RULE_ERROR", "warning", None,
                                f"A validation rule failed to run: {exc}", None))

    issues.extend(rule_low_confidence(values, confidence_threshold))
    if existing:
        issues.extend(rule_duplicates(values, existing))

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        decision = "blocked"
    elif warnings:
        decision = "needs_review"
    else:
        decision = "auto_approved"

    # Document-level trust score: field confidence penalised by open issues.
    present = [v for v in values.values() if v.get("value") is not None]
    base = sum(float(v.get("confidence") or 0) for v in present) / len(present) if present else 0.0
    penalty = min(0.6, 0.18 * len(errors) + 0.06 * len(warnings))
    trust = max(0.0, round((base - penalty) * 100, 1))

    return {
        "decision": decision,
        "trust_score": trust,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "info_count": len(issues) - len(errors) - len(warnings),
        "issues": [i.to_dict() for i in issues],
        "signature": parcel_signature(values),
    }
