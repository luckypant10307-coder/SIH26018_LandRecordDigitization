"""
HTTP API + ingestion pipeline.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Standard library only (http.server + sqlite3). No pip install, no build step,
no internet. Start it with `python3 run.py` and it serves both the JSON API and
the front-end.

Role-based access control is enforced on every mutating endpoint:

  operator -> upload documents, correct fields
  verifier -> everything an operator can do, plus approve / reject records
  admin    -> everything, plus retraining and export
  auditor  -> read-only (can read the audit trail, cannot change anything)
"""

from __future__ import annotations

import copy
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anomaly_detector
import auth
import bhashini
import cadastral
import document_authenticity
import doc_type as doc_type_mod
import fact_checker
import field_extractor
import gazetteer
import geocode
import georeference
import handwriting
import learning
import llm_extractor
import ocr_engine
import shapefile_import
import table_structure
import topology
import validator as validator_mod
from db import Database
from field_extractor import (
    FIELD_BY_KEY, FIELD_SPECS, extract_fields, fields_to_dict, summarise,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
STORAGE_DIR = os.path.join(ROOT, "storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
WORK_DIR = os.path.join(STORAGE_DIR, "work")
SAMPLES_DIR = os.path.join(ROOT, "samples")

for _d in (STORAGE_DIR, UPLOAD_DIR, WORK_DIR):
    os.makedirs(_d, exist_ok=True)

# LANDRECORDS_DB relocates the SQLite file - for a mounted volume, or to run
# against a throwaway database without touching the real one. Ignored entirely
# when DATABASE_URL selects Postgres, which carries its own location.
DB = Database(os.environ.get("LANDRECORDS_DB")
              or os.path.join(STORAGE_DIR, "landrecords.db"))

# Windows MAX_PATH, third encounter in this codebase (after api_seed's whole
# batch aborting on one long filename, and ocr_engine silently failing to
# write preprocessed images). The obvious "<timestamp>_<original name>"
# stored-file name overflowed 260 characters here - this project's upload
# directory alone is 212 deep, so
# "1788880550001_sample_04_bigha_biswa_rajasthan.pdf" reached 262 and
# shutil.copyfile failed with a bare "No such file or directory". Four of
# the fourteen bundled samples, including all three scans, silently failed
# to load because of it.
#
# The stored name does not need to be human-readable: documents.filename
# keeps the original for display and stored_path is only ever resolved
# through the database, so a short unique name costs nothing and removes a
# whole bug class rather than pushing the limit a little further away.
_MAX_WINDOWS_PATH = 260


def _stored_upload_path(original_name: str) -> str:
    ext = os.path.splitext(original_name)[1].lower()[:8]
    stored = os.path.join(
        UPLOAD_DIR, f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}")
    if os.name == "nt" and len(os.path.abspath(stored)) >= _MAX_WINDOWS_PATH:
        # Even the short form does not fit: say so loudly rather than let
        # copyfile fail later with an error that names no cause.
        raise ApiError(500, (
            f"Storage path is too long for this operating system "
            f"({len(os.path.abspath(stored))} chars, limit {_MAX_WINDOWS_PATH}). "
            "Move this project to a shorter directory, e.g. C:\\landrecords."))
    return stored

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
ALLOWED_EXT = {
    # Scans and native PDFs - the OCR and text-layer paths.
    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".txt", ".md",
    # Office and web documents, read by backend/office_reader.py with the
    # standard library alone. A revenue office holds many more of these than
    # it does clean scans: a clerk's khatauni extract is usually .docx, a
    # district parcel list .xlsx, and offices running LibreOffice - which most
    # government installations do - produce .odt and .ods.
    ".docx", ".odt", ".xlsx", ".ods", ".pptx",
    ".csv", ".tsv", ".html", ".htm", ".rtf",
}

ROLE_RIGHTS = {
    "operator": {"upload", "correct"},
    "verifier": {"upload", "correct", "approve", "reject", "revalidate"},
    "admin": {"upload", "correct", "approve", "reject", "revalidate", "retrain", "export", "purge"},
    "auditor": set(),
}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------
# Multipart parsing (the stdlib `cgi` module was removed in Python 3.13)
# --------------------------------------------------------------------------

def parse_multipart(body: bytes, content_type: str) -> List[dict]:
    """Minimal but correct multipart/form-data parser. Returns part dicts."""
    m = re.search(r'boundary="?([^";]+)"?', content_type or "", re.I)
    if not m:
        raise ApiError(400, "Malformed upload: no multipart boundary.")
    boundary = m.group(1).encode()
    delim = b"--" + boundary

    parts: List[dict] = []
    for raw in body.split(delim):
        if raw in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        raw = raw.lstrip(b"\r\n")
        if raw.startswith(b"--"):
            break
        head, _, payload = raw.partition(b"\r\n\r\n")
        if not _:
            continue
        headers: Dict[str, str] = {}
        for line in head.decode("utf-8", "replace").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        disp = headers.get("content-disposition", "")
        name = re.search(r'name="([^"]*)"', disp)
        filename = re.search(r'filename="([^"]*)"', disp)
        parts.append({
            "name": name.group(1) if name else None,
            "filename": filename.group(1) if filename else None,
            "content_type": headers.get("content-type"),
            "data": payload[:-2] if payload.endswith(b"\r\n") else payload,
        })
    return parts


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "document")
    name = re.sub(r"[^A-Za-z0-9._\u0900-\u097f -]", "_", name).strip() or "document"
    return name[:150]


# --------------------------------------------------------------------------
# Document integrity signals (signature / seal / stamp-paper presence)
# --------------------------------------------------------------------------

# A department/management seal is mandatory on every land record document
# in this domain, not just ones that already claim to be registered - a
# missing one blocks the record, the same way a missing required field
# does. A signature is held to the lighter, registration-gated standard
# below. Either way, see document_authenticity.py's module docstring for
# why this checks *presence*, never authenticity: a detected seal is not
# proof it is genuine, and a missed one may be a detector false negative,
# not proof there is none - a verifier can look at the actual page either way.
_REGISTERED_INSTRUMENT_FIELDS = ("registration_number", "mutation_number")


def _add_authenticity_issues(result: dict, values: Dict[str, dict],
                             render_path: Optional[str]) -> None:
    """Mutates `result["issues"]` in place with document-integrity signals."""
    if not document_authenticity.CV_AVAILABLE:
        result["issues"].append({
            "rule": "AUTHENTICITY_CHECK_UNAVAILABLE", "severity": "info", "field": None,
            "message": "OpenCV/numpy not installed; signature/seal/stamp-paper "
                       "presence detection was skipped.",
            "suggestion": None,
        })
        return
    if not render_path or not os.path.exists(render_path):
        return    # nothing rendered to look at (e.g. a plain .txt upload)

    try:
        signals = document_authenticity.analyze_document(render_path)
    except Exception as exc:
        result["issues"].append({
            "rule": "AUTHENTICITY_CHECK_FAILED", "severity": "info", "field": None,
            "message": f"Signature/seal/stamp-paper detection failed to run: {exc}",
            "suggestion": None,
        })
        return

    is_registered = any(values.get(k, {}).get("value") for k in _REGISTERED_INSTRUMENT_FIELDS)

    if signals.stamp_paper_detected:
        result["issues"].append({
            "rule": "STAMP_PAPER_DETECTED", "severity": "info", "field": None,
            "message": "Page appears to be printed on India Non-Judicial stamp paper "
                      + (f"({signals.stamp_paper_text})." if signals.stamp_paper_text else "."),
            "suggestion": None,
        })

    if signals.seals:
        result["issues"].append({
            "rule": "SEAL_DETECTED", "severity": "info", "field": None,
            "message": f"{len(signals.seals)} seal-like mark(s) detected on the page. "
                       "This confirms a stamp-shaped mark is present, not whose seal it is.",
            "suggestion": None,
        })
    else:
        result["issues"].append({
            "rule": "SEAL_MISSING", "severity": "warning", "field": None,
            "message": "No department/management seal-like mark was detected on the page. "
                      "A seal is mandatory on land record documents.",
            "suggestion": "Confirm the source document actually bears an official seal; "
                         "a faint, small, or unusually coloured stamp may need a clearer "
                         "scan for automatic detection to find it.",
        })

    if signals.signature_detected:
        result["issues"].append({
            "rule": "SIGNATURE_DETECTED", "severity": "info", "field": None,
            "message": "A signature-like mark was detected in the expected region. "
                       "This confirms a mark is present, not whose signature it is.",
            "suggestion": None,
        })
    elif is_registered:
        result["issues"].append({
            "rule": "SIGNATURE_MISSING", "severity": "warning", "field": None,
            "message": "Record has a registration/mutation number but no signature-like "
                      "mark was detected on the page.",
            "suggestion": "Confirm the source document is actually signed.",
        })

    # A MISSING seal refers the record for review; it does not block it.
    #
    # This blocked, and was wrong for the reason its own detector documents:
    # a miss may be a false negative, not proof there is no seal. Blocking
    # asserts "this document has no seal" on evidence that only supports "we
    # did not find one" - the exact move _add_handwriting_issue refuses to
    # make a few lines below, where silence deliberately means "nothing stood
    # out" rather than "certainly all print".
    #
    # Measured on the real Bhu-Naksha corpus: after the document-type fix
    # removed every REQUIRED_MISSING error, SEAL_MISSING was the ONLY thing
    # still blocking the scanned plot reports - and a Bhu-Naksha download is
    # a digitally generated map extract that carries "Signatory/Officer" and
    # a timestamp rather than an inked departmental seal. There is nothing
    # for the detector to find, and the record is not defective for it.
    #
    # Review is the right destination: a verifier opens the page and looks,
    # which is the only thing that can actually settle it.
    seal_missing_count = sum(1 for i in result["issues"] if i["rule"] == "SEAL_MISSING")
    if seal_missing_count:
        for issue in result["issues"]:
            if issue["rule"] == "SEAL_MISSING":
                issue["severity"] = "warning"
        result["warning_count"] = result.get("warning_count", 0) + seal_missing_count
        if result["decision"] == "auto_approved":
            result["decision"] = "needs_review"

    signature_missing_count = sum(1 for i in result["issues"] if i["rule"] == "SIGNATURE_MISSING")
    if signature_missing_count:
        result["warning_count"] = result.get("warning_count", 0) + signature_missing_count
        if result["decision"] == "auto_approved":
            result["decision"] = "needs_review"


# --------------------------------------------------------------------------
# Table/layout structure (see table_structure.py - optional, informational
# only, never affects the validation decision)
# --------------------------------------------------------------------------

def _add_handwriting_issue(result: dict, extraction) -> None:
    """
    Report lines that look handwritten rather than printed.

    Raised at WARNING severity, which in this system's severity model
    downgrades an otherwise auto-approved document to needs_review without
    blocking it. That is the correct weight: handwriting on a khatauni is
    completely normal - a Patwari's mutation entry or a marginal note - so it
    is not an error. What it is not is safely machine-readable, because
    Tesseract's LSTM is trained on print and returns confident-looking text
    for handwriting anyway. A human has to confirm those lines.

    Note the deliberate asymmetry: this fires when handwriting IS suspected,
    and stays silent when no line was flagged. Silence here means "nothing
    stood out", never "this page is certainly all print" - the detector's
    recall on real handwriting is unverified (see backend/handwriting.py), so
    claiming the negative would be claiming something unmeasured.
    """
    if handwriting is None or not handwriting.available():
        result["issues"].append({
            "rule": "HANDWRITING_CHECK_UNAVAILABLE", "severity": "info", "field": None,
            "message": "No printed-text profile is installed, so handwritten "
                       "entries were not detected on this page.",
            "suggestion": "Run tools/fit_print_profile.py --install to enable "
                          "handwriting detection.",
        })
        return

    flagged = [ln for ln in getattr(extraction, "lines", [])
               if (ln.handwriting or {}).get("is_handwriting_suspected")]
    if not flagged:
        return

    worst = max(flagged, key=lambda ln: (ln.handwriting or {}).get("worst_z", 0.0))
    sample = ", ".join(repr(ln.text[:28]) for ln in flagged[:3])
    result["issues"].append({
        "rule": "HANDWRITING_SUSPECTED", "severity": "warning", "field": None,
        "message": (f"{len(flagged)} line(s) on this document look handwritten "
                    f"rather than printed (e.g. {sample}). "
                    + handwriting.explain(worst.handwriting)),
        "suggestion": "Enter or confirm these values by hand. Tesseract is "
                      "trained on printed text and does not report failure on "
                      "handwriting - it returns plausible text instead.",
    })


def _recover_places_from_prose(values: Dict[str, dict], lines) -> None:
    """
    Fill EMPTY place fields from names appearing in the document's prose.

    Mutates `values` in place. A field that already holds a value is left
    alone: a value printed in its own field outranks one mentioned in a
    sentence, and silently replacing the first with the second would be a
    downgrade disguised as an improvement.

    Everything filled here is marked `recovered_from` so the provenance
    survives into the UI and the audit trail - a reviewer must be able to
    see that this district was inferred from a sentence about a past sale
    rather than read off the record.
    """
    empty = [k for k in ("village", "tehsil", "district")
             if not (values.get(k) or {}).get("value")]
    if not empty:
        return
    try:
        text = "\n".join(getattr(l, "text", str(l)) for l in (lines or []))
        found = gazetteer.places_from_prose(text)
    except Exception:
        return
    for key in empty:
        hit = found.get(key)
        if not hit:
            continue
        slot = values.setdefault(key, {"value": None, "confidence": 0.0, "extra": {}})
        slot["value"] = hit["value"]
        slot["confidence"] = hit["confidence"]
        extra = slot.get("extra")
        if not isinstance(extra, dict):
            extra = {}
        extra["recovered_from"] = hit["evidence"]
        extra["source"] = "prose"
        slot["extra"] = extra


def _add_doc_type_issue(result: dict, doctype: dict) -> None:
    """
    Record what kind of document this was taken to be, and why.

    Informational, never blocking. The type decision changes which fields are
    demanded, so a reviewer who disagrees with it needs to see the evidence
    that produced it rather than having to guess why a field went unasked.
    An unrecognised document is reported as such and keeps the full
    Record-of-Rights schema - "we could not identify this" is a finding, and
    quietly relaxing the requirements on it would hide that.
    """
    if not doctype:
        return
    if doctype.get("claimed"):
        message = f"Identified as {doctype['display']}"
        if doctype.get("carrier_display"):
            message += f", executed on {doctype['carrier_display']}"
        message += f". Matched on: {', '.join(doctype.get('evidence') or [])}."
        if doctype["type"] != "record_of_rights":
            message += (" Fields that this document type does not carry were "
                        "not required of it.")
    else:
        message = (f"Document type not recognised ({doctype.get('reason')}). "
                   f"The full Record-of-Rights schema was applied.")
    result["issues"].append({
        "rule": "DOCUMENT_TYPE", "severity": "info", "field": None,
        "message": message, "suggestion": None,
    })


def _add_vocabulary_issues(result: dict, corrections: List) -> None:
    """
    Report what the vocabulary pass did, at a severity matching its risk.

    A value that was CHANGED is a warning, not information: the machine
    rewrote a place name on a land record and a human has to agree. An
    ambiguous or malformed one is a warning because it needs a decision. A
    value simply absent from the bundled extract is only info - the extract
    is a partial snapshot and absence is not evidence of error.
    """
    severity_by_outcome = {
        "corrected": "warning",
        "ambiguous": "warning",
        "invalid_shape": "warning",
        "suggested": "info",
        "not_found": "info",
        "no_vocabulary": "info",
    }
    for correction in corrections:
        if correction.outcome == "confirmed":
            continue
        severity = severity_by_outcome.get(correction.outcome)
        if severity is None:
            continue
        result["issues"].append({
            "rule": "VOCAB_" + correction.outcome.upper(),
            "severity": severity,
            "field": correction.field_key,
            "message": correction.message,
            "suggestion": (", ".join(correction.candidates)
                           if correction.candidates else None),
        })


def _add_structure_issue(result: dict, render_path: Optional[str], work_dir: str) -> None:
    """Mutates `result["issues"]` in place. Purely informational - unlike the
    authenticity/anomaly checks, this never changes error_count, warning_count
    or decision, because it reports document structure, not a pass/fail signal."""
    if not table_structure.PADDLE_STRUCTURE_AVAILABLE:
        message = "PaddleOCR/PP-StructureV3 not installed; table/layout structure analysis was skipped."
        if table_structure._PYTHON_TOO_NEW:
            message += (" (PaddlePaddle does not yet support this Python version - "
                        "3.8-3.12 is required.)")
        result["issues"].append({
            "rule": "TABLE_STRUCTURE_UNAVAILABLE", "severity": "info", "field": None,
            "message": message, "suggestion": None,
        })
        return
    if not render_path or not os.path.exists(render_path):
        return

    try:
        structure = table_structure.analyze(render_path, work_dir)
    except Exception as exc:
        result["issues"].append({
            "rule": "TABLE_STRUCTURE_FAILED", "severity": "info", "field": None,
            "message": f"Table/layout structure analysis failed to run: {exc}",
            "suggestion": None,
        })
        return

    if not structure.available:
        result["issues"].append({
            "rule": "TABLE_STRUCTURE_FAILED", "severity": "info", "field": None,
            "message": "Table/layout structure analysis did not produce a result"
                      + (f" ({'; '.join(structure.warnings)})" if structure.warnings else "."),
            "suggestion": None,
        })
        return

    if structure.table_count:
        result["issues"].append({
            "rule": "TABLE_STRUCTURE_DETECTED", "severity": "info", "field": None,
            "message": f"{structure.table_count} table region(s) recognised on the page "
                      "(layout/structure, separate from the 17 land-record fields above).",
            "suggestion": None,
        })


# --------------------------------------------------------------------------
# LLM-assisted field suggestions (see llm_extractor.py - optional, off by
# default, requires explicit consent because it sends document text to a
# third party. Purely informational: never affects error_count,
# warning_count or decision, and never overwrites an extracted value.)
# --------------------------------------------------------------------------

def _add_llm_suggestions(result: dict, values: Dict[str, dict], full_text: str) -> None:
    """Mutates result["issues"] in place. Deliberately does NOT report an
    'unavailable' info issue the way every purely-local optional dependency
    in this project does (compare _add_structure_issue) - advertising
    'install/configure this to enable' the same way would undersell that
    turning this on means document text starts leaving the machine. Status
    is still visible in run.py --check, just not repeated on every
    document."""
    if not llm_extractor.LLM_AVAILABLE:
        return

    eligible = [k for k, v in values.items()
               if v.get("value") is None
               or float(v.get("confidence") or 0.0) < llm_extractor.CONFIDENCE_THRESHOLD]
    if not eligible:
        return

    try:
        suggestions = llm_extractor.suggest_fields(full_text, eligible)
    except Exception as exc:
        result["issues"].append({
            "rule": "LLM_SUGGESTION_FAILED", "severity": "info", "field": None,
            "message": f"LLM field-suggestion request failed: {exc}",
            "suggestion": None,
        })
        return

    for key, sug in suggestions.items():
        spec = FIELD_BY_KEY.get(key)
        display = spec.display if spec else key
        result["issues"].append({
            "rule": "LLM_SUGGESTION", "severity": "info", "field": key,
            "message": f"LLM suggests {display} = \"{sug['suggested_value']}\" "
                      f"(from {sug['source']}) - not verified, review against "
                      "the source document before accepting.",
            "suggestion": "Confirm against the source document before accepting this value.",
        })


# --------------------------------------------------------------------------
# Anomaly baseline (see anomaly_detector.py for what this is and is not)
# --------------------------------------------------------------------------

_ANOMALY_MODEL = None
_ANOMALY_MODEL_LOAD_ATTEMPTED = False


def _anomaly_model():
    """Lazy-load and cache: a fresh install has no baseline yet, and every
    ingestion should not re-read the pickle file from disk to find that out."""
    global _ANOMALY_MODEL, _ANOMALY_MODEL_LOAD_ATTEMPTED
    if not _ANOMALY_MODEL_LOAD_ATTEMPTED:
        _ANOMALY_MODEL = anomaly_detector.load_model()
        _ANOMALY_MODEL_LOAD_ATTEMPTED = True
    return _ANOMALY_MODEL


def _add_anomaly_issue(result: dict, summary: dict, quality: dict) -> None:
    """Mutates `result["issues"]` in place with the anomaly-baseline signal."""
    if not anomaly_detector.SKLEARN_AVAILABLE:
        result["issues"].append({
            "rule": "ANOMALY_CHECK_UNAVAILABLE", "severity": "info", "field": None,
            "message": "scikit-learn is not installed; anomaly-baseline comparison was skipped.",
            "suggestion": None,
        })
        return

    model = _anomaly_model()
    if model is None:
        result["issues"].append({
            "rule": "ANOMALY_BASELINE_NOT_TRAINED", "severity": "info", "field": None,
            "message": f"No anomaly baseline yet - needs at least "
                      f"{anomaly_detector.MIN_TRAINING_SAMPLES} approved documents, "
                      "then an admin can retrain from the Learning tab.",
            "suggestion": None,
        })
        return

    doc_like = {
        "trust_score": result.get("trust_score"),
        "error_count": result.get("error_count"),
        "warning_count": result.get("warning_count"),
        "summary": summary,
        "quality": quality,
        "issues": result["issues"],
    }
    features = anomaly_detector.extract_features(doc_like)
    is_anomaly, raw_score = anomaly_detector.score(model, features)
    if is_anomaly:
        result["issues"].append({
            "rule": "ANOMALY_DETECTED", "severity": "warning", "field": None,
            "message": "This record's overall pattern (confidence, completeness, seal/"
                      "signature presence) is statistically unusual compared to "
                      "previously approved documents. This flags it as worth a second "
                      "look, not as fake - an unusual but genuine record scores the "
                      "same way an overlooked problem would.",
            "suggestion": "Review against the source document before approving.",
        })
        result["warning_count"] = result.get("warning_count", 0) + 1
        if result["decision"] == "auto_approved":
            result["decision"] = "needs_review"


# Ordered, and the order is a dependency chain: a tehsil can only be checked
# once its district is in Latin, and a village only once its tehsil is.
_SCRIPT_BRIDGED_FIELDS = ("district", "state", "tehsil", "village")


def _find_district_entry(master, district: Optional[str]):
    """The master's (canonical name, record) for a district, matched loosely."""
    if not district:
        return None, None
    target = master._norm(district)
    for state in master.states.values():
        for name, record in (state.get("districts") or {}).items():
            if master._norm(name) == target:
                return name, record
    return None, None


def _canonical_place(master, kind: str, value: str,
                     district: Optional[str] = None,
                     tehsil: Optional[str] = None) -> Optional[str]:
    """
    The master's own spelling of `value`, or None if the master does not know it.

    Returning the canonical form rather than a boolean is what keeps
    "uttar pradesh" from being written into a land record: the transliteration
    only has to be close enough to identify the place, and the spelling that
    gets stored is the authority's. Matching is exact after normalisation, not
    fuzzy - a fuzzy match here would silently RENAME a district rather than
    merely fail to find it.
    """
    if not value:
        return None
    target = master._norm(value)

    if kind == "state":
        for name in master.states:
            if master._norm(name) == target:
                return name
        return None

    if kind == "district":
        name, _ = _find_district_entry(master, value)
        return name

    canonical_district, record = _find_district_entry(master, district)
    if not record:
        return None

    if kind == "tehsil":
        for name in (record.get("tehsils") or []):
            if master._norm(name) == target:
                return name
        return None

    if kind == "village":
        villages = record.get("villages") or {}
        # Prefer the tehsil the record claims, so two tehsils in one district
        # that reuse a village name cannot cross-match. Fall back to the whole
        # district only when no tehsil is known yet.
        groups = []
        if tehsil:
            for name, listed in villages.items():
                if master._norm(name) == master._norm(tehsil):
                    groups.append(listed)
        if not groups:
            groups = list(villages.values())
        for listed in groups:
            for name in listed or []:
                if master._norm(name) == target:
                    return name
        return None

    return None


def _normalise_place_scripts(fields) -> List[dict]:
    """
    Rewrite Indic-script administrative names to the master's own spelling.

    Order matters and is a dependency, not a preference. District is resolved
    first because the tehsil authority needs a district to check against:
    tehsil_in_district("sadar", "लखनऊ") can only fail, while the same call with
    "Lucknow" succeeds. So each field is bridged with an authority that can
    actually adjudicate it, rather than all four against the district list.

    Every name is written back in the MASTER's spelling, not the model's. The
    transliteration only has to be close enough to identify the place; storing
    its raw output would put "uttar pradesh" on a land record where the
    directory says "Uttar Pradesh". Matching is exact after normalisation
    rather than fuzzy, because a fuzzy match here would silently rename a
    district instead of merely failing to find it.

    Two groups of fields are deliberately NOT bridged:

      * owner_name and father_name. Nothing validates a person's name against a
        Latin list, so transliterating one would discard the form the document
        actually carries and gain nothing. सुनीता देवी stays as written, and
        that is the form a verifier compares against the page.
      * the identifier fields, which are digits and have no script to bridge.

    A name the master does not know is left in its own script - the master has
    443 villages and most of India's are not among them, so an unmatched
    village keeps exactly what the OCR read rather than acquiring a plausible
    spelling nothing confirmed.

    Returns one record per change, for the audit trail and the reviewer UI. A
    verifier must always be able to see that the value on screen is not
    character-for-character what the page says, and why.
    """
    if not bhashini.AVAILABLE:
        return []

    by_key = {f.key: f for f in fields}
    pending = [k for k in _SCRIPT_BRIDGED_FIELDS
               if k in by_key and getattr(by_key[k], "value", None)
               and bhashini.is_indic(by_key[k].value)]
    if not pending:
        return []

    master = validator_mod._MASTER
    applied: List[dict] = []
    # Context for the dependent lookups. Each starts as whatever the record says
    # and is replaced the moment that field is bridged, so a tehsil is checked
    # against a Latin district rather than a Devanagari one.
    context = {key: getattr(by_key.get(key), "value", None)
               for key in ("district", "tehsil")}

    for key in _SCRIPT_BRIDGED_FIELDS:
        if key not in pending:
            continue
        field = by_key[key]
        original = field.value

        # The authority returns the master's spelling, and None reads as false,
        # so the same callable both screens candidates and canonicalises the
        # winner. That is what keeps 'uttar pradesh' out of a land record.
        def authority(candidate: str, _k=key) -> Optional[str]:
            return _canonical_place(master, _k, candidate,
                                    district=context.get("district"),
                                    tehsil=context.get("tehsil"))

        try:
            resolved = bhashini.resolve_to_reference([original], authority)
        except Exception as exc:
            # Never let a third-party outage stop an ingest. Names stay in their
            # own script, the hierarchy rule reports them as unknown exactly as
            # it did before this feature existed, and the reason goes to stderr
            # rather than being swallowed.
            print(f"  script bridge unavailable: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return applied

        winner = resolved.get(original)
        canonical = authority(winner) if winner else None
        if not canonical or canonical == original:
            continue

        field.value = canonical
        if key in context:
            context[key] = canonical
        field.notes.append(
            f"Script bridged: '{original}' read as '{winner}' and matched to "
            f"'{canonical}' in the administrative master. The document "
            f"reads '{original}'.")
        applied.append({"field_key": key, "from": original, "to": canonical})

    return applied


def process_document(stored_path: str, original_name: str, user: dict) -> dict:
    """
    The full pipeline for one document:
      extract text -> extract fields -> apply learned model -> validate
      -> persist -> route to the right queue.
    """
    started = time.time()

    sha = Database.file_hash(stored_path)
    duplicate_file = DB.find_by_hash(sha)

    doc_work = os.path.join(WORK_DIR, sha[:16])
    os.makedirs(doc_work, exist_ok=True)

    extraction = ocr_engine.extract(stored_path, work_dir=doc_work)
    fields = extract_fields(extraction.lines)

    learned = learning.apply_model(fields, corroborate=_make_corroborator(fields))

    # Script bridge. The OCR reads a place name in fourteen Indic scripts; the
    # administrative master it will be validated against is Latin-only, so a
    # perfectly-read Devanagari district resolves to nothing at all. This maps
    # such names onto the master's own spelling before anything downstream
    # tries to match them.
    #
    # It runs BEFORE the gazetteer on purpose: the gazetteer's vocabularies are
    # Latin too, so bridging the script first lets its correction work as well.
    # No-ops entirely on Latin input and when Bhashini is unconfigured.
    scripts = _normalise_place_scripts(fields)

    # Post-OCR correction against closed vocabularies (LGD gazetteer, land
    # classes, identifier shapes). Runs AFTER the learned model so a learned
    # alias gets first refusal on a value, and before validation so the rules
    # see the corrected text rather than the raw misreading.
    vocab = gazetteer.apply_to_fields(fields)

    values = {}
    for f in fields:
        d = f.to_dict()
        values[f.key] = {"value": d["value"], "confidence": d["confidence"],
                         "extra": d["extra"]}

    # LAST RESORT for place fields: read them out of the running text.
    #
    # Measured on 20 genuine Bhu-Naksha downloads: village and district are
    # not printed as fields on a plot report at all. Where they exist, they
    # are inside the mutation-order prose -
    #
    #   "...सा0 मौजा अमारी, परगना गड़वारा, तहसील बदलापुर, जिला जौनपुर का..."
    #
    # which a label-anchored extractor cannot see, because there is no label
    # and no colon. gazetteer.places_from_prose anchors on the administrative
    # noun instead and confirms the candidate against the master, so an
    # unrecognised word is discarded rather than guessed at.
    #
    # Only fills fields that are genuinely EMPTY - a value actually printed
    # in its own field is better evidence than one mentioned in a sentence
    # about a past transfer, and must never be overwritten by it.
    _recover_places_from_prose(values, extraction.lines)

    # WHAT KIND of document is this? The answer changes which fields may
    # legitimately be demanded of it.
    #
    # Measured on a real notarised Power of Attorney: the OCR read every
    # printed value correctly and the record was still blocked on four
    # REQUIRED_MISSING errors, because a GPA structurally carries no khasra,
    # khata or area. Scoping the requirement to the document type took it from
    # 4 spurious errors and a trust score of 27.4 to 1 real error and 69.4.
    doctype = doc_type_mod.detect(extraction.lines)

    result = validator_mod.validate(values, existing=DB.existing_signatures(),
                                    doc_type=doctype["type"])
    _add_doc_type_issue(result, doctype)

    if duplicate_file:
        result["issues"].insert(0, {
            "rule": "FILE_ALREADY_UPLOADED", "severity": "warning", "field": None,
            "message": f"An identical file was already uploaded as document "
                       f"#{duplicate_file['id']} ({duplicate_file['filename']}).",
            "suggestion": "Check whether this is a redundant re-scan.",
        })
        result["warning_count"] += 1
        if result["decision"] == "auto_approved":
            result["decision"] = "needs_review"

    _add_authenticity_issues(result, values, extraction.render_path)
    _add_structure_issue(result, extraction.render_path, doc_work)
    _add_handwriting_issue(result, extraction)
    _add_vocabulary_issues(result, vocab)
    # LLM suggestions run only at ingestion, deliberately not in revalidate()
    # below: the source text never changes on a field correction, so asking
    # again would resend the same document content to a third party and
    # incur the same cost for zero new information. See llm_extractor.py.
    _add_llm_suggestions(result, values, extraction.full_text)
    _add_geo_issues(result, values)

    summary = summarise(fields)
    _add_anomaly_issue(result, summary, extraction.quality)

    preview = extraction.render_path
    if preview and os.path.exists(preview):
        dest = os.path.join(doc_work, "preview" + os.path.splitext(preview)[1])
        if os.path.abspath(preview) != os.path.abspath(dest):
            try:
                shutil.copyfile(preview, dest)
                preview = dest
            except Exception:
                pass

    elapsed_ms = int((time.time() - started) * 1000)

    doc_id = DB.insert_document(
        filename=original_name,
        stored_path=stored_path,
        preview_path=preview,
        sha256=sha,
        file_size=os.path.getsize(stored_path),
        mime=mimetypes.guess_type(original_name)[0],
        uploaded_by=user.get("id"),
        ocr_engine=extraction.engine,
        page_count=extraction.page_count,
        mean_ocr_conf=extraction.mean_confidence,
        legibility=(extraction.quality or {}).get("legibility_score"),
        quality_json=json.dumps(extraction.quality, ensure_ascii=False),
        warnings_json=json.dumps(extraction.warnings, ensure_ascii=False),
        full_text=extraction.full_text[:200000],
        status=result["decision"],
        decision=result["decision"],
        trust_score=result["trust_score"],
        error_count=result["error_count"],
        warning_count=result["warning_count"],
        signature=result["signature"],
        issues_json=json.dumps(result["issues"], ensure_ascii=False),
        summary_json=json.dumps(summary, ensure_ascii=False),
        # The position, kept as data rather than only as a sentence in the
        # issue list. null when the record could not be placed at all, which
        # is a different statement from "placed at 0,0".
        geotag_json=(json.dumps(result["geotag"], ensure_ascii=False)
                     if result.get("geotag") else None),
        processing_ms=elapsed_ms,
    )

    for f in fields:
        DB.insert_field(doc_id, f.to_dict())

    DB.audit(user, "document_ingested", doc_id,
             detail=f"engine={extraction.engine}; decision={result['decision']}; "
                    f"errors={result['error_count']}; warnings={result['warning_count']}; "
                    f"{elapsed_ms}ms")
    for adj in learned:
        DB.audit(user, "learned_adjustment", doc_id, adj.get("field_key"),
                 adj.get("from"), adj.get("to"), detail=adj.get("type"))
    # A script bridge changes a value the reviewer will compare against the
    # page, so it belongs in the audit trail for the same reason a learned
    # correction does: the record must say why the screen and the paper differ.
    for adj in scripts:
        DB.audit(user, "script_bridged", doc_id, adj.get("field_key"),
                 adj.get("from"), adj.get("to"), detail="transliteration")

    return {
        "document_id": doc_id,
        "filename": original_name,
        "engine": extraction.engine,
        "decision": result["decision"],
        "trust_score": result["trust_score"],
        "error_count": result["error_count"],
        "warning_count": result["warning_count"],
        "summary": summary,
        "warnings": extraction.warnings,
        "learned_adjustments": learned,
        "processing_ms": elapsed_ms,
    }


def revalidate(document_id: int, user: dict) -> dict:
    """
    Re-run the rule engine after human corrections.

    Also re-runs the authenticity (seal/signature/stamp-paper), table/layout
    structure, and anomaly checks, using the document's own persisted preview
    image/quality/summary - not just validator_mod.validate(). Without this,
    correcting any one field and triggering a revalidation would silently
    drop SEAL_MISSING, SIGNATURE_MISSING, ANOMALY_DETECTED and the structure
    issues from the stored issue list (set_document_validation overwrites
    issues_json wholesale), which would accidentally clear a blocking
    seal-missing error without the seal ever actually being re-examined.
    The table/structure re-check is not itself blocking, but the same "an
    issue must not silently vanish on save" discipline applies to it; the
    repeated PP-StructureV3 inference cost on correction-save is an accepted
    tradeoff of that discipline, not an oversight.

    Deliberately NOT re-run here: llm_extractor.py's suggestions
    (_add_llm_suggestions). Those exist to fill in fields the rule-based
    extractor could not, but a field a human has just corrected no longer
    needs a suggestion for itself, and the document's source text has not
    changed - re-running it on every correction would resend the same
    document content to a third-party API and pay for it again for zero new
    information. Any still-missing LLM_SUGGESTION issues from ingestion time
    do not survive a revalidation call; that is an accepted asymmetry with
    the authenticity/structure/anomaly checks above, not a bug matching
    theirs.
    """
    values = DB.field_values_map(document_id)
    if not values:
        raise ApiError(404, f"Document {document_id} not found.")
    doc = DB.get_document(document_id) or {}
    result = validator_mod.validate(
        values, existing=DB.existing_signatures(exclude_id=document_id))
    _add_authenticity_issues(result, values, doc.get("preview_path"))
    doc_work = os.path.join(WORK_DIR, (doc.get("sha256") or "")[:16])
    _add_structure_issue(result, doc.get("preview_path"), doc_work)
    _add_geo_issues(result, values)
    _add_anomaly_issue(result, doc.get("summary") or {}, doc.get("quality") or {})
    DB.set_document_validation(document_id, result)
    DB.audit(user, "document_revalidated", document_id,
             detail=f"decision={result['decision']}; errors={result['error_count']}")
    return result


# --------------------------------------------------------------------------
# Cadastral map (S12a vectorization/georeferencing, S12d GIS map view)
# --------------------------------------------------------------------------

CADASTRAL_DIR = os.path.join(SAMPLES_DIR, "cadastral")
CADASTRAL_MAP = os.path.join(CADASTRAL_DIR, "village_map_narharpur.png")
CADASTRAL_CONTROL_POINTS = os.path.join(CADASTRAL_DIR, "control_points.json")

# Real village maps live here, one directory per map, alongside the bundled
# demo. Kept under storage/ rather than samples/ because they are operational
# data an office adds, not fixtures that ship with the project.
CADASTRAL_STORE = os.path.join(STORAGE_DIR, "cadastral")
os.makedirs(CADASTRAL_STORE, exist_ok=True)

MAP_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_CADASTRAL_CACHE: Dict[str, dict] = {}


def _read_map_metadata(control_points_path: str) -> dict:
    try:
        with open(control_points_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _village_aliases(payload: dict) -> set:
    aliases = list(payload.get("village_aliases") or [])
    if payload.get("village"):
        aliases.append(payload["village"])
    return {a.strip().casefold() for a in aliases if a and a.strip()}


def list_cadastral_maps() -> List[dict]:
    """
    Every cadastral map available to this installation: the bundled demo plus
    any real village map added under storage/cadastral/.

    A map is just a directory holding an image and a control_points.json, so
    adding one needs no migration and no database row - which also means a
    map can be dropped in by hand on a machine with no working upload path.
    """
    maps: List[dict] = []

    if os.path.exists(CADASTRAL_MAP) and os.path.exists(CADASTRAL_CONTROL_POINTS):
        meta = _read_map_metadata(CADASTRAL_CONTROL_POINTS)
        maps.append({
            "id": "demo",
            "village": meta.get("village") or "Narharpur",
            "district": meta.get("district"),
            "aliases": _village_aliases(meta),
            "map_path": CADASTRAL_MAP,
            "shapefile_path": None,
            "control_points_path": CADASTRAL_CONTROL_POINTS,
            "bundled": True,
        })

    for entry in sorted(os.listdir(CADASTRAL_STORE)) if os.path.isdir(CADASTRAL_STORE) else []:
        folder = os.path.join(CADASTRAL_STORE, entry)
        if not os.path.isdir(folder):
            continue
        cp = os.path.join(folder, "control_points.json")
        images = [f for f in sorted(os.listdir(folder))
                  if os.path.splitext(f)[1].lower() in MAP_IMAGE_EXT]
        # A digitized parcel layer needs no raster and no georeferencing, so a
        # folder holding only a .shp set is a complete map on its own. The
        # metadata JSON is still required, because a shapefile carries
        # geometry but not the village name records are matched on.
        shapefile = shapefile_import.find_shapefile(folder)
        if not os.path.exists(cp) or not (images or shapefile):
            continue
        meta = _read_map_metadata(cp)
        maps.append({
            "id": entry,
            "village": meta.get("village") or entry,
            "district": meta.get("district"),
            "aliases": _village_aliases(meta),
            "map_path": os.path.join(folder, images[0]) if images else None,
            "shapefile_path": shapefile,
            "control_points_path": cp,
            "bundled": False,
        })
    return maps


def _find_map(map_id: Optional[str]) -> Optional[dict]:
    maps = list_cadastral_maps()
    if not maps:
        return None
    if map_id:
        for m in maps:
            if m["id"] == map_id:
                return m
        return None
    return maps[0]


def _cadastral_geojson(map_id: Optional[str] = None) -> dict:
    """
    Vectorize + georeference one cadastral map, cached per map id - the result
    never changes for a fixed input, so redoing contour detection on every
    map-tab load would be pure waste. Returns an honest empty-with-reason
    payload rather than raising when OpenCV is unavailable or the files are
    missing, matching every other optional-capability path in this module.
    """
    target = _find_map(map_id)
    key = target["id"] if target else f"__missing__{map_id}"
    if key in _CADASTRAL_CACHE:
        return _CADASTRAL_CACHE[key]

    if not cadastral.CV_AVAILABLE:
        result = {"type": "FeatureCollection", "features": [],
                  "_error": "OpenCV/numpy not installed; cadastral vectorization was skipped."}
        _CADASTRAL_CACHE[key] = result
        return result
    if target is None:
        result = {"type": "FeatureCollection", "features": [],
                  "_error": (f"No cadastral map named '{map_id}'." if map_id else
                             "No cadastral map has been loaded. A georeferenced "
                             "village sheet must be added before parcels can be "
                             "shown on the map.")}
        _CADASTRAL_CACHE[key] = result
        return result

    try:
        disclaimer = cadastral.DEMO_DISCLAIMER if target["bundled"] else None

        # A digitized layer beats anything recovered from pixels. The operator
        # decided where each boundary runs; the vectoriser only ever infers it
        # from a threshold, and on a folded, mudded village sheet that
        # inference is worth metres on the ground. So if a .shp is present it
        # wins outright - no vectorising, no affine fit, nothing to go wrong
        # in either step.
        if target.get("shapefile_path"):
            geojson = shapefile_import.to_geojson(target["shapefile_path"],
                                                  disclaimer=disclaimer)
            geojson["_control_points"] = []
            geojson["_map"] = {"id": target["id"], "village": target["village"],
                               "district": target["district"],
                               "bundled": target["bundled"]}
            _CADASTRAL_CACHE[key] = geojson
            return geojson

        # A georeferencing produced by ArcGIS or QGIS wins over control points
        # picked here. It was established against a real basemap, usually with
        # real ground control, by someone whose job that is - re-fitting an
        # affine from a handful of GCPs on top of it could only add error.
        # discover() returns None when there is no sidecar to read, and RAISES
        # when one exists but cannot be used, so a rejected georeferencing
        # surfaces as an error instead of quietly reverting to the GCPs.
        imported = georeference.discover(
            target["map_path"], *_image_size(target["map_path"]))

        if imported is not None:
            geojson = cadastral.vectorize_and_georeference(
                target["map_path"],
                transform=cadastral.AffineTransform(**imported.as_transform_kwargs()),
                georeference_meta=imported.to_dict(),
                disclaimer=disclaimer)
            geojson["_control_points"] = []
        else:
            control_points = cadastral.load_control_points(target["control_points_path"])
            geojson = cadastral.vectorize_and_georeference(
                target["map_path"], control_points, disclaimer=disclaimer)
            # The GCPs the affine fit was derived from, exposed so the map can
            # show the georeferencing's own basis as a layer rather than asking
            # the viewer to take the transform on trust. Pixel coords are
            # dropped; only the real-world anchors are meaningful on a map.
            geojson["_control_points"] = [
                {"lon": lon, "lat": lat} for _px, _py, lon, lat in control_points
            ]
        geojson["_map"] = {"id": target["id"], "village": target["village"],
                           "district": target["district"], "bundled": target["bundled"]}
    except Exception as exc:
        geojson = {"type": "FeatureCollection", "features": [],
                  "_error": f"Cadastral vectorization failed: {exc}"}
    _CADASTRAL_CACHE[key] = geojson
    return geojson


# A record and its parcel are allowed to disagree on area by this much before
# it is worth a human's attention. Vectorising a scanned map and fitting an
# affine transform both carry real error, so a few percent means nothing; 15%
# of a one-hectare plot is roughly 1,500 m², which is a materially different
# piece of land and not something a survey and a register should differ over
# silently. Deliberately reported as a DISAGREEMENT, never as "the record is
# wrong": the geometry is at least as likely to be the inaccurate side.
GEO_AREA_TOLERANCE = 0.15


def _image_size(path: str) -> Tuple[int, int]:
    """
    (width, height) of a raster, or (0, 0) if it cannot be read.

    Only used to sanity-check an imported georeferencing at the sheet's
    corners. Zeros collapse that check to the origin rather than failing the
    load outright, because an unreadable image is the vectoriser's problem to
    report, not this helper's.
    """
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (0, 0)
        return (int(img.shape[1]), int(img.shape[0]))
    except Exception:
        return (0, 0)


def _identifier_authority(village: str) -> Dict[str, set]:
    """
    Every plot identifier we can actually vouch for in one village, gathered
    from the external registry and from any loaded cadastral map.

    Scoped by village on purpose. Khasra numbers repeat across villages, so a
    nationwide "does this number exist anywhere" check would corroborate
    almost anything and make the learned OCR repair below dangerous rather
    than useful.
    """
    known: Dict[str, set] = {"khasra_number": set(), "khata_number": set(),
                             "survey_number": set()}
    if not village:
        return known
    target = village.casefold().strip()

    for record in getattr(fact_checker._REGISTRY, "records", []):
        if (record.get("village") or "").casefold().strip() != target:
            continue
        for key in known:
            raw = record.get(key)
            if raw:
                known[key].add(fact_checker._normalise_identifier(raw))

    for entry in list_cadastral_maps():
        aliases = entry.get("aliases") or set()
        if aliases and target not in aliases:
            continue
        for feature in _cadastral_geojson(entry["id"]).get("features", []):
            raw = (feature.get("properties") or {}).get("khasra_number")
            if raw:
                known["khasra_number"].add(fact_checker._normalise_identifier(raw))
    return known


def _make_corroborator(fields: List) -> Any:
    """
    Build the authority callable that learning.apply_model uses to decide
    whether a candidate plot identifier is real.

    The village comes from the same extraction being corrected, so it can
    itself be wrong; when it resolves to nothing we hold no authority for that
    parcel and every candidate is refused, which disables repair rather than
    guessing. Refusing to act is the correct behaviour here - a wrong plot
    number written confidently is the exact failure this system exists to
    prevent.
    """
    village = ""
    for f in fields:
        if f.key == "village" and getattr(f, "value", None):
            village = str(f.value).strip()
            break
    known = _identifier_authority(village)

    def corroborate(field_key: str, value: str) -> bool:
        pool = known.get(field_key)
        if not pool:
            return False
        return fact_checker._normalise_identifier(value) in pool

    return corroborate


def _topology_report(geojson: dict) -> dict:
    """
    Run the topology rules over one village's vectorised parcels.

    Vectorising produces polygons that look right and can still be
    topologically wrong, and the wrong ones are the ones that matter: a gap
    between two parcels is land belonging to nobody on the map, an overlap is
    land given to two people at once. Neither is visible by eye on a sheet
    carrying hundreds of parcels, and both are how disputes start.

    Rings come back from GeoJSON in degrees, so the half-metre snap tolerance
    has to be converted - read as degrees it would be a 55 km tolerance and
    would collapse an entire district onto one point.
    """
    features = (geojson or {}).get("features") or []
    rings: Dict[str, list] = {}
    for idx, feat in enumerate(features):
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        coords = (geom.get("coordinates") or [None])[0]
        if not coords or len(coords) < 3:
            continue
        props = feat.get("properties") or {}
        key = str(props.get("khasra_number") or props.get("parcel_id") or idx)
        rings[key] = [tuple(c[:2]) for c in coords]

    if not rings:
        return {"parcel_count": 0, "clean": True, "checked": False}

    in_degrees = any(abs(pt[0]) <= 180 and abs(pt[1]) <= 90
                     for ring in rings.values() for pt in ring[:1])
    tol = topology.suggested_tolerance(in_degrees=in_degrees)
    report = topology.validate(rings, tolerance=tol)
    report["checked"] = True
    report["units"] = "degrees" if in_degrees else "pixels"
    return report


def _parcel_for_record(values: Dict[str, dict]) -> Optional[dict]:
    """
    Find the cadastral parcel a record refers to, across every loaded map,
    scoped by village.

    Khasra numbers are only unique WITHIN a village - two villages routinely
    reuse the same numbers - so matching on the number alone would happily
    tag a Bhopal record onto a Lucknow parcel. The record's village must
    match the village a map declares (in either script) before its parcels
    are even searched, which is also what makes holding several villages'
    maps at once safe. A record whose village is blank is not matched at all
    rather than guessed at.
    """
    khasra = (values.get("khasra_number", {}).get("value") or "").strip()
    village = (values.get("village", {}).get("value") or "").strip()
    if not khasra or not village:
        return None

    for entry in list_cadastral_maps():
        aliases = entry.get("aliases") or set()
        if aliases and village.casefold() not in aliases:
            continue
        geojson = _cadastral_geojson(entry["id"])
        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            if (props.get("khasra_number") or "").strip() == khasra:
                return {"props": props,
                        "map": entry,
                        "is_demo": bool(geojson.get("_disclaimer"))}
    return None


def _add_place_geotag(result: dict, values: Dict[str, dict]) -> None:
    """
    Position a record by the place names it carries, when it has no geometry.

    This is the only geotag available for the majority of documents people
    actually upload. A deed, an attorney grant or a mutation order names a
    village and a district but contains no coordinate, no world file and no
    control points - and absolute position cannot be recovered from an
    unreferenced scan, because every measurement inside such an image is
    invariant under translation. Measured on a real Delhi general power of
    attorney, georeference.discover() returned None and the only locational
    evidence on the sheet was the string "VILLAGE NARELA, SABOLI ROAD,
    DELHI".

    So the coordinate here is an administrative one, and the code is careful
    to say so rather than let it pass for a survey result:

      * its own rule name, GEO_PLACE_APPROXIMATE, never GEO_TAGGED
      * accuracy_m is always reported, and it is kilometres
      * severity "info" only - it never blocks and never downgrades a
        decision, because a coordinate inferred from a name carries no new
        information about the parcel and therefore cannot contradict
        anything the record says
      * no area cross-check, unlike the parcel path. A district centroid
        cannot confirm or deny 57 square yards, and pretending otherwise
        would manufacture agreement out of nothing.

    A name that disagrees with the record's own stated state IS surfaced,
    because that is a genuine internal inconsistency worth a reviewer's
    attention even though it is not a geometric finding.
    """
    match = geocode.resolve(values)
    if match is None:
        return

    km = match.accuracy_m / 1000.0
    scope = {"locality": "town or village", "district": "district",
             "state": "state"}.get(match.level, match.level)
    how = "named in the record" if match.exact else "inferred from a damaged or transliterated spelling"

    result["geotag"] = {
        "lat": match.lat, "lon": match.lon,
        "source": "place_name",
        "precision": match.level,
        "accuracy_m": match.accuracy_m,
        "place": match.name,
        "matched_from": match.matched_from,
        "matched_text": match.matched_text,
        "exact": match.exact,
        "corroborated": match.corroborated,
        "state": match.state,
        "district": match.district,
    }

    result["issues"].append({
        "rule": "GEO_PLACE_APPROXIMATE", "severity": "info", "field": None,
        "message": (
            f"Approximate location {match.lat:.4f}, {match.lon:.4f} from the "
            f"{scope} \"{match.name}\" ({how}). This locates the {scope} to "
            f"within roughly {km:,.0f} km, NOT the parcel. No cadastral "
            f"parcel, world file or control points were available for this "
            f"document, and absolute coordinates cannot be derived from an "
            f"unreferenced scan."),
        "suggestion": ("Import a georeferenced cadastral map, a world file, "
                       "or at least 3 ground control points to obtain "
                       "parcel-level coordinates."),
    })

    for conflict in match.conflicts:
        result["issues"].append({
            "rule": "GEO_PLACE_CONFLICT", "severity": "warning", "field": "village",
            "message": (f"Place names in this record disagree: {conflict}. "
                        f"The stated state was used for the approximate "
                        f"location."),
            "suggestion": "Check the village, district and state against each other.",
        })
        result["warning_count"] = result.get("warning_count", 0) + 1
        if result.get("decision") == "auto_approved":
            result["decision"] = "needs_review"


def _add_geo_issues(result: dict, values: Dict[str, dict]) -> None:
    """
    Geo-tag the record and cross-check it against its own parcel geometry.

    This is what turns the cadastral map from a display into a validation
    step: a matched record gains real coordinates, and its stated area is
    compared against the area its parcel actually measures. Informational
    when they agree, a warning when they do not - never blocking, because a
    disagreement can as easily mean the map was digitised imprecisely as
    that the register is wrong.
    """
    parcel = _parcel_for_record(values)
    if not parcel:
        # No parcel geometry for this record. Fall back to positioning it by
        # the place names it carries, which is all a deed or attorney grant
        # usually offers. Kept strictly separate from the parcel-grade path
        # below - see _add_place_geotag for why that separation matters.
        _add_place_geotag(result, values)
        return

    props = parcel["props"]
    lat, lon = props.get("centroid_lat"), props.get("centroid_lon")
    caveat = (" Coordinates come from the bundled demo map's illustrative "
              "control points, not a real survey." if parcel["is_demo"] else "")

    if lat is not None and lon is not None:
        result["geotag"] = {
            "lat": lat, "lon": lon,
            "source": "cadastral_parcel",
            "precision": "parcel",
            "accuracy_m": None,
            "place": props.get("khasra_number"),
            "parcel_id": props.get("parcel_id"),
            "is_demo": bool(parcel["is_demo"]),
        }
        result["issues"].append({
            "rule": "GEO_TAGGED", "severity": "info", "field": None,
            "message": f"Matched to cadastral parcel {props.get('parcel_id')} "
                      f"(khasra {props.get('khasra_number')}) at "
                      f"{lat:.6f}, {lon:.6f}." + caveat,
            "suggestion": None,
        })

    stated = values.get("area", {}).get("value")
    parsed = field_extractor.parse_area(stated) if stated else None
    stated_sqm = (parsed or {}).get("sqm")
    parcel_sqm = props.get("area_m2")
    if not stated_sqm or not parcel_sqm:
        return

    diff = abs(stated_sqm - parcel_sqm) / parcel_sqm
    if diff > GEO_AREA_TOLERANCE:
        result["issues"].append({
            "rule": "GEO_AREA_MISMATCH", "severity": "warning", "field": "area",
            "message": (
                f"Recorded area ({stated_sqm:,.0f} m²) differs from the mapped "
                f"parcel's measured area ({parcel_sqm:,.0f} m²) by "
                f"{diff * 100:.0f}%. This is a disagreement between the register "
                f"and the map, not proof either one is wrong." + caveat),
            "suggestion": "Check the record against the cadastral map before approving.",
        })
        result["warning_count"] = result.get("warning_count", 0) + 1
        if result["decision"] == "auto_approved":
            result["decision"] = "needs_review"
    else:
        result["issues"].append({
            "rule": "GEO_AREA_VERIFIED", "severity": "info", "field": "area",
            "message": (f"Recorded area agrees with the mapped parcel "
                        f"({stated_sqm:,.0f} m² vs {parcel_sqm:,.0f} m², "
                        f"{diff * 100:.0f}% apart)." + caveat),
            "suggestion": None,
        })


def _link_cadastral_documents(geojson: dict) -> dict:
    """
    Live join between uploaded land-record documents and the (cached, fixed)
    parcel geometry - this is the actual "geo-tag a document on the map"
    behaviour: once a document is extracted, validated and routed, if its
    khasra_number matches a parcel already present in the vectorized
    cadastral map, that parcel is annotated with the document's id, status
    and trust score. It is a MATCH against already-known geometry, not new
    geometry synthesised from the text record - a khatauni/khasra document
    carries no coordinates of its own, so there is nothing to plot for a
    khasra number that has no corresponding parcel in a vectorized map yet.

    Recomputed on every request, unlike the geometry itself: approvals and
    corrections happen continuously and this must never show a stale status.
    The base geojson (module-level cache) is deep-copied before mutation so
    concurrent requests never see a half-updated shared object.

    Matched on khasra_number alone, not also village/tehsil: this project's
    cadastral pipeline vectorizes one village's map at a time, so there is
    no second village's khasra numbers to collide with in the bundled demo.
    A real multi-village deployment would need to scope this match by
    village too - noted in the README rather than silently assumed safe.
    """
    if not geojson.get("features"):
        return geojson

    index: Dict[str, dict] = {}
    for doc in DB.list_documents(limit=5000):
        if doc.get("status") == "rejected":
            continue
        k = (doc.get("khasra_number") or "").strip()
        if k and k not in index:   # list_documents is newest-first; keep the newest match
            index[k] = doc

    out = copy.deepcopy(geojson)
    for feature in out["features"]:
        khasra = (feature["properties"].get("khasra_number") or "").strip()
        doc = index.get(khasra)
        feature["properties"]["linked_document"] = ({
            "document_id": doc["id"], "status": doc["status"],
            "trust_score": doc.get("trust_score"), "owner_name": doc.get("owner_name"),
            "filename": doc.get("filename"),
            # Carried so the map's thematic layers (S12a) can colour a parcel
            # by what the record actually says, not just by its workflow state.
            "land_classification": doc.get("land_classification"),
            "village": doc.get("village"),
            "area": doc.get("area"),
        } if doc else None)
    return out


# --------------------------------------------------------------------------
# Request handler
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "LRDVS/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str,
              extra_headers: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise ApiError(413, "Upload exceeds the 40 MB limit.")
        return self.rfile.read(length) if length else b""

    def _json_body(self) -> dict:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise ApiError(400, "Request body is not valid JSON.")

    # -- auth ---------------------------------------------------------
    def _current_user(self) -> dict:
        """
        Identify the caller, from a verified token where one is available.

        Order matters. A Bearer token is always preferred and, once a secret
        is configured, always verified - so a caller cannot downgrade
        themselves to the header path by simply omitting the token. The
        header path survives only where nothing can be verified anyway
        (see auth.dev_mode), which keeps an offline demo working without
        leaving a network-facing server open.
        """
        token = ""
        header = self.headers.get("Authorization") or ""
        if header.lower().startswith("bearer "):
            token = header[7:].strip()

        secret = auth.jwt_secret()

        if token and secret:
            try:
                claims = auth.verify_token(token, secret)
            except auth.AuthError as exc:
                # The reason is safe to return here: the caller supplied the
                # token, so it tells them nothing they did not already have.
                raise ApiError(401, f"Sign-in required: {exc}")
            username = auth.identity_from_claims(claims)
            row = DB.ensure_user(username,
                                 claims.get("email") or username,
                                 auth.role_from_claims(claims))
            return dict(row)

        # A token presented to a server with no secret is IGNORED, not
        # refused.
        #
        # Refusing was the first implementation and it was wrong in a way
        # that only showed up in a browser: a returning operator still holds
        # a Supabase session, so the client attaches a Bearer token, and the
        # server rejected it although it would have accepted the very same
        # request with no token at all. Presenting credentials must never
        # leave a caller worse off than presenting none.
        #
        # Ignoring is not the same as trusting. The token contributes nothing
        # to the identity below - it is discarded, and the caller is
        # identified exactly as an anonymous one would be. When a secret IS
        # configured the branch above runs first and the token is verified
        # properly, so this path cannot be used to bypass anything.

        if not auth.dev_mode():
            raise ApiError(401, "Sign-in required.")

        username = self.headers.get("X-User") or "operator1"
        row = DB.get_user(username)
        if not row:
            raise ApiError(401, f"Unknown user '{username}'.")
        return dict(row)

    def _require(self, right: str) -> dict:
        user = self._current_user()
        if right not in ROLE_RIGHTS.get(user["role"], set()):
            raise ApiError(403, f"Role '{user['role']}' is not permitted to {right}. "
                                f"This action is restricted by role-based access control.")
        return user

    # -- routing ------------------------------------------------------
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if not path.startswith("/api"):
                return self._serve_static(path)
            handler = self._route(method, path)
            if handler is None:
                raise ApiError(404, f"No API route for {method} {path}.")
            handler(query)
        except ApiError as exc:
            self._json({"error": exc.message, "status": exc.status}, exc.status)
        except BrokenPipeError:
            pass
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"Internal error: {exc}", "status": 500}, 500)

    def _route(self, method: str, path: str):
        doc_match = re.match(r"^/api/documents/(\d+)$", path)
        sub_match = re.match(r"^/api/documents/(\d+)/([a-z]+)$", path)

        if method == "GET":
            if path == "/api/auth/status":
                return self.api_auth_status
            if path == "/api/session":
                return self.api_session
            if path == "/api/documents":
                return self.api_list_documents
            if path == "/api/stats":
                return self.api_stats
            if path == "/api/audit":
                return self.api_audit
            if path == "/api/audit/verify":
                return self.api_audit_verify
            if path == "/api/learning":
                return self.api_learning
            if path == "/api/schema":
                return self.api_schema
            if path == "/api/export/csv":
                return self.api_export_csv
            if path == "/api/export/json":
                return self.api_export_json
            if path == "/api/cadastral/parcels":
                return self.api_cadastral_parcels
            if path == "/api/cadastral/maps":
                return self.api_cadastral_maps
            if doc_match:
                return lambda q: self.api_get_document(int(doc_match.group(1)), q)
            if sub_match and sub_match.group(2) == "preview":
                return lambda q: self.api_preview(int(sub_match.group(1)), q)
            if sub_match and sub_match.group(2) == "text":
                return lambda q: self.api_text(int(sub_match.group(1)), q)

        if method == "POST":
            if path == "/api/upload":
                return self.api_upload
            if path == "/api/seed":
                return self.api_seed
            if path == "/api/learning/retrain":
                return self.api_retrain
            if path == "/api/reset":
                return self.api_reset
            if path == "/api/cadastral/maps":
                return self.api_cadastral_upload
            if sub_match:
                doc_id, action = int(sub_match.group(1)), sub_match.group(2)
                if action == "fields":
                    return lambda q: self.api_update_field(doc_id, q)
                if action == "approve":
                    return lambda q: self.api_approve(doc_id, q)
                if action == "reject":
                    return lambda q: self.api_reject(doc_id, q)
                if action == "revalidate":
                    return lambda q: self.api_revalidate(doc_id, q)
        return None

    # -- static -------------------------------------------------------
    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = os.path.abspath(os.path.join(FRONTEND_DIR, rel))
        if not target.startswith(os.path.abspath(FRONTEND_DIR)):
            raise ApiError(403, "Path traversal blocked.")
        if not os.path.isfile(target):
            raise ApiError(404, f"Not found: {rel}")
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(target, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # -- API: reads ---------------------------------------------------
    def api_auth_status(self, query: dict) -> None:
        """
        Whether this server requires a signed-in user. Deliberately public.

        The client has to be able to ask this BEFORE it can authenticate,
        which is why it is the one route that does not call _current_user.
        It discloses nothing an unauthenticated caller could not already
        learn by making a request and reading the status code.

        It exists because the frontend redirected to the sign-in page
        unconditionally while the backend, with no JWT secret configured,
        accepted the request anyway. On a laptop with nothing set up that
        left the app unreachable behind a login it did not need - which
        breaks the promise that `python3 run.py` just works offline.
        """
        self._json(auth.auth_status())

    def api_session(self, query: dict) -> None:
        user = self._current_user()
        self._json({
            "user": user,
            "rights": sorted(ROLE_RIGHTS.get(user["role"], set())),
            "users": DB.list_users(),
            "capabilities": ocr_engine.capabilities(),
            "admin_master_loaded": validator_mod._MASTER.loaded,
            "fact_check_available": fact_checker.SKLEARN_AVAILABLE,
            "fact_check_registry_loaded": fact_checker._REGISTRY.loaded,
            "samples_available": sorted(
                f for f in os.listdir(SAMPLES_DIR)
                if os.path.splitext(f)[1].lower() in ALLOWED_EXT
            ) if os.path.isdir(SAMPLES_DIR) else [],
        })

    def api_schema(self, query: dict) -> None:
        self._json({"fields": [
            {"key": s.key, "display": s.display, "kind": s.kind,
             "required": s.required, "labels": s.labels}
            for s in FIELD_SPECS
        ]})

    def api_list_documents(self, query: dict) -> None:
        status = (query.get("status") or ["all"])[0]
        search = (query.get("search") or [None])[0]
        limit = min(500, int((query.get("limit") or [100])[0]))
        offset = int((query.get("offset") or [0])[0])
        self._json({"documents": DB.list_documents(status, search, limit, offset)})

    def api_get_document(self, doc_id: int, query: dict) -> None:
        doc = DB.get_document(doc_id)
        if not doc:
            raise ApiError(404, f"Document {doc_id} not found.")
        doc.pop("full_text", None)
        doc["audit"] = DB.audit_trail(doc_id, limit=60)
        self._json(doc)

    def api_text(self, doc_id: int, query: dict) -> None:
        row = DB.one("SELECT full_text FROM documents WHERE id = ?", (doc_id,))
        if not row:
            raise ApiError(404, f"Document {doc_id} not found.")
        self._json({"document_id": doc_id, "full_text": row["full_text"] or ""})

    def api_preview(self, doc_id: int, query: dict) -> None:
        row = DB.one("SELECT preview_path, stored_path FROM documents WHERE id = ?",
                     (doc_id,))
        if not row:
            raise ApiError(404, f"Document {doc_id} not found.")
        path = row["preview_path"] or row["stored_path"]
        if not path or not os.path.isfile(path):
            raise ApiError(404, "No preview available for this document.")
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def api_stats(self, query: dict) -> None:
        self._json(DB.stats())

    def api_audit(self, query: dict) -> None:
        doc_id = query.get("document_id")
        limit = min(500, int((query.get("limit") or [150])[0]))
        self._json({"entries": DB.audit_trail(
            int(doc_id[0]) if doc_id else None, limit)})

    def api_audit_verify(self, query: dict) -> None:
        """
        Re-verify the audit tamper-evidence chain.

        Deliberately open to every signed-in role, auditor included. An
        integrity check that only an administrator may run is worth very
        little, because the administrator is exactly who an auditor is
        checking up on.
        """
        self._json(DB.verify_audit_chain())

    def api_learning(self, query: dict) -> None:
        model = learning.load_model()
        self._json({
            "model": model,
            "recent_corrections": DB.learning_signals(limit=40),
            "thresholds": {
                "min_alias_support": learning.MIN_ALIAS_SUPPORT,
                "min_confusion_support": learning.MIN_CONFUSION_SUPPORT,
                "min_calibration_sample": learning.MIN_CALIBRATION_SAMPLE,
            },
        })

    # -- API: writes --------------------------------------------------
    def api_upload(self, query: dict) -> None:
        user = self._require("upload")
        ctype = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in ctype.lower():
            raise ApiError(400, "Upload must be multipart/form-data.")

        parts = parse_multipart(self._read_body(), ctype)
        files = [p for p in parts if p.get("filename")]
        if not files:
            raise ApiError(400, "No file was included in the upload.")

        results, errors = [], []
        for part in files:
            name = safe_filename(part["filename"])
            ext = os.path.splitext(name)[1].lower()
            if ext not in ALLOWED_EXT:
                errors.append({"filename": name,
                               "error": f"Unsupported file type '{ext}'."})
                continue
            if not part["data"]:
                errors.append({"filename": name, "error": "File was empty."})
                continue

            stored = _stored_upload_path(name)
            with open(stored, "wb") as fh:
                fh.write(part["data"])
            try:
                results.append(process_document(stored, name, user))
            except Exception as exc:
                traceback.print_exc()
                errors.append({"filename": name, "error": str(exc)})

        self._json({"processed": results, "errors": errors,
                    "count": len(results)}, 200 if results else 400)

    def api_seed(self, query: dict) -> None:
        """Ingest the bundled sample records - the one-click demo path."""
        user = self._require("upload")
        if not os.path.isdir(SAMPLES_DIR):
            raise ApiError(404, "No samples directory found.")
        names = sorted(f for f in os.listdir(SAMPLES_DIR)
                       if os.path.splitext(f)[1].lower() in ALLOWED_EXT)
        if not names:
            raise ApiError(404, "No sample documents found. Run tools/make_samples.py.")

        results, errors = [], []
        for name in names:
            src = os.path.join(SAMPLES_DIR, name)
            stored = _stored_upload_path(name)
            try:
                # copyfile is inside the try too: one file failing to copy
                # (e.g. a Windows MAX_PATH violation under a deeply nested
                # install directory) must not abort the whole batch any more
                # than one failing to process does.
                shutil.copyfile(src, stored)
                results.append(process_document(stored, name, user))
            except Exception as exc:
                traceback.print_exc()
                errors.append({"filename": name, "error": str(exc)})
        self._json({"processed": results, "errors": errors, "count": len(results)})

    def api_update_field(self, doc_id: int, query: dict) -> None:
        user = self._require("correct")
        body = self._json_body()
        field_key = body.get("field_key")
        if field_key not in FIELD_BY_KEY:
            raise ApiError(400, f"Unknown field '{field_key}'.")

        confirm = bool(body.get("confirm"))
        new_value = body.get("value")
        if new_value is not None:
            new_value = str(new_value).strip() or None

        try:
            change = DB.update_field(doc_id, field_key, new_value, user, confirm_only=confirm)
        except KeyError as exc:
            raise ApiError(404, str(exc))

        result = revalidate(doc_id, user)
        self._json({"change": change, "validation": result,
                    "fields": DB.get_fields(doc_id)})

    def api_revalidate(self, doc_id: int, query: dict) -> None:
        user = self._require("revalidate")
        self._json({"validation": revalidate(doc_id, user)})

    def api_approve(self, doc_id: int, query: dict) -> None:
        user = self._require("approve")
        doc = DB.get_document(doc_id)
        if not doc:
            raise ApiError(404, f"Document {doc_id} not found.")

        values = DB.field_values_map(doc_id)
        result = validator_mod.validate(
            values, existing=DB.existing_signatures(exclude_id=doc_id))
        # Without these, this gate would check only validator.py's rules -
        # SEAL_MISSING and ANOMALY_DETECTED live in server.py, not there, so
        # a seal-less document would sail through this check with
        # error_count=0 even though it is stored as "blocked". The mandatory
        # seal requirement is only real if this final gate sees it too.
        _add_authenticity_issues(result, values, doc.get("preview_path"))
        _add_geo_issues(result, values)
        _add_anomaly_issue(result, doc.get("summary") or {}, doc.get("quality") or {})
        if result["error_count"] > 0:
            DB.set_document_validation(doc_id, result)
            raise ApiError(409, f"Cannot approve: {result['error_count']} blocking "
                                f"error(s) remain. Resolve them first.")

        DB.approve_document(doc_id, user)
        model = learning.retrain(DB)
        self._json({"status": "approved", "document_id": doc_id,
                    "learning": {"samples": model.get("samples"),
                                 "active_rules": model.get("active_rules")}})

    def api_reject(self, doc_id: int, query: dict) -> None:
        user = self._require("reject")
        body = self._json_body()
        reason = (body.get("reason") or "").strip()
        if len(reason) < 4:
            raise ApiError(400, "A rejection reason is required for the audit trail.")
        if not DB.get_document(doc_id):
            raise ApiError(404, f"Document {doc_id} not found.")
        DB.reject_document(doc_id, user, reason)
        self._json({"status": "rejected", "document_id": doc_id})

    def api_retrain(self, query: dict) -> None:
        user = self._require("retrain")
        model = learning.retrain(DB)
        anomaly_result = anomaly_detector.retrain(DB)
        if anomaly_result["trained"]:
            global _ANOMALY_MODEL, _ANOMALY_MODEL_LOAD_ATTEMPTED
            _ANOMALY_MODEL, _ANOMALY_MODEL_LOAD_ATTEMPTED = None, False   # force a reload next ingestion
        DB.audit(user, "model_retrained",
                 detail=f"samples={model['samples']}; rules={model['active_rules']}; "
                        f"anomaly_baseline={anomaly_result}")
        self._json({"model": model, "anomaly_baseline": anomaly_result})

    def api_reset(self, query: dict) -> None:
        """Clear all records. Admin only - used to re-run a clean demo."""
        user = self._require("purge")
        for table in ("corrections", "fields", "audit_log", "documents"):
            DB.run(f"DELETE FROM {table}")
        learning.save_model({"version": 1, "samples": 0, "confusions": [],
                             "aliases": [], "calibration": [], "active_rules": 0})
        DB.audit(user, "system_reset", detail="All documents and audit history cleared.")
        self._json({"status": "reset"})

    # -- API: export --------------------------------------------------
    def api_export_csv(self, query: dict) -> None:
        self._require("export")
        import csv
        keys = [s.key for s in FIELD_SPECS]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["document_id", "filename", "status", "trust_score",
                         "ocr_engine", "uploaded_at"] + keys)

        for row in DB.list_documents(limit=5000):
            fields = {f["field_key"]: f["value"] for f in DB.get_fields(row["id"])}
            writer.writerow([row["id"], row["filename"], row["status"],
                             row["trust_score"], row["ocr_engine"], row["uploaded_at"]]
                            + [fields.get(k, "") or "" for k in keys])

        self._send(200, buf.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8",
                   {"Content-Disposition": 'attachment; filename="land_records_export.csv"'})

    def api_export_json(self, query: dict) -> None:
        self._require("export")
        payload = []
        for row in DB.list_documents(limit=5000):
            doc = DB.get_document(row["id"]) or {}
            doc.pop("full_text", None)
            payload.append(doc)
        body = json.dumps({"records": payload}, ensure_ascii=False, indent=2, default=str)
        self._send(200, body.encode("utf-8"), "application/json; charset=utf-8",
                   {"Content-Disposition": 'attachment; filename="land_records_export.json"'})

    def api_cadastral_maps(self, query: dict) -> None:
        """Which village maps this installation holds."""
        self._json({"maps": [
            {"id": m["id"], "village": m["village"], "district": m["district"],
             "bundled": m["bundled"],
             "map_file": os.path.basename(m["map_path"])}
            for m in list_cadastral_maps()
        ]})

    def api_cadastral_upload(self, query: dict) -> None:
        """
        Add a real village cadastral map: the scanned map image plus the
        control_points.json that georeferences it.

        Admin-gated because a map is reference data that every record in that
        village will be validated against - a wrong one silently mis-tags a
        whole village rather than one document. Both files are required
        together: an image with no control points cannot be georeferenced,
        and this system does not display parcels it cannot place.
        """
        self._require("retrain")     # admin-only, same right as model retraining
        ctype = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in ctype.lower():
            raise ApiError(400, "Upload must be multipart/form-data.")

        parts = parse_multipart(self._read_body(), ctype)
        files = {p["name"]: p for p in parts if p.get("filename")}
        image = files.get("map")
        points = files.get("control_points")
        # Optional: the world file ArcGIS or QGIS writes beside the raster.
        # When present it carries the georeferencing itself, so no control
        # points are needed - see backend/georeference.py.
        world = files.get("world_file")
        if not image or not points:
            raise ApiError(400, "Both a 'map' image and a 'control_points' JSON file are required.")

        ext = os.path.splitext(image["filename"])[1].lower()
        if ext not in MAP_IMAGE_EXT:
            raise ApiError(400, f"Unsupported map image type '{ext}'.")

        try:
            meta = json.loads(points["data"].decode("utf-8"))
        except Exception as exc:
            raise ApiError(400, f"control_points is not valid JSON: {exc}")
        village = (meta.get("village") or "").strip()
        if not village:
            raise ApiError(400, "control_points.json must name the 'village' this map covers, "
                                "so records can be matched to it without colliding with "
                                "another village's khasra numbers.")

        # A georeferenced GeoTIFF carries its transform in its own tags, and a
        # world file carries it alongside. In either case demanding hand-picked
        # GCPs as well would be asking a surveyor to redo work they have
        # already done better; the JSON is then only there to name the village.
        embedded = ext in (".tif", ".tiff")
        if not world and not embedded and len(meta.get("control_points") or []) < 3:
            raise ApiError(400,
                           "At least 3 control points are needed to fit an affine "
                           "transform. Alternatively, upload the georeferencing "
                           "produced by ArcGIS or QGIS: send the world file "
                           "(.tfw/.jgw/.pgw) as 'world_file', or upload the map as "
                           "a georeferenced GeoTIFF - either must be in EPSG:4326.")

        # Short ASCII slug for the directory name: the village name is often
        # in an Indic script, which strips to nothing, so the Latin alias is
        # tried before falling back to an opaque id. A readable folder matters
        # because these maps are meant to be droppable in by hand.
        candidates = [village] + [a for a in (meta.get("village_aliases") or [])]
        map_id = ""
        for name in candidates:
            slug = re.sub(r"[^a-z0-9]+", "-", str(name).casefold()).strip("-")[:24]
            if slug:
                map_id = slug
                break
        map_id = map_id or ("map-" + uuid.uuid4().hex[:8])
        folder = os.path.join(CADASTRAL_STORE, map_id)
        if os.name == "nt" and len(os.path.abspath(os.path.join(folder, "control_points.json"))) >= _MAX_WINDOWS_PATH:
            raise ApiError(500, "Storage path is too long for this operating system. "
                                "Move this project to a shorter directory.")
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "map" + ext), "wb") as fh:
            fh.write(image["data"])
        with open(os.path.join(folder, "control_points.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        if world:
            # Saved under the sidecar name georeference.find_world_file expects
            # for this raster type, not under whatever the browser called it.
            sidecar = georeference.WORLD_FILE_EXTENSIONS.get(ext, (".wld",))[0]
            with open(os.path.join(folder, "map" + sidecar), "wb") as fh:
                fh.write(world["data"])

        # A new map changes what every record in that village matches against,
        # so nothing cached may survive it.
        _CADASTRAL_CACHE.clear()
        geojson = _cadastral_geojson(map_id)
        # The georeferencing is auditable evidence, not a detail. A boundary
        # dispute turns on who georeferenced this sheet, from which control
        # points, at what residual - and all of that was previously computed,
        # used to place the parcels, and then dropped on the floor. Without it
        # a reviewer can see the outlines but cannot defend where they came
        # from, which is the one thing an audit trail exists to support.
        geo = geojson.get("_georeferencing") or {}
        topo = _topology_report(geojson)
        DB.audit(self._current_user(), "cadastral_map_added", None,
                 detail="; ".join(str(x) for x in filter(None, [
                     f"village={village}",
                     f"parcels={len(geojson.get('features', []))}",
                     f"method={geo.get('method')}" if geo.get("method") else "",
                     f"gcps={geo.get('control_point_count')}"
                     if geo.get("control_point_count") is not None else "",
                     f"rms_m={geo.get('rms_metres')}"
                     if geo.get("rms_metres") is not None else "",
                     f"tolerance_met={geo.get('passed')}"
                     if geo.get("passed") is not None else "",
                     f"redundancy={geo.get('redundancy')}"
                     if geo.get("redundancy") is not None else "",
                     f"gcps_dropped={len(geo.get('dropped') or [])}"
                     if geo.get("dropped") else "",
                     f"crs={geo.get('crs') or geo.get('crs_name')}"
                     if (geo.get("crs") or geo.get("crs_name")) else "",
                     f"initial_rms_m={geo.get('initial_rms_metres')}"
                     if geo.get("initial_rms_metres") is not None else "",
                     f"topology_clean={topo.get('clean')}" if topo.get("checked") else "",
                     f"overlaps={len(topo.get('overlaps') or [])}"
                     if topo.get("overlaps") else "",
                     f"unsnapped={len(topo.get('unsnapped_vertices') or [])}"
                     if topo.get("unsnapped_vertices") else "",
                     f"invalid_rings={len(topo.get('invalid') or {})}"
                     if topo.get("invalid") else "",
                 ])))

        self._json({
            "id": map_id, "village": village,
            "parcels": len(geojson.get("features", [])),
            "error": geojson.get("_error"),
            "georeferencing": geojson.get("_georeferencing"),
            # Counts rather than the full witness lists: a sheet with three
            # hundred unsnapped vertices would otherwise bury the response.
            # The audit row carries the same counts; the detail is available
            # from the parcels endpoint.
            "topology": {
                "checked": topo.get("checked"),
                "units": topo.get("units"),
                "clean": topo.get("clean"),
                "invalid_rings": len(topo.get("invalid") or {}),
                "overlaps": len(topo.get("overlaps") or []),
                "slivers": len(topo.get("slivers") or []),
                "containments": len(topo.get("containments") or []),
                "unsnapped_vertices": len(topo.get("unsnapped_vertices") or []),
                "caveat": topo.get("caveat"),
            },
        })

    def api_cadastral_parcels(self, query: dict) -> None:
        """
        GeoJSON for the map view (S12a GIS integration). Vectorization is
        classical CV, cheap enough to not need a background job, but the
        result never changes for the bundled demo map, so it is computed
        once per process and cached rather than redone on every map load.
        The document-linkage overlay (which parcel matches an uploaded,
        validated document) is a live join and is recomputed every call.
        """
        map_id = (query.get("map") or [None])[0]
        self._json(_link_cadastral_documents(_cadastral_geojson(map_id)))


def _auto_seed(port: int) -> None:
    """
    Ingest the bundled corpus once, in the background, if the database is empty.

    Why this exists: a managed host gives every deploy a fresh, empty
    filesystem, so a judge opening the public URL would land on a dashboard of
    zeroes and conclude the system does not work. Seeding on first boot means
    the link always shows a populated demo.

    Two deliberate choices. It runs in a BACKGROUND thread started after the
    socket is listening, because ingesting fifteen documents takes tens of
    seconds and a platform health check that cannot connect in that window will
    kill the container before it ever serves a request. And it drives the real
    /api/seed endpoint over loopback rather than re-implementing the ingest
    loop, so the demo path a judge would click is the exact path that runs
    here - there is no second, untested copy of it to drift.

    Off unless AUTO_SEED=1, so `python3 run.py` on a laptop behaves as before.
    """
    if os.environ.get("AUTO_SEED") != "1":
        return
    try:
        if DB.one("SELECT id FROM documents LIMIT 1"):
            return                      # already populated; never duplicate
    except Exception:
        return

    import http.client

    def worker() -> None:
        # Wait for our own socket rather than sleeping a fixed guess.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                probe = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                probe.request("GET", "/api/stats")
                probe.getresponse().read()
                probe.close()
                break
            except Exception:
                time.sleep(1)
        else:
            print("  auto-seed: server never became reachable; skipped.",
                  file=sys.stderr)
            return
        try:
            print("  auto-seed: ingesting the bundled corpus...", flush=True)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
            conn.request("POST", "/api/seed", body=b"",
                         headers={"Content-Length": "0"})
            response = conn.getresponse()
            response.read()
            conn.close()
            print(f"  auto-seed: finished (HTTP {response.status}).", flush=True)
        except Exception as exc:
            # A failed seed leaves an empty but working system, which is far
            # better than a container that refuses to start.
            print(f"  auto-seed failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    threading.Thread(target=worker, name="auto-seed", daemon=True).start()


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    caps = ocr_engine.capabilities()
    httpd = ThreadingHTTPServer((host, port), Handler)
    _auto_seed(port)
    print("=" * 72)
    print("  Intelligent Land Record Digitization and Validation System")
    print("  SIH 2026 | PS 26018 | Ministry of Rural Development (DoLR)")
    print("=" * 72)
    print(f"  Server      : http://{host}:{port}")
    print(f"  PDF text    : {'yes' if caps['pdf_text_layer'] else 'no'}")
    print(f"  Preprocess  : {'yes (OpenCV)' if caps['image_preprocessing'] else 'no'}")
    print(f"  Tesseract   : {'yes -> ' + ', '.join(caps['tesseract_languages'][:8]) if caps['tesseract'] else 'NOT INSTALLED (scanned images queue for manual entry)'}")
    print(f"  Admin master: {'loaded' if validator_mod._MASTER.loaded else 'missing'}")
    print("=" * 72)
    print("  Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve(port=p)
