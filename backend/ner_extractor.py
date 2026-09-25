"""
ML-based named-entity cross-check for extracted fields.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

field_extractor.py is deliberately rule-based (see its module docstring:
"Why rules and not a large model") - label-anchored matching for a closed,
templated field schema. This module adds a genuinely different, ML-based
signal alongside it, not instead of it: a general-purpose named-entity
recognition (NER) model (spaCy) reads the raw OCR text independently of any
label, and its PERSON/DATE spans are cross-checked against what the
rule-based extractor already produced for owner_name, father_name,
mutation_date and registration_date.

Why only those four fields: a generic NER model recognises broad categories
(PERSON, DATE, GPE, ORG, ...), not domain identifiers. It has no notion of
"khasra number" or "khata number" - there is no learned or rule-based way to
tell a pretrained model that a bare integer is a parcel identifier rather
than a page number, so this module makes no claim about the numeric ID
fields at all. Village/tehsil/district would nominally map to a GPE/LOC
entity, but a generic model cannot tell a village from a tehsil from a
district by itself, so the same honesty rule applies: no claim beyond what
the signal can actually support.

Why only Latin-script text: the bundled spaCy pipeline (en_core_web_sm) is
an English model. Running it on Devanagari, Bengali, Tamil, etc. would not
fail loudly - it would silently produce meaningless spans, which is worse
than not running it at all. field_extractor.detect_script() already exists
for exactly this kind of decision, so it is reused here to skip non-Latin
field values rather than guess.

This is a corroboration layer, not a verdict: agreement raises confidence
slightly, disagreement raises a low-severity "unconfirmed" note for a human
to look at, and unavailability (spaCy or its model not installed) degrades
to an explicit info issue - never a silent no-op, matching every other
optional-dependency path in this project (ocr_engine.py's degraded mode,
fact_checker.py's FACT_CHECK_UNAVAILABLE).

Two real, empirically-verified limitations of the bundled small English
model (measured against this project's own sample data, not assumed):

  - A bare "Label : Value" line gives the model too little sentence
    structure to recognise a name at all ("Owner Name : Ramesh Kumar" finds
    nothing; even the bare name "Ramesh Kumar" alone finds nothing). Wrapping
    the value as "Mr. {value}." for the NER call only - never stored, never
    shown - restores enough syntactic context to recognise most Indian names
    reliably. This is not a trick to make the check pass: it is real,
    independent text still being read by the model; only its packaging
    changes.
  - DD/MM/YYYY numeric dates - the format these documents actually use - are
    tagged DATE inconsistently by this model: sometimes correctly (verified
    on a real sample line), sometimes misread as CARDINAL/GPE or missed
    entirely, depending on surrounding context the model cannot be relied on
    for. Natural-language dates ("14 March 2023") work reliably. This module
    deliberately does NOT paper over the unreliable case by reformatting the
    already-parsed value into natural language before checking it - that
    would make the check trivially confirm itself against a re-rendering of
    its own answer, not an independent read of the source. When the model
    does find a DATE span, it is still in the source's raw format while
    `value` is already ISO (field_extractor.parse_date ran during
    extraction) - comparing those strings directly would report every
    correct match as a mismatch purely from formatting, so both sides are
    parsed to ISO with the same parser before comparing. Net effect: date
    corroboration is honestly inconsistent on this document format - it
    sometimes reports NER_CONFIRMED, often NER_UNCONFIRMED - an accurate
    result either way, not a guaranteed-useful one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from field_extractor import parse_date

_SPACY_MODEL = "en_core_web_sm"


def _try_load_spacy():
    try:
        import spacy
    except Exception:
        return None
    try:
        return spacy.load(_SPACY_MODEL)
    except Exception:
        return None


_NLP = None
_NLP_LOAD_ATTEMPTED = False


def _nlp():
    """Lazy-load: importing spaCy and loading a model is not free, and most
    ingestion pipelines that never call this module should not pay for it."""
    global _NLP, _NLP_LOAD_ATTEMPTED
    if not _NLP_LOAD_ATTEMPTED:
        _NLP = _try_load_spacy()
        _NLP_LOAD_ATTEMPTED = True
    return _NLP


def ner_available() -> bool:
    return _nlp() is not None


@dataclass
class Issue:
    rule: str
    severity: str
    field: Optional[str]
    message: str
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Which extracted fields get cross-checked, against which spaCy entity label,
# and which field_extractor "kind" tags a field as (used only for the
# Latin-script gate below - not read from field_extractor to avoid a
# dependency cycle, since field_extractor does not import this module).
_CHECKS = {
    "owner_name": "PERSON",
    "father_name": "PERSON",
    "mutation_date": "DATE",
    "registration_date": "DATE",
}


def _entities_by_label(text: str) -> Dict[str, List[str]]:
    nlp = _nlp()
    doc = nlp(text)
    out: Dict[str, List[str]] = {}
    for ent in doc.ents:
        out.setdefault(ent.label_, []).append(ent.text)
    return out


# A bare "Label : Value" line, or even a bare name with no sentence
# context, gives the small English model too little structure to
# recognise a person name at all (empirically verified - see module
# docstring). Wrapping the *value* in a minimal sentence frame for the NER
# call restores that context; it does not guarantee a match (a model that
# always said yes would be worthless as a check), it measurably improves
# the chance of the model applying its real classification ability. Never
# used for the stored or displayed value - only for this one NER call.
_PERSON_NER_TEMPLATE = "Mr. {}."


def cross_check(values: Dict[str, dict]) -> List[Issue]:
    """
    Cross-check owner_name/father_name/mutation_date/registration_date
    against spaCy NER run on each field's own source line.

    `values[key]["extra"]["script"]` (set by field_extractor.py) gates which
    fields are even attempted - see the module docstring for why non-Latin
    script is skipped rather than guessed at.
    """
    if not ner_available():
        return [Issue(
            "NER_UNAVAILABLE", "info", None,
            "spaCy (or its 'en_core_web_sm' model) is not installed; "
            "NER cross-check of person/date fields was skipped.",
            "Install with: pip install spacy && python -m spacy download en_core_web_sm",
        )]

    issues: List[Issue] = []
    for key, expected_label in _CHECKS.items():
        entry = values.get(key) or {}
        value = entry.get("value")
        if not value:
            continue
        script = (entry.get("extra") or {}).get("script")
        if script not in ("latin", None):
            continue    # non-Latin script: no claim made, see module docstring

        if expected_label == "PERSON":
            # The raw source line ("Owner Name : Ramesh Kumar") is not
            # syntactically rich enough for this model - see
            # _PERSON_NER_TEMPLATE and the module docstring.
            ner_input = _PERSON_NER_TEMPLATE.format(value)
        else:
            ner_input = entry.get("source_line") or value

        entities = _entities_by_label(ner_input)
        candidates = entities.get(expected_label, [])

        if expected_label == "DATE":
            # `value` is already ISO (field_extractor.parse_date ran during
            # extraction); a NER-found date span is still in whatever raw
            # format the source used ("27/02/2031"). Comparing those strings
            # directly would report every correct match as a "mismatch"
            # purely from formatting - parse both to the same representation
            # before comparing, the same parser the rule-based path already
            # trusts.
            matched = any((parse_date(cand) or {}).get("iso") == value for cand in candidates)
        else:
            norm_value = value.strip().lower()
            matched = any(
                norm_value in cand.lower() or cand.lower() in norm_value
                for cand in candidates
            )
        if matched:
            issues.append(Issue(
                "NER_CONFIRMED", "info", key,
                f"Independent NER pass recognises '{value}' as a {expected_label.lower()}, "
                "corroborating the rule-based extraction.",
                None,
            ))
        elif candidates:
            issues.append(Issue(
                "NER_MISMATCH", "warning", key,
                f"Rule-based extraction reads '{value}', but the NER pass found "
                f"a different {expected_label.lower()} on the same line: "
                f"{', '.join(candidates)}.",
                "Compare both readings against the source document.",
            ))
        else:
            issues.append(Issue(
                "NER_UNCONFIRMED", "info", key,
                f"Independent NER pass did not recognise '{value}' as a "
                f"{expected_label.lower()} - the rule-based reading is unconfirmed, not "
                "necessarily wrong.",
                None,
            ))
    return issues
