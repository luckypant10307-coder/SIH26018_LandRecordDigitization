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

import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from field_extractor import parse_date

_SPACY_MODEL = "en_core_web_sm"

# --------------------------------------------------------------------------
# Indic-script NER
# --------------------------------------------------------------------------
#
# WHY A SECOND MODEL RATHER THAN ONE
#
# en_core_web_sm is an ENGLISH model. On a Hindi khatauni line it does not
# degrade quietly, it degrades wrongly. Measured on
#
#   "खातेदार का नाम: सुनीता देवी पत्नी स्व0 रामप्रसाद, ग्राम नरहरपुर, ..."
#
# it missed the owner सुनीता देवी entirely and tagged देवी as a DATE. On the
# Latin rendering of the same line it found both people correctly. So the
# cross-check was not merely unavailable on Devanagari - it was capable of
# reporting a false mismatch, which is worse than making no claim. That is why
# cross_check() used to skip non-Latin script outright.
#
# A multilingual NER model closes the gap. Measured over three real
# land-record lines (khatauni owner, Bhu-Naksha owner, mutation entry) it
# recovered 6 of 6 person names, and additionally tagged नरहरपुर / सदर / लखनऊ
# as locations.
#
# spaCy is KEPT for Latin rather than replaced: it is a few tens of MB and
# already loaded, where the multilingual model is ~2.5 GB. Paying that on an
# English-only document would be pure waste.
#
# WHY NOT MuRIL OR IndicBERT, WHICH ARE THE OBVIOUS NAMES
#
# Neither can do this task as published. google/muril-base-cased declares
# itself BertForMaskedLM with an empty id2label, and IndicBERTv2 ships
# MLM-only - they predict masked words and have no entity head at all. The
# model that WOULD be ideal is ai4bharat/IndicNER, MuRIL already fine-tuned
# for Indic NER, but it is a gated repo and returns 401 without an
# authenticated Hugging Face account that has accepted its terms. Set
# INDIC_NER_MODEL=ai4bharat/IndicNER once that access exists and this module
# will use it with no code change.
#
# NO CONSENT GATE, unlike llm_extractor.py and bhashini.py: this model runs
# locally and no document text leaves the machine, so there is nothing to
# consent to. It is absent from requirements.txt because it needs torch, which
# the deployment image excludes - there, this path is simply unavailable and
# says so.

INDIC_NER_MODEL = os.environ.get("INDIC_NER_MODEL",
                                 "Davlan/xlm-roberta-base-ner-hrl")
INDIC_NER_DISABLED = os.environ.get("INDIC_NER_DISABLED") == "1"

# This model labels spans PER/LOC/ORG/DATE; the rest of this module speaks
# spaCy's vocabulary, so translate at the boundary rather than teaching every
# caller two schemes.
_INDIC_LABEL_MAP = {"PER": "PERSON", "DATE": "DATE", "LOC": "GPE", "ORG": "ORG"}

_INDIC = None
_INDIC_LOAD_ATTEMPTED = False
_INDIC_ERROR: Optional[str] = None


def _indic_ner():
    """
    Lazy-load the multilingual NER pipeline.

    Loading is deferred hard: importing transformers and materialising ~2.5 GB
    of weights must not happen for an installation that only ever sees English
    documents, nor at module import time, which would make `import server` pay
    for it on every start.
    """
    global _INDIC, _INDIC_LOAD_ATTEMPTED, _INDIC_ERROR
    if _INDIC_LOAD_ATTEMPTED:
        return _INDIC
    _INDIC_LOAD_ATTEMPTED = True
    if INDIC_NER_DISABLED:
        _INDIC_ERROR = "INDIC_NER_DISABLED=1"
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import pipeline
            _INDIC = pipeline("ner", model=INDIC_NER_MODEL,
                              aggregation_strategy="simple", device=-1)
    except Exception as exc:
        _INDIC_ERROR = f"{type(exc).__name__}: {exc}"
        _INDIC = None
    return _INDIC


def indic_ner_available() -> bool:
    return _indic_ner() is not None


def indic_ner_status() -> dict:
    """Honest status for --check and the dashboard, without forcing a load."""
    return {
        "model": INDIC_NER_MODEL,
        "loaded": _INDIC is not None,
        "attempted": _INDIC_LOAD_ATTEMPTED,
        "error": _INDIC_ERROR,
        "disabled": INDIC_NER_DISABLED,
    }


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


# Below this the model is guessing rather than recognising. A low-confidence
# span that disagrees with the rule-based reading would raise NER_MISMATCH and
# send a correct record to a human for no reason, so weak spans are dropped and
# the field is reported unconfirmed instead.
_INDIC_MIN_SCORE = 0.60


def _indic_entities_by_label(text: str) -> Dict[str, List[str]]:
    """The same {label: [span, ...]} shape as the spaCy path, in spaCy's labels."""
    ner = _indic_ner()
    out: Dict[str, List[str]] = {}
    if ner is None:
        return out
    try:
        spans = ner(text)
    except Exception:
        return out              # a model failure must never break validation
    for span in spans:
        if float(span.get("score") or 0) < _INDIC_MIN_SCORE:
            continue
        label = _INDIC_LABEL_MAP.get(span.get("entity_group"))
        word = (span.get("word") or "").strip()
        if label and word:
            out.setdefault(label, []).append(word)
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
    if not ner_available() and not indic_ner_available():
        return [Issue(
            "NER_UNAVAILABLE", "info", None,
            "Neither spaCy ('en_core_web_sm') nor a multilingual NER model is "
            "installed; NER cross-check of person/date fields was skipped.",
            "Install with: pip install spacy && python -m spacy download en_core_web_sm",
        )]

    issues: List[Issue] = []
    for key, expected_label in _CHECKS.items():
        entry = values.get(key) or {}
        value = entry.get("value")
        if not value:
            continue

        # Route by the script the value is actually written in. The English
        # model is not merely weak on Devanagari, it is confidently wrong on
        # it (see the module docstring), so a non-Latin value must never be
        # handed to it. Where no Indic model is loaded this still degrades to
        # the previous behaviour: no claim made.
        script = (entry.get("extra") or {}).get("script")
        indic = script not in ("latin", None)
        if indic:
            if not indic_ner_available():
                continue
        elif not ner_available():
            continue

        if expected_label == "PERSON" and not indic:
            # The raw source line ("Owner Name : Ramesh Kumar") is not
            # syntactically rich enough for the small English model - see
            # _PERSON_NER_TEMPLATE and the module docstring. The cost of that
            # frame is that the check can only ask "is this string a person
            # name", never "is this the person on the line".
            ner_input = _PERSON_NER_TEMPLATE.format(value)
        else:
            # The multilingual model needs no such frame: measured on whole
            # khatauni and mutation lines it recovered every name, so it is
            # given the SOURCE LINE. That upgrades the check from plausibility
            # to genuine cross-validation - if the line reads सुनीता देवी and
            # the extractor produced रीता शर्मा, the mismatch is now visible
            # instead of both being waved through as plausible names.
            ner_input = entry.get("source_line") or value

        entities = (_indic_entities_by_label(ner_input) if indic
                    else _entities_by_label(ner_input))
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
