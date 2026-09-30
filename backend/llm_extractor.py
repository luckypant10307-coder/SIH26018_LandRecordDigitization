"""
Optional LLM-assisted field-extraction suggestions.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

READ THIS BEFORE ENABLING - see the two-flag activation gate below. Unlike
every other optional capability in this project, this one sends real
document content - for a land record, that means owner names, khasra
numbers and other identifying detail - to a third-party API over the
internet. Every other ML feature here (fact_checker's TF-IDF matching,
ner_extractor's spaCy NER, anomaly_detector's IsolationForest,
table_structure's PaddleOCR) runs entirely on this machine; this is the one
deliberate exception, added because it was explicitly requested with that
trade-off understood, not because it fits this project's usual stance.

ACTIVATION REQUIRES TWO SEPARATE THINGS, NOT JUST AN API KEY:
  1. ANTHROPIC_API_KEY, OPENAI_API_KEY, or NVIDIA_API_KEY is set.
  2. LLM_FIELD_EXTRACTION_CONSENT=1 is set.
An API key can exist in an environment for a reason that has nothing to do
with this project (a developer's own unrelated tooling, a shared CI
environment) and must never be treated as silent permission to export
land-record PII. Both must be present, deliberately, before a single byte
of document text leaves this machine - a stronger bar than every other
optional dependency here, because what is at stake (PII leaving the
machine) is different in kind, not just in degree, from "a library isn't
installed."

WHAT THIS DOES, AND DOES NOT DO: for fields field_extractor.py already
reported as missing or below the same 0.80 confidence threshold
validator.rule_low_confidence uses, this asks an LLM to read the document's
OCR text and suggest a value - explicitly instructed to never invent a
value the text does not support. A suggestion is tagged
source="llm:<model>" and shown to a human reviewer alongside, never instead
of, the rule-based result; it is never written into the accepted value the
way a confirmed value normally is. LLM output is not auditable the way
every other rule in this system is (a regex or a gazetteer lookup can be
read and verified line by line; a language model's reasoning cannot),
which is exactly why this project has stayed rule-based everywhere else -
this module is an explicit, bounded exception to that stance, not a
reversal of it.

Uses stdlib `urllib.request` only - no SDK, no extra pip install. That
makes this, unusually, one of the *lighter* optional dependencies in this
project despite depending on the most powerful technology of any of them.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import field_extractor

CONFIDENCE_THRESHOLD = 0.80     # matches validator.rule_low_confidence's default
MAX_TEXT_CHARS = 6000           # land records are short documents; caps prompt size/cost

# 20s was the first guess and it was measurably wrong: a full 17-field
# request against NVIDIA's hosted catalog timed out at 20s and the identical
# retry succeeded, so the original value was turning a working call into a
# silent no-op roughly at random. A badly-degraded scan is exactly the case
# where every field is eligible, the prompt is longest, and the answer is
# slowest - i.e. the timeout was tightest precisely when the feature was
# needed most. Overridable because a self-hosted NIM or a faster model has
# no reason to wait this long.
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "90"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
CONSENT_GIVEN = os.environ.get("LLM_FIELD_EXTRACTION_CONSENT") == "1"

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# NVIDIA's `GET /v1/models` lists ~80 catalog entries, but a given API key
# is not necessarily provisioned to actually invoke all of them - verified
# directly: this project's test key got HTTP 404 "Function ... Not found
# for account" on mistral-7b-instruct-v0.3, several Nemotron variants,
# Mixtral, Gemma and Granite, and only meta/llama-3.2-11b-vision-instruct
# actually worked. That is why this default is a vision-instruct model for
# a text-only task, not a mismatch - it is simply the model this project
# could confirm works, and a vision-language model handles a text-only
# prompt fine. A different NVIDIA key may have a different working set;
# override with NVIDIA_MODEL if this default 404s for yours.
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")

# Sarvam AI - an Indian provider with Indic-first models, which is the
# relevant property here: these documents are Devanagari.
#
# sarvam-105b is a REASONING model. It emits reasoning_content first and only
# produces `content` once it has finished thinking, so a small max_tokens
# returns finish_reason="length" with content=None - measured: 4000 tokens was
# not enough for a 43-line document and produced no answer at all. The budget
# below is sized for that, and the latency it buys is real: ~95s and ~9,000
# completion tokens for one plot report with 16 co-owners.
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
SARVAM_MODEL = os.environ.get("SARVAM_MODEL", "sarvam-105b")
SARVAM_MAX_TOKENS = int(os.environ.get("SARVAM_MAX_TOKENS", "16000"))
# And its own timeout. The shared REQUEST_TIMEOUT_SECONDS is 90, which was
# tuned for a non-reasoning model and is BELOW the measured ~95s this one
# takes on a 16-owner document - so the first wiring of this path timed out
# at exactly 90.1s and returned None, which reads identically to "the model
# found nothing". A timeout must not be able to masquerade as an empty
# answer, so this one is sized for the model that is actually being called.
SARVAM_TIMEOUT_SECONDS = int(os.environ.get("SARVAM_TIMEOUT_SECONDS", "300"))

LLM_AVAILABLE = CONSENT_GIVEN and bool(
    ANTHROPIC_API_KEY or OPENAI_API_KEY or NVIDIA_API_KEY or SARVAM_API_KEY)


# An explicit choice beats the fallback order. A machine can hold several
# keys for unrelated reasons, and silently picking one by position is how
# this project ended up calling NVIDIA for a Devanagari document while a
# Sarvam key sat unused two lines below.
LLM_PROVIDER = (os.environ.get("LLM_PROVIDER") or "").strip().lower() or None

# Sarvam is tried before the general-purpose providers because these are
# Devanagari land records and its models are Indic-first - measured on a real
# plot report, sarvam-105b returned all 16 co-owners with the correct
# father's name each, every value present verbatim in the source. That is a
# property of the documents this system reads, not a general ranking.
_PROVIDER_ORDER = (
    ("sarvam", lambda: SARVAM_API_KEY),
    ("anthropic", lambda: ANTHROPIC_API_KEY),
    ("openai", lambda: OPENAI_API_KEY),
    ("nvidia", lambda: NVIDIA_API_KEY),
)


def _provider() -> Optional[str]:
    if not CONSENT_GIVEN:
        return None
    if LLM_PROVIDER:
        for name, key in _PROVIDER_ORDER:
            if name == LLM_PROVIDER:
                # Named but unusable is an error worth surfacing as "off",
                # not silently substituting a different provider than the
                # one the operator asked for.
                return name if key() else None
        return None
    for name, key in _PROVIDER_ORDER:
        if key():
            return name
    return None


def unavailable_reason() -> Optional[str]:
    """Human-readable explanation for run.py --check - honest about the
    privacy trade-off, not phrased as a bare 'install this' nudge like every
    other optional-dependency status line in this project."""
    any_key = (ANTHROPIC_API_KEY or OPENAI_API_KEY or NVIDIA_API_KEY
               or SARVAM_API_KEY)
    if CONSENT_GIVEN and any_key:
        return None
    if not any_key:
        return ("no ANTHROPIC_API_KEY/OPENAI_API_KEY/NVIDIA_API_KEY/"
                "SARVAM_API_KEY set")
    return ("API key present but LLM_FIELD_EXTRACTION_CONSENT=1 not set - "
            "this sends document text to a third party, so an API key alone "
            "is deliberately not enough to turn it on")


def _prompt(text: str, field_keys: List[str]) -> str:
    fields_list = ", ".join(field_keys)
    return (
        "You are assisting with digitizing an Indian land record (khatauni/"
        "khasra document). Below is OCR-extracted text from one document.\n\n"
        f"Extract ONLY these fields, if clearly present in the text: {fields_list}.\n"
        "Return STRICT JSON: an object mapping each field key to its value "
        "exactly as it appears in the text. If a field is not clearly "
        "present, omit its key entirely - never guess, infer, or fabricate "
        "a value the text does not support. Return nothing but the JSON "
        "object, no commentary, no code fence.\n\n"
        "--- DOCUMENT TEXT ---\n"
        f"{text[:MAX_TEXT_CHARS]}\n"
        "--- END DOCUMENT TEXT ---"
    )


def _call_anthropic(prompt: str) -> Optional[str]:
    body = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY or "",
            "anthropic-version": "2023-06-01",
        })
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    parts = payload.get("content") or []
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return text or None


def _call_openai(prompt: str) -> Optional[str]:
    body = json.dumps({
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY or ''}",
        })
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choices = payload.get("choices") or []
    return choices[0]["message"]["content"] if choices else None


def _call_nvidia(prompt: str) -> Optional[str]:
    """NVIDIA's hosted NIM catalog (integrate.api.nvidia.com) exposes an
    OpenAI-compatible chat-completions surface, so this is _call_openai's
    request/response shape against a different host, key header, and model
    catalog - not a separate protocol."""
    body = json.dumps({
        "model": NVIDIA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions", data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NVIDIA_API_KEY or ''}",
        })
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choices = payload.get("choices") or []
    return choices[0]["message"]["content"] if choices else None


# Invisible characters that change nothing about WHICH name a string is.
#
# Devanagari uses ZWJ (U+200D) and ZWNJ (U+200C) to control whether a
# conjunct renders as a ligature or with an explicit halant. They are
# presentation, not identity, and OCR output and model output disagree about
# them constantly.
#
# Measured: the source read "श्‍यामसुन्‍दर" with a ZWJ after the first
# letter; the model returned the same name without it, and the grounding
# check discarded a CORRECT owner as not-in-source. That is the expensive
# direction for this guard to fail in - a false rejection silently loses
# real data, and looks exactly like the model having invented something.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))


def _normalise_for_grounding(text: str) -> str:
    """Casefold, unify Indic digits to ASCII, drop zero-width formatting
    marks, and collapse all whitespace, so a genuine extraction still matches
    OCR text that spaces, scripts or joins its characters differently (the
    OCR of the verification sample rendered khata 9073 as '९०७३' and
    khasra 42/4 as '§42/4')."""
    folded = field_extractor.normalise_digits(text or "").casefold()
    folded = folded.translate(_ZERO_WIDTH)
    return re.sub(r"\s+", "", folded)


def is_grounded(value: str, source_text: str) -> bool:
    """
    True only if `value` actually occurs in the document text.

    THIS IS THE GUARD THAT MAKES THIS MODULE SAFE TO SHIP, and it exists
    because of an observed failure, not a hypothetical one. On the first
    real end-to-end run against a degraded scan, the model returned a
    complete, fluent, entirely invented land record - 'Khasra No. 1234',
    'Shri Ram Kumar', 'District - Alwar', 'State - Rajasthan' - for a
    Madhya Pradesh document whose real values (42/4, सुनीता बाई, Bhopal)
    were sitting in the prompt it had been given, and which the same model
    had read correctly on an earlier call. Instructing a model not to
    fabricate does not stop it from fabricating.

    Extraction has a property that makes this cheap to enforce: a real
    extracted value must, by definition, appear in the source document. So
    a suggestion that cannot be found in the OCR text is discarded no
    matter how plausible it reads. That turns the LLM from something whose
    word is taken into something that can only point at text which
    demonstrably exists - a non-deterministic component wrapped in a
    deterministic, auditable check, which is the only shape in which it
    belongs anywhere near a legal land record.
    """
    if not value or not source_text:
        return False
    return _normalise_for_grounding(value) in _normalise_for_grounding(source_text)


def _extract_json(text: str) -> Optional[dict]:
    """LLMs occasionally wrap JSON in prose or a code fence despite being
    told not to - this recovers the first {...} block rather than failing
    outright on an otherwise-usable response."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def suggest_fields(full_text: str, field_keys: List[str]) -> Dict[str, dict]:
    """
    Returns {field_key: {"suggested_value": str, "source": "llm:<model>"}}
    for only the fields the model actually returned a value for - never
    fabricates an entry for a field it omitted. An empty dict means the
    model genuinely offered nothing usable; it does NOT mean "something
    went wrong".

    RAISES on transport failure (timeout, HTTP error, unreachable host),
    deliberately. An earlier version swallowed those and returned {},
    which made a timed-out request indistinguishable from a model that
    honestly found nothing - so a failure showed the reviewer exactly
    nothing, in a project whose entire premise (README S3) is that the
    system must never quietly do nothing and let it read as success. The
    caller (server._add_llm_suggestions) turns the exception into a visible
    LLM_SUGGESTION_FAILED issue, the same contract
    document_authenticity.analyze_document already has with
    _add_authenticity_issues.
    """
    provider = _provider()
    if provider is None or not field_keys:
        return {}

    prompt = _prompt(full_text or "", field_keys)
    if provider == "anthropic":
        raw = _call_anthropic(prompt)
        model = ANTHROPIC_MODEL
    elif provider == "openai":
        raw = _call_openai(prompt)
        model = OPENAI_MODEL
    else:
        raw = _call_nvidia(prompt)
        model = NVIDIA_MODEL

    # An unparseable response is a model-quality outcome, not a transport
    # failure - the request worked, the answer was just unusable - so it
    # stays a quiet empty result rather than an error the reviewer must act on.
    parsed = _extract_json(raw or "")
    if not isinstance(parsed, dict):
        return {}

    out: Dict[str, dict] = {}
    for key in field_keys:
        value = parsed.get(key)
        if value is None or not str(value).strip():
            continue
        value = str(value).strip()
        # Every suggestion must be findable in the document it claims to
        # come from. See is_grounded() for the observed fabrication this
        # prevents - it is not a defensive nicety, it is the only reason
        # this module's output is usable at all.
        if not is_grounded(value, full_text or ""):
            continue
        out[key] = {"suggested_value": value, "source": f"llm:{model}"}
    return out


# ==========================================================================
# Structured whole-document extraction
#
# WHY THIS IS SEPARATE FROM suggest_fields()
#
# suggest_fields asks for individual values the rule extractor missed, and
# returns one value per field. That shape is the problem here, not the
# accuracy: a Bhu-Naksha plot report lists every co-owner of the parcel, and
# measured across 20 genuine documents there are 162 owner rows. The 17-field
# schema holds exactly ONE owner_name, so the rule extractor collapses the
# line
#
#   "नाम : इन्द्रभान  संरक्षक का नाम : हरिप्रसाद  निवास स्थान : नि.ग्राम"
#
# to "नि.ग्रााम" - the residence marker at the end, not a name at all,
# because it takes the last colon on a line carrying three label:value pairs.
#
# This function asks for the document's real shape instead, owners as a LIST,
# and is the one place in this project where a model is asked to decide
# structure rather than to read a value.
#
# MEASURED, on the 16-owner document: 16 of 16 owners returned with the
# correct father's name each, 38 of 38 values present verbatim in the source,
# nothing invented - and village/tehsil/district recovered from the
# mutation-order prose, which the rule path cannot reach because the
# gazetteer has no entry to confirm them against.
#
# EVERY VALUE IS STILL GROUNDED. is_grounded() is applied to each one and
# anything not present in the source text is DROPPED, not stored. A model
# that returns sixteen plausible names which are not in the document is worse
# than a rule that returns nothing.
# ==========================================================================

STRUCTURED_SCHEMA = """Return ONLY a JSON object, no prose, no code fence,
with exactly these keys:

{
  "khasra_number": string|null,
  "khata_number": string|null,
  "area": string|null,
  "village": string|null,
  "tehsil": string|null,
  "district": string|null,
  "owners": [ {"name": string, "father_name": string|null} ]
}

Rules:
- Copy values EXACTLY as they appear in the document. Do not translate,
  transliterate, correct or normalise them.
- "owners" must contain EVERY co-owner listed, in document order.
- In a line of the form "नाम : X संरक्षक का नाम : Y निवास स्थान : Z",
  X is the owner's name, Y is the father's or guardian's name, and Z is a
  residence marker which is NOT a name.
- If a field does not appear in the document, use null. Never invent a value.
"""


def _call_sarvam(prompt: str) -> Optional[str]:
    """
    Sarvam's chat API is OpenAI-shaped, with one difference that matters.

    sarvam-105b reasons before answering, so the answer arrives in `content`
    only after `reasoning_content` is complete. A response truncated by the
    token budget has content=None and finish_reason="length" - which is a
    failure to answer, not an empty answer, and must not be read as "the
    model found nothing".
    """
    body = json.dumps({
        "model": SARVAM_MODEL,
        "messages": [
            {"role": "system",
             "content": "You extract structured data from Indian land "
                        "records. You output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": SARVAM_MAX_TOKENS,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.sarvam.ai/v1/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + (SARVAM_API_KEY or "")})
    with urllib.request.urlopen(req, timeout=SARVAM_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choice = (payload.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length" and not choice.get("message", {}).get("content"):
        return None                       # ran out of budget mid-reasoning
    return (choice.get("message") or {}).get("content")


def extract_structured(full_text: str) -> Optional[dict]:
    """
    The document's real shape, including every co-owner.

    Returns None when unavailable or when the model did not answer - never a
    partial or invented record. Returns a dict whose values have each been
    checked against the source text; ungrounded values are dropped and
    counted in the "_dropped" key so the loss is visible rather than silent.
    """
    if not LLM_AVAILABLE:
        return None
    provider = _provider()
    if provider is None:
        return None

    prompt = STRUCTURED_SCHEMA + "\n\nDOCUMENT:\n" + full_text[:MAX_TEXT_CHARS]
    caller = {"anthropic": _call_anthropic, "openai": _call_openai,
              "nvidia": _call_nvidia, "sarvam": _call_sarvam}[provider]
    try:
        raw = caller(prompt)
    except Exception:
        return None
    if not raw:
        return None

    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None

    dropped = []

    def keep(value):
        return value if (value and is_grounded(str(value), full_text)) else None

    out = {"_provider": f"llm:{provider}", "_model": (
        SARVAM_MODEL if provider == "sarvam" else provider)}
    for key in ("khasra_number", "khata_number", "area",
                "village", "tehsil", "district"):
        value = data.get(key)
        kept = keep(value)
        if value and not kept:
            dropped.append(f"{key}={value!r}")
        out[key] = kept

    owners = []
    for entry in (data.get("owners") or []):
        if not isinstance(entry, dict):
            continue
        name = keep(entry.get("name"))
        if not name:
            if entry.get("name"):
                dropped.append(f"owner={entry.get('name')!r}")
            continue                      # an owner we cannot ground is not an owner
        owners.append({"name": name,
                       "father_name": keep(entry.get("father_name"))})
    out["owners"] = owners
    out["_dropped"] = dropped
    return out
