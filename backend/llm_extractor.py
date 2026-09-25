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

LLM_AVAILABLE = CONSENT_GIVEN and bool(ANTHROPIC_API_KEY or OPENAI_API_KEY or NVIDIA_API_KEY)


def _provider() -> Optional[str]:
    if not CONSENT_GIVEN:
        return None
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    if NVIDIA_API_KEY:
        return "nvidia"
    return None


def unavailable_reason() -> Optional[str]:
    """Human-readable explanation for run.py --check - honest about the
    privacy trade-off, not phrased as a bare 'install this' nudge like every
    other optional-dependency status line in this project."""
    any_key = ANTHROPIC_API_KEY or OPENAI_API_KEY or NVIDIA_API_KEY
    if CONSENT_GIVEN and any_key:
        return None
    if not any_key:
        return ("no ANTHROPIC_API_KEY/OPENAI_API_KEY/NVIDIA_API_KEY set")
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


def _normalise_for_grounding(text: str) -> str:
    """Casefold, unify Indic digits to ASCII, and collapse all whitespace, so
    a genuine extraction still matches OCR text that spaces or scripts its
    digits differently (the OCR of the verification sample rendered khata
    9073 as '९०७३' and khasra 42/4 as '§42/4')."""
    folded = field_extractor.normalise_digits(text or "").casefold()
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
