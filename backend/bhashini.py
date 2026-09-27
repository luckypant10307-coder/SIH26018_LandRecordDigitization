"""
Script transliteration through Bhashini (ULCA / Dhruva).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

THE GAP THIS CLOSES
-------------------
ocr_engine.py reads a village or district name in fourteen Indic scripts.
validator.py then checks that name against backend/data/admin_master.json,
which contains 65,869 Latin characters and zero Devanagari. So a name the OCR
read perfectly cannot be validated at all:

    district_exists("Lucknow")  -> "Uttar Pradesh"
    district_exists("लखनऊ")     -> None          # the same district

That is not an OCR failure and no amount of better reading fixes it. The name
has to be rendered into the script the reference data actually uses.

WHY NOT A CHARACTER MAPPING
---------------------------
A hand-written Devanagari->Latin table was tried first and measured on eight
well-known districts. It resolved five. The three failures split into two
classes, and only one of them is fixable by rules:

  * SCHWA DELETION. Devanagari writes an implicit vowel after every consonant
    that speakers do not pronounce: जयपुर is written "jayapura" and said
    "Jaipur". Knowing which implicit vowels to drop is a real problem in Indic
    text processing. Bhashini's IndicXlit model learned it from data and
    returns "jaipur" as its first suggestion. The character table returned
    "jayapura", which fell outside the fuzzy matcher's threshold.

  * ANGLICISED EXONYMS. लखनऊ is "Lucknow" and दिल्ली is "Delhi" by history,
    not by phonetics. Measured against the live API, Bhashini returns
    "lakhanau" and "dilli" - correct transliterations and still not the names
    in the master. NO transliteration service will ever produce these, so they
    are handled by the lookup table in data/place_exonyms.json, which is
    consulted BEFORE the network is touched.

TWO MEASURED API DETAILS THAT SHAPE THIS MODULE
-----------------------------------------------
* isSentence. With it false, a multi-word name comes back as a flat list of
  per-word alternatives - "कानपुर नगर" gives ['kanpur','kaanpur','kaanapur',
  'nagar','nager','nugger'], which cannot be reassembled without guessing
  where one word's suggestions end. With it true the same input returns
  ['kanpur nagar']. So the primary request sends isSentence=true, and the
  per-word alternatives are fetched only when the primary candidate fails to
  resolve.
* The right answer is not always the first suggestion. पुणे returns
  ['puney','pune','punay'] - the master's "Pune" is second. Callers must try
  every candidate rather than trusting suggestion one.

TLS
---
dhruva-api.bhashini.gov.in chains to the Indian emSign root, which is present
in certifi but not in the Windows trust store, so Python fails to verify it
while curl succeeds. This module builds its context from certifi when it is
installed and falls back to the system store otherwise. It NEVER disables
verification: silently accepting any certificate on a government API would be
a worse bug than the one it papers over.

DEGRADATION AND CONSENT
-----------------------
Bhashini is a third party, and a place name is land-record content, so the
same discipline llm_extractor.py applies holds here: nothing leaves this
machine unless BHASHINI_CONSENT=1. With no keys, no consent or no network,
resolution falls back to the exonym table and the existing fuzzy matcher, and
unavailable_reason() says which, rather than failing silently.
"""

from __future__ import annotations

import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXONYM_PATH = os.path.join(_HERE, "data", "place_exonyms.json")
_CACHE_PATH = os.path.join(_HERE, "..", "storage", "transliteration_cache.json")

AUTH_URL = os.environ.get(
    "BHASHINI_AUTH_URL",
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline")
# The MeitY-published pipeline. Overridable because a state deployment may be
# issued its own.
PIPELINE_ID = os.environ.get("BHASHINI_PIPELINE_ID", "64392f96daac500b55c543cd")

PROJECT_ID = os.environ.get("BHASHINI_PROJECT_ID")
UDYAT_KEY = os.environ.get("BHASHINI_UDYAT_KEY")
CONSENT_GIVEN = os.environ.get("BHASHINI_CONSENT") == "1"

REQUEST_TIMEOUT_SECONDS = int(os.environ.get("BHASHINI_TIMEOUT_SECONDS", "45"))
# How many names go in one inference call. The API accepts a list; batching
# keeps a bulk ingest from opening one connection per village.
MAX_BATCH = int(os.environ.get("BHASHINI_MAX_BATCH", "25"))
NUM_SUGGESTIONS = 3
# A resolved pipeline config carries a bearer token; re-fetch hourly rather
# than once per process, so a long-running server does not hold a stale one.
_CONFIG_TTL_SECONDS = 3600

AVAILABLE = bool(PROJECT_ID and UDYAT_KEY and CONSENT_GIVEN)

_LOCK = threading.Lock()
_config: Optional[dict] = None
_config_at: float = 0.0
_cache: Optional[Dict[str, List[str]]] = None
_cache_dirty = False
_last_error: Optional[str] = None


def unavailable_reason() -> Optional[str]:
    """Why transliteration is off, in words a deployer can act on."""
    if not PROJECT_ID or not UDYAT_KEY:
        return "BHASHINI_PROJECT_ID / BHASHINI_UDYAT_KEY are not set"
    if not CONSENT_GIVEN:
        return ("BHASHINI_CONSENT is not 1 - place names would be sent to a "
                "third-party service, so this is opt-in")
    return None


def _ssl_context() -> ssl.SSLContext:
    """
    A verifying context that trusts the Indian emSign root.

    certifi carries it; the Windows store often does not. Falling back to the
    system store is correct - it may well work on Linux - but verification
    stays on either way.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _post(url: str, body: dict, headers: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers, **{"Content-Type": "application/json"}))
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS,
                                context=_ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


# --------------------------------------------------------------------------
# Exonyms - consulted before the network, because no model can derive these
# --------------------------------------------------------------------------

_exonyms: Optional[Dict[str, str]] = None


def _load_exonyms() -> Dict[str, str]:
    global _exonyms
    if _exonyms is None:
        try:
            with open(_EXONYM_PATH, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            _exonyms = {k.strip(): v for k, v in
                        (payload.get("exonyms") or {}).items()}
        except Exception:
            _exonyms = {}
    return _exonyms


def exonym(name: str) -> Optional[str]:
    """The established English name for an Indic-script place, if there is one."""
    return _load_exonyms().get((name or "").strip())


# --------------------------------------------------------------------------
# Transliteration cache - place names repeat heavily across a district's records
# --------------------------------------------------------------------------

def _load_cache() -> Dict[str, List[str]]:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            _cache = {k: list(v) for k, v in loaded.items()
                      if isinstance(v, list)}
        except Exception:
            _cache = {}
    return _cache


def flush_cache() -> None:
    """Persist the cache. Cheap to call; writes only when something changed."""
    global _cache_dirty
    if not _cache_dirty or _cache is None:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(_CACHE_PATH)), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh, ensure_ascii=False, indent=1, sort_keys=True)
        _cache_dirty = False
    except Exception:
        pass


def _cache_key(text: str, sentence: bool) -> str:
    return ("s:" if sentence else "w:") + text


# --------------------------------------------------------------------------
# Pipeline configuration
# --------------------------------------------------------------------------

def _pipeline_config(source_language: str) -> dict:
    """
    Resolve the inference endpoint, its bearer token and the service id.

    Cached: this is a setup call, and doing it per lookup would triple the
    network cost of transliterating one village name.
    """
    global _config, _config_at, _last_error
    now = time.time()
    with _LOCK:
        fresh = (_config is not None
                 and _config.get("source_language") == source_language
                 and (now - _config_at) < _CONFIG_TTL_SECONDS)
        if fresh:
            return _config

    body = {
        "pipelineTasks": [{
            "taskType": "transliteration",
            "config": {"language": {"sourceLanguage": source_language,
                                    "targetLanguage": "en"}},
        }],
        "pipelineRequestConfig": {"pipelineId": PIPELINE_ID},
    }
    payload = _post(AUTH_URL, body,
                    {"userID": PROJECT_ID, "ulcaApiKey": UDYAT_KEY})

    endpoint = payload.get("pipelineInferenceAPIEndPoint") or {}
    key = endpoint.get("inferenceApiKey") or {}
    tasks = payload.get("pipelineResponseConfig") or []
    services = (tasks[0].get("config") or []) if tasks else []
    if not (endpoint.get("callbackUrl") and key.get("name") and services):
        raise RuntimeError("Bhashini returned no usable transliteration pipeline "
                           f"for source language '{source_language}'.")

    resolved = {
        "source_language": source_language,
        "url": endpoint["callbackUrl"],
        "header_name": key["name"],
        "header_value": key.get("value") or "",
        "service_id": services[0].get("serviceId"),
    }
    with _LOCK:
        _config, _config_at = resolved, now
        _last_error = None
    return resolved


# --------------------------------------------------------------------------
# Transliteration
# --------------------------------------------------------------------------

def transliterate(texts: Sequence[str], source_language: str = "hi",
                  sentence: bool = True) -> Dict[str, List[str]]:
    """
    Indic script -> Latin, as {input: [candidate, ...]}.

    `sentence=True` keeps a multi-word name whole, which is what a place name
    needs. `sentence=False` returns per-word alternatives and is only useful
    as a fallback on a single word - see the module docstring.

    Never raises for network reasons. An input that could not be transliterated
    is simply absent from the result, so a caller's `.get(name, [])` degrades
    to "no candidate" rather than to a wrong one.
    """
    global _last_error
    wanted = [t for t in dict.fromkeys(texts) if t and t.strip()]
    if not wanted:
        return {}

    cache = _load_cache()
    out: Dict[str, List[str]] = {}
    missing: List[str] = []
    for text in wanted:
        hit = cache.get(_cache_key(text, sentence))
        if hit is not None:
            out[text] = list(hit)
        else:
            missing.append(text)

    if not missing or not AVAILABLE:
        return out

    global _cache_dirty
    try:
        config = _pipeline_config(source_language)
    except Exception as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        return out

    for start in range(0, len(missing), MAX_BATCH):
        batch = missing[start:start + MAX_BATCH]
        body = {
            "pipelineTasks": [{
                "taskType": "transliteration",
                "config": {
                    "language": {"sourceLanguage": source_language,
                                 "targetLanguage": "en"},
                    "serviceId": config["service_id"],
                    "numSuggestions": NUM_SUGGESTIONS,
                    "isSentence": bool(sentence),
                },
            }],
            "inputData": {"input": [{"source": t} for t in batch]},
        }
        try:
            payload = _post(config["url"], body,
                            {config["header_name"]: config["header_value"]})
        except Exception as exc:
            _last_error = f"{type(exc).__name__}: {exc}"
            break

        responses = payload.get("pipelineResponse") or []
        results = (responses[0].get("output") or []) if responses else []
        for item in results:
            source = item.get("source")
            target = item.get("target")
            if not source:
                continue
            candidates = ([target] if isinstance(target, str)
                          else [c for c in (target or []) if isinstance(c, str)])
            # De-duplicate but keep the service's ordering: its first
            # suggestion is its best guess even though it is not always right.
            candidates = list(dict.fromkeys(c.strip() for c in candidates if c.strip()))
            if candidates:
                out[source] = candidates
                cache[_cache_key(source, sentence)] = candidates
                _cache_dirty = True

    flush_cache()
    return out


def candidates(name: str, source_language: str = "hi") -> List[str]:
    """
    Every Latin spelling worth trying for one Indic-script place name, best
    first: the established exonym, then the whole-name transliterations, then
    per-word alternatives.

    The caller matches these against its own reference data. This module
    deliberately does not decide which candidate is right - the admin master
    is the authority on that, not a transliteration model.
    """
    name = (name or "").strip()
    if not name:
        return []

    ordered: List[str] = []
    established = exonym(name)
    if established:
        ordered.append(established)

    ordered.extend(transliterate([name], source_language, sentence=True).get(name, []))
    if " " not in name:
        ordered.extend(transliterate([name], source_language, sentence=False).get(name, []))

    return list(dict.fromkeys(c for c in ordered if c))


INDIC_RANGES = (
    (0x0900, 0x097F),   # Devanagari
    (0x0980, 0x09FF),   # Bengali / Assamese
    (0x0A00, 0x0A7F),   # Gurmukhi
    (0x0A80, 0x0AFF),   # Gujarati
    (0x0B00, 0x0B7F),   # Odia
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0C80, 0x0CFF),   # Kannada
    (0x0D00, 0x0D7F),   # Malayalam
)

# Which Bhashini source language to ask for, per script block. Transliteration
# is script-directed, so sending Tamil text as 'hi' returns nothing useful.
SCRIPT_LANGUAGE = {
    0x0900: "hi", 0x0980: "bn", 0x0A00: "pa", 0x0A80: "gu", 0x0B00: "or",
    0x0B80: "ta", 0x0C00: "te", 0x0C80: "kn", 0x0D00: "ml",
}


def script_language(text: str) -> Optional[str]:
    """
    The Bhashini language code for the first Indic character in `text`, or None
    when the text is already Latin and needs no transliteration at all.
    """
    for ch in text or "":
        code = ord(ch)
        for low, high in INDIC_RANGES:
            if low <= code <= high:
                return SCRIPT_LANGUAGE.get(low)
    return None


def is_indic(text: str) -> bool:
    return script_language(text) is not None


def resolve_to_reference(names: Sequence[str], accepts) -> Dict[str, str]:
    """
    Map Indic-script place names onto the spelling the reference data uses.

    `accepts(candidate) -> bool` is the authority - the administrative master,
    injected by the caller the same way learning.apply_model() takes its
    `corroborate`. This module never decides that a transliteration is correct;
    it only proposes, and something that actually knows the district list
    decides. Names the authority rejects are simply absent from the result, so
    an unrecognised village is left exactly as the OCR read it rather than
    being replaced by a plausible guess.

    Latin input is skipped without a network call, which is most of the corpus.
    """
    out: Dict[str, str] = {}
    wanted = [n for n in dict.fromkeys(names) if n and is_indic(n)]
    if not wanted:
        return out

    # Warm the cache for the whole batch in one call per language, rather than
    # one call per village during a bulk ingest.
    by_language: Dict[str, List[str]] = {}
    for name in wanted:
        by_language.setdefault(script_language(name) or "hi", []).append(name)
    for language, group in by_language.items():
        uncached = [n for n in group if not exonym(n)]
        if uncached:
            transliterate(uncached, language, sentence=True)

    for name in wanted:
        for candidate in candidates(name, script_language(name) or "hi"):
            try:
                if accepts(candidate):
                    out[name] = candidate
                    break
            except Exception:
                continue
    return out


def capabilities() -> dict:
    """Honest status, for run.py --check and the dashboard."""
    return {
        "available": AVAILABLE,
        "reason": unavailable_reason(),
        "project_configured": bool(PROJECT_ID and UDYAT_KEY),
        "consent": CONSENT_GIVEN,
        "exonyms": len(_load_exonyms()),
        "cached_names": len(_load_cache()),
        "last_error": _last_error,
        "pipeline_id": PIPELINE_ID,
    }
