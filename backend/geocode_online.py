"""
Online geocoding for the place-name tier.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHY THIS EXISTS

The bundled coordinate table covers 36 states and does not reach most
districts. Measured on the real corpus:

    Amari, Jaunpur, Uttar Pradesh   -> fell back to the STATE   300,000 m
    Narharpur, Lucknow              -> fell back to the district 40,000 m

A 300 km answer is not a geotag. The table is a demo extract and no amount
of matching logic fixes missing coordinates, so this asks a real gazetteer
instead and keeps the table as the fallback.

WHAT IT DOES AND DOES NOT IMPROVE

It moves the PLACE-NAME tier from hundreds of kilometres to roughly one, by
finding the village's own centre rather than its district's. It does NOT
produce parcel-grade accuracy and must never be read as doing so: no
geocoder can say where khasra 89 lies. That still requires the parcel
geometry - the state's cadastral vector layer, or ground control points -
and the two tiers stay separate for exactly that reason.

PRIVACY, HONESTLY

This sends the record's village, tehsil and district to a third party. That
is administrative geography rather than the owner's identity, so it is
gated on a single flag rather than the two-flag consent llm_extractor
requires for document text - but it is still the record's location leaving
the machine, and it is off unless asked for.

OFF BY DEFAULT, AND NEVER WORSE THAN BEFORE. Every failure path - flag
unset, no network, no match, a malformed reply - returns None, and the
caller falls back to the offline table. Enabling this can raise accuracy;
it cannot lower it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_HERE, "..", "storage", "geocode_cache.json")

ENABLED = os.environ.get("ONLINE_GEOCODING") == "1"

# Nominatim is the default because it needs no key and answers immediately.
# The endpoint is configurable so an installation can point at Bhuvan (ISRO)
# or a departmental gazetteer instead - the response shape is what this
# module parses, so a swap needs a matching adapter, but the URL, the contact
# header and the rate limit are all deployment decisions rather than code.
ENDPOINT = os.environ.get("GEOCODER_URL", "https://nominatim.openstreetmap.org/search")

# Nominatim's usage policy requires an identifying User-Agent and no more
# than one request a second. Both are honoured here rather than left to the
# operator, because a project that ignores a free service's terms should not
# be recommending that service.
USER_AGENT = os.environ.get(
    "GEOCODER_USER_AGENT",
    "SIH26018-LandRecordDigitization/1.0 (government land records prototype)")
MIN_REQUEST_INTERVAL_S = float(os.environ.get("GEOCODER_MIN_INTERVAL", "1.1"))
REQUEST_TIMEOUT_S = int(os.environ.get("GEOCODER_TIMEOUT", "15"))

# Accuracy claimed per result granularity. These are deliberately pessimistic:
# a village centroid is not the village boundary, and overstating precision
# on a land record is the failure this project exists to avoid.
ACCURACY_BY_LEVEL = {
    "locality": 2000,      # a named village or town centre
    "district": 40000,
    "state": 300000,
}

# Nominatim "type" values that mean we actually landed on a settlement rather
# than on the administrative unit containing it.
_LOCALITY_TYPES = {
    "village", "hamlet", "town", "city", "suburb", "neighbourhood",
    "locality", "isolated_dwelling",
}

_cache: Optional[Dict[str, dict]] = None
_cache_dirty = False
_last_request_at = 0.0


def available() -> bool:
    return ENABLED


def status() -> dict:
    """Honest capability line for run.py --check."""
    if not ENABLED:
        return {"available": False,
                "reason": ("ONLINE_GEOCODING=1 not set - village coordinates "
                           "come from the bundled table, which reaches "
                           "district level at best")}
    return {"available": True, "endpoint": ENDPOINT,
            "cached": len(_load_cache())}


def _load_cache() -> Dict[str, dict]:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
                _cache = json.load(fh)
        except Exception:
            _cache = {}
    return _cache


def save_cache() -> None:
    """Persist looked-up places so a rerun makes no network calls at all."""
    global _cache_dirty
    if not _cache_dirty or _cache is None:
        return
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh, ensure_ascii=False, indent=1, sort_keys=True)
        _cache_dirty = False
    except Exception:
        pass


def _key(village, tehsil, district, state) -> str:
    return "|".join((village or "", tehsil or "", district or "", state or "")).lower()


def _throttle() -> None:
    global _last_request_at
    wait = MIN_REQUEST_INTERVAL_S - (time.time() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


def _query(terms: str) -> Optional[list]:
    url = ENDPOINT + "?" + urllib.parse.urlencode({
        "q": terms, "format": "jsonv2", "limit": "1",
        "countrycodes": "in",          # a land record is in India by definition
        "addressdetails": "1",
    })
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    _throttle()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def lookup(village: Optional[str], tehsil: Optional[str],
           district: Optional[str], state: Optional[str]) -> Optional[dict]:
    """
    Coordinates for the finest place this record names, or None.

    Tries the most specific combination first and widens, so a village that
    the gazetteer does not know still yields its district rather than
    nothing - the same laddering the offline table does, against far better
    data.

    Returns {"lat", "lon", "accuracy_m", "level", "matched", "source"}.
    """
    if not ENABLED:
        return None

    # A QUERY MUST BE ANCHORED BY A DISTRICT OR A STATE. A bare village name
    # is never sent.
    #
    # This is not a tuning preference, it is the module's refusal property.
    # geocode.py returns nothing for "Atlantis", "Springfield", "Zzzyx" and
    # for pure address furniture like "TEHSIL DISTRICT", and its own tests
    # say why: "a wrong pin is worse than a blank, so these matter more than
    # the positive cases." A general gazetteer will cheerfully find SOMETHING
    # for any of those, so querying on an unanchored name traded a blank for
    # a wrong pin - exactly the trade this system exists not to make.
    #
    # Anchored, the same lookup is sound: "Amari, Jaunpur, Uttar Pradesh"
    # scopes the search to a real administrative unit, and a stray name
    # inside a real district resolves no further than that district.
    if not (district or state):
        return None
    if not any((village, district, state)):
        return None

    cache = _load_cache()
    key = _key(village, tehsil, district, state)
    if key in cache:
        return cache[key] or None       # a cached miss is still a miss

    attempts = []
    if village and district:
        attempts.append(([village, tehsil, district, state], "locality"))
    if district:
        attempts.append(([district, state], "district"))
    if state:
        attempts.append(([state], "state"))

    global _cache_dirty
    for parts, level in attempts:
        terms = ", ".join([p for p in parts if p] + ["India"])
        rows = _query(terms)
        if not rows:
            continue
        row = rows[0]
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        # Trust the level the RESULT reports, not the level asked for. A
        # village query that Nominatim answers with an administrative
        # boundary has not found the village, and claiming 2 km for it would
        # be inventing precision.
        kind = str(row.get("type") or "").lower()
        actual = level
        if level == "locality" and kind not in _LOCALITY_TYPES:
            actual = "district" if district else "state"

        result = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "accuracy_m": ACCURACY_BY_LEVEL.get(actual, 300000),
            "level": actual,
            "matched": row.get("display_name", "")[:160],
            "source": "online",
        }
        cache[key] = result
        _cache_dirty = True
        save_cache()
        return result

    cache[key] = None                   # remember the miss; do not re-ask
    _cache_dirty = True
    save_cache()
    return None
