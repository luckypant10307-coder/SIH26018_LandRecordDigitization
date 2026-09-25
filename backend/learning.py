"""
AI-driven learning loop.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The problem statement asks for an "AI-driven learning mechanism that improves
extraction accuracy over time". This module implements that loop in the form
that is actually defensible for a government deployment: supervised learning
from verifier corrections, applied as transparent, inspectable adjustments
rather than an opaque model update.

Three learned artefacts are derived from the corrections table:

  1. CHARACTER CONFUSION MAP
     Aligns the machine value against the human-corrected value and counts
     which characters were misread. Frequent confusions become candidate
     auto-corrections for numeric fields.

  2. VALUE ALIASES
     When the same wrong string is corrected to the same right string enough
     times (e.g. 'Lucknov' -> 'Lucknow'), that mapping is promoted to an alias
     and applied on ingestion.

  3. CONFIDENCE RECALIBRATION
     Compares stated confidence against observed correctness per field, and
     produces a per-field multiplier. A field that is wrong more often than its
     confidence claims gets deflated, which pushes it into the review queue
     earlier. This is the single highest-value part of the loop: it makes the
     system's self-doubt accurate.

Everything here is derived on demand from the audit-backed corrections table,
so the learning is fully explainable: for every applied correction we can name
the verifier decisions that produced it.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Tuple

from field_extractor import FIELD_BY_KEY, REVIEW_THRESHOLD, normalise

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "storage", "learned_model.json")

# Promotion thresholds. Deliberately conservative: an auto-correction that
# fires wrongly in land records is worse than one that never fires.
MIN_ALIAS_SUPPORT = 3          # identical correction seen this many times
MIN_CONFUSION_SUPPORT = 4      # character confusion seen this many times
MIN_CALIBRATION_SAMPLE = 8     # reviewed fields needed before recalibrating


# --------------------------------------------------------------------------
# 1. Character confusion mining
# --------------------------------------------------------------------------

def _align_confusions(wrong: str, right: str) -> List[Tuple[str, str]]:
    """
    Character-level substitutions that turn `wrong` into `right`.
    Only 1:1 replacements are recorded; insertions and deletions are ignored
    because they are usually segmentation faults, not confusions.
    """
    pairs: List[Tuple[str, str]] = []
    matcher = SequenceMatcher(None, wrong, right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        if (i2 - i1) != (j2 - j1):
            continue
        for a, b in zip(wrong[i1:i2], right[j1:j2]):
            if a != b:
                pairs.append((a, b))
    return pairs


def mine_confusions(corrections: List[dict]) -> List[dict]:
    counter: Counter = Counter()
    contexts: Dict[Tuple[str, str], set] = defaultdict(set)

    for row in corrections:
        ai, human = row.get("ai_value"), row.get("human_value")
        if not ai or not human or ai == human:
            continue
        spec = FIELD_BY_KEY.get(row.get("field_key", ""))
        for a, b in _align_confusions(str(ai), str(human)):
            counter[(a, b)] += 1
            if spec:
                contexts[(a, b)].add(spec.kind)

    out = []
    for (a, b), count in counter.most_common():
        if count < MIN_CONFUSION_SUPPORT:
            continue
        kinds = sorted(contexts[(a, b)])
        out.append({
            "from": a, "to": b, "support": count, "field_kinds": kinds,
            # Eligible to be *offered* as a repair inside identifier fields,
            # where the alphabet is closed. The gate asks whether this
            # confusion was ever seen in such a field - NOT whether it was
            # only ever seen there. Requiring `all(kinds) == number` was too
            # strict to ever fire: on the sample corpus '4' -> '1' reached 14
            # supporting corrections but was disqualified because the same
            # misread also showed up once in a village name, as though seeing
            # a confusion in prose made it untrue of digits.
            #
            # Being eligible is not being trusted. apply_model() still only
            # touches fields of kind 'number', only substitutes one character
            # at a time, only accepts a candidate the external registry or a
            # cadastral map can vouch for, and always sends the result to a
            # human. This flag opens that path; it does not shortcut it.
            "auto_apply": ("number" in kinds) and b.isdigit(),
        })
    return out


# --------------------------------------------------------------------------
# 2. Value alias mining
# --------------------------------------------------------------------------

def _alias_key(field_key: str, value: str) -> str:
    return f"{field_key}::{re.sub(r'[^a-z0-9\u0900-\u097f]', '', normalise(value).lower())}"


def mine_aliases(corrections: List[dict]) -> List[dict]:
    votes: Dict[str, Counter] = defaultdict(Counter)
    display: Dict[str, Tuple[str, str]] = {}

    for row in corrections:
        ai, human = row.get("ai_value"), row.get("human_value")
        field_key = row.get("field_key")
        if not ai or not human or ai == human or not field_key:
            continue
        # Aliases only make sense for closed-vocabulary text fields.
        spec = FIELD_BY_KEY.get(field_key)
        if not spec or spec.kind not in ("text", "class"):
            continue
        key = _alias_key(field_key, str(ai))
        votes[key][str(human)] += 1
        display[key] = (field_key, str(ai))

    out = []
    for key, counter in votes.items():
        winner, support = counter.most_common(1)[0]
        if support < MIN_ALIAS_SUPPORT:
            continue
        field_key, wrong = display[key]
        total = sum(counter.values())
        out.append({
            "field_key": field_key,
            "wrong_value": wrong,
            "corrected_value": winner,
            "support": support,
            "agreement": round(support / total, 3),
            "auto_apply": support >= MIN_ALIAS_SUPPORT and (support / total) >= 0.8,
        })
    return sorted(out, key=lambda r: -r["support"])


# --------------------------------------------------------------------------
# 3. Confidence recalibration
# --------------------------------------------------------------------------

def calibrate(field_stats: List[dict]) -> List[dict]:
    """
    `field_stats` comes from Database.stats()['field_accuracy'] and carries
    reviewed / corrected counts plus the mean stated confidence.

    We compare stated confidence with observed precision. The multiplier moves
    stated confidence toward reality, damped so a small sample cannot swing it.
    """
    out = []
    for row in field_stats:
        reviewed = row.get("reviewed") or 0
        if reviewed < MIN_CALIBRATION_SAMPLE:
            continue
        observed = row.get("precision")
        stated = row.get("avg_conf")
        if observed is None or not stated:
            continue

        raw_ratio = observed / stated if stated > 0 else 1.0
        # Damping: trust the observation proportionally to sample size,
        # saturating around 40 reviews.
        weight = min(1.0, reviewed / 40.0)
        multiplier = 1.0 + (raw_ratio - 1.0) * weight
        multiplier = max(0.5, min(1.15, multiplier))

        out.append({
            "field_key": row["field_key"],
            "display": row.get("display") or row["field_key"],
            "reviewed": reviewed,
            "stated_confidence": round(stated, 4),
            "observed_precision": round(observed, 4),
            "multiplier": round(multiplier, 4),
            "direction": "deflate" if multiplier < 0.98 else
                         ("inflate" if multiplier > 1.02 else "stable"),
        })
    return sorted(out, key=lambda r: r["multiplier"])


# --------------------------------------------------------------------------
# Model assembly / application
# --------------------------------------------------------------------------

def build_model(corrections: List[dict], field_stats: List[dict]) -> dict:
    confusions = mine_confusions(corrections)
    aliases = mine_aliases(corrections)
    calibration = calibrate(field_stats)
    return {
        "version": 1,
        "samples": len(corrections),
        "confusions": confusions,
        "aliases": aliases,
        "calibration": calibration,
        "active_rules": (sum(1 for c in confusions if c["auto_apply"])
                         + sum(1 for a in aliases if a["auto_apply"])
                         + len(calibration)),
    }


def save_model(model: dict, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=2)


def load_model(path: str = MODEL_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"version": 0, "samples": 0, "confusions": [], "aliases": [],
                "calibration": [], "active_rules": 0}


def _confusion_variants(value: str, confusions: List[dict]) -> Dict[str, dict]:
    """
    Every single-character variant of `value` reachable by one learned
    confusion.

    One substitution at a time, deliberately. A learned confusion says "the
    engine sometimes reads 4 where the truth is 1"; it does NOT say which 4 on
    the page was the wrong one. Replacing them all turns khata 4474 into 1171
    and invents a parcel that does not exist. Generating one candidate per
    position and making something else choose between them is the only safe
    reading of the evidence.
    """
    out: Dict[str, dict] = {}
    for rule in confusions:
        if not rule.get("auto_apply"):
            continue
        a, b = rule.get("from"), rule.get("to")
        if not a or not b:
            continue
        for i, ch in enumerate(value):
            if ch == a:
                out.setdefault(value[:i] + b + value[i + 1:], rule)
    return out


def apply_model(fields: List, model: Optional[dict] = None,
                corroborate: Optional[Callable[[str, str], bool]] = None,
                review_threshold: float = REVIEW_THRESHOLD) -> List[dict]:
    """
    Apply learned aliases, confusion repairs and recalibration to freshly
    extracted fields. Mutates value/confidence/status in place and returns a
    list of the adjustments made, so the UI can show exactly what the learning
    loop did.

    `corroborate(field_key, value) -> bool` is the authority that decides
    whether a candidate plot number actually exists - the external registry and
    the cadastral parcel index, injected by the caller so this module stays
    free of those dependencies. Without it, confusion repair is skipped
    entirely rather than guessed at.
    """
    model = model or load_model()
    if not model.get("active_rules"):
        return []

    alias_index = {
        _alias_key(a["field_key"], a["wrong_value"]): a
        for a in model.get("aliases", []) if a.get("auto_apply")
    }
    calib_index = {c["field_key"]: c for c in model.get("calibration", [])}
    confusions = model.get("confusions", [])
    applied: List[dict] = []

    for f in fields:
        if getattr(f, "value", None):
            alias = alias_index.get(_alias_key(f.key, f.value))
            if alias and alias["corrected_value"] != f.value:
                old = f.value
                f.value = alias["corrected_value"]
                f.notes.append(
                    f"Learned correction applied: '{old}' -> '{f.value}' "
                    f"(from {alias['support']} verifier corrections).")
                applied.append({"field_key": f.key, "type": "alias",
                                "from": old, "to": f.value,
                                "support": alias["support"]})

        # Confusion repair. Only for identifiers, where the alphabet is closed
        # and an authority exists to check a candidate against; and only when
        # the extracted value is NOT already corroborated, so a plot number
        # that checks out is never second-guessed.
        spec = FIELD_BY_KEY.get(f.key)
        if (corroborate is not None and confusions and getattr(f, "value", None)
                and spec is not None and spec.kind == "number"
                and not corroborate(f.key, f.value)):
            variants = _confusion_variants(f.value, confusions)
            hits = [v for v in variants if corroborate(f.key, v)]
            if len(hits) == 1:
                old, new = f.value, hits[0]
                rule = variants[new]
                f.value = new
                # Repaired, never silently trusted: a learned guess about a
                # plot number goes in front of a human even though it now
                # matches the registry.
                f.status = "needs_review"
                f.notes.append(
                    f"Learned OCR repair: '{old}' -> '{new}' "
                    f"('{rule['from']}' misread as '{rule['to']}' in "
                    f"{rule['support']} past corrections). '{old}' matches no "
                    f"registry or cadastral record; '{new}' does. Confirm "
                    f"against the source document.")
                applied.append({"field_key": f.key, "type": "confusion_repair",
                                "from": old, "to": new,
                                "support": rule["support"]})
            elif len(hits) > 1:
                # Several learned repairs are equally plausible. Picking one
                # would be a coin flip on someone's land, so the value stands
                # and the doubt is recorded instead.
                f.status = "needs_review"
                f.notes.append(
                    f"'{f.value}' matches no registry or cadastral record, and "
                    f"learned OCR confusions make {', '.join(sorted(hits))} "
                    f"equally plausible. Needs a human decision.")
                applied.append({"field_key": f.key, "type": "confusion_ambiguous",
                                "from": f.value, "candidates": sorted(hits)})

        calib = calib_index.get(f.key)
        if calib and f.confidence > 0:
            before = f.confidence
            f.confidence = max(0.0, min(1.0, f.confidence * calib["multiplier"]))
            if abs(f.confidence - before) >= 0.02:
                f.notes.append(
                    f"Confidence recalibrated {before:.2f} -> {f.confidence:.2f} "
                    f"using {calib['reviewed']} past reviews of this field.")
                applied.append({"field_key": f.key, "type": "calibration",
                                "from": round(before, 4),
                                "to": round(f.confidence, 4),
                                "multiplier": calib["multiplier"]})

        # Recalibration is only worth doing if it changes where the field
        # GOES. extract_fields() stamped status from the raw confidence before
        # any of the above ran, so without this the deflation was cosmetic:
        # a field the loop had learned to distrust still sailed past review
        # with a lower number printed next to it.
        if getattr(f, "status", "") not in ("missing", "needs_review"):
            f.status = ("needs_review" if f.confidence < review_threshold
                        else "extracted")

    return applied


def retrain(db) -> dict:
    """
    Rebuild and persist the learned model from everything the verifiers have
    corrected so far. Cheap enough to run after every batch.
    """
    corrections = db.learning_signals(limit=100000)
    stats = db.stats()
    model = build_model(corrections, stats.get("field_accuracy", []))
    save_model(model)
    return model
