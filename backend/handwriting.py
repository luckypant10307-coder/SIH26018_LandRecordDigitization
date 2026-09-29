"""
Handwriting detection for scanned land records.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHY THIS EXISTS, AND WHY IT IS DETECTION RATHER THAN RECOGNITION
----------------------------------------------------------------
Patwari mutation entries, marginal notes and endorsements are handwritten.
Tesseract's LSTM is trained on PRINT, so on a handwritten line it does not
fail cleanly - it returns something that looks like text, with a confidence
that looks ordinary. Field extraction then treats that output like any other,
and a fabricated owner name can be auto-approved onto a land record. That is
the worst failure this system can produce, and it is currently invisible.

Detecting the handwriting fixes the invisibility. A line identified as
not-print gets its confidence suppressed and is routed to a human instead of
being trusted, which is a correct and useful outcome even with no recogniser
attached.

WHY NOT TrOCR / Donut / PaddleOCR-HTR
-------------------------------------
Two separate blockers, and the second is the one that matters:

  1. PyTorch cannot load on this platform - Windows Application Control
     blocks torch/lib/shm.dll (WinError 4551) - and merely INSTALLING it
     broke spaCy, because thinc detects torch and imports it eagerly.
     PaddleOCR was already ruled out (no Python 3.14 build; ~20 min/page on
     CPU). onnxruntime DOES load here, so the framework blocker is
     surmountable via an exported .onnx model.
  2. The pretrained weights are the real problem. TrOCR and Donut are trained
     on LATIN handwriting (IAM/RIMES). They cannot read Devanagari, which is
     what a Patwari actually wrote. Indic HTR needs fine-tuning on an Indic
     handwriting corpus (IIIT-HW-Dev and similar) - a training job with data
     this project does not have, not an integration.

So the honest architecture is: detect locally and refuse to guess, expose a
hook where a real Indic HTR model can be dropped in once trained (ONNX, since
that runs here), and optionally transcribe flagged crops with a consented
cloud vision model - never trusting either without review.

WHAT IS AND IS NOT VERIFIED
---------------------------
This is NOVELTY detection, not a trained two-class classifier, because the
corpus contains a great deal of print and no handwriting at all. Print is
therefore characterised precisely and anything far from that profile is
flagged as not-print.

The consequence is asymmetric and must be stated plainly:

  * The FALSE-POSITIVE rate is measurable and measured - how often a printed
    line is wrongly flagged - because printed lines are abundant. See
    tools/fit_print_profile.py, which reports it.
  * The TRUE-POSITIVE rate is NOT verified. Whether this catches real Patwari
    handwriting cannot be established without real Patwari handwriting. The
    features are chosen from the handwriting-analysis literature and are
    individually explainable, but "flags print correctly" is the only claim
    this module has earned.

Do not present the second as measured. It is a designed capability awaiting
a real corpus.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence

try:
    import cv2 as _cv2
except Exception:                                    # pragma: no cover
    _cv2 = None
try:
    import numpy as _np
except Exception:                                    # pragma: no cover
    _np = None

CV_AVAILABLE = _cv2 is not None and _np is not None

_HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_PATH = os.path.join(_HERE, "..", "storage", "print_profile.json")

# The features, each one a property that separates a machine-set line from a
# hand-written one for a reason that can be stated in a sentence.
FEATURE_NAMES = (
    # A printing press or laser printer lays down one stroke weight. A nib or
    # ballpoint varies with pressure and direction.
    "stroke_width_cv",
    # Printed glyphs sit on an exact baseline. Handwriting drifts off it.
    "baseline_residual",
    # A typeface has a fixed x-height and cap-height; handwritten characters
    # vary in size across a single line.
    "height_cv",
    # Handwritten spacing is irregular where typeset spacing is metric.
    "gap_cv",
)

# A line must be at least this tall for the features to mean anything; below
# it the connected components are too few and too coarse to measure.
MIN_LINE_HEIGHT = 12
MIN_COMPONENTS = 4

# How many standard deviations from the print profile before a line is called
# not-print. Set by measurement, not taste: tools/fit_print_profile.py reports
# the false-positive rate on held-out PRINTED lines at this value, and it is
# chosen to keep that rate low, because a false handwriting flag sends a
# perfectly good printed field to a human for no reason.
DEFAULT_Z_THRESHOLD = 4.0


# Quantiles of the printed-text distribution stored in the profile. The
# threshold used at runtime is read from these, so it is expressed as "how
# often may this wrongly flag print" rather than as a number of sigmas.
PROFILE_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)

# Default operating point. Chosen from the measured sweep in
# tools/fit_print_profile.py, which reports the actual combined
# false-positive rate on held-out printed runs at each candidate.
DEFAULT_QUANTILE = 0.995

# Quantile thresholding needs enough printed samples for the tail to mean
# anything. With a few dozen runs the 99.5th percentile IS essentially the
# maximum, so any layout the profile did not happen to contain exceeds it and
# ordinary print gets flagged. Below this count the sigma rule is used
# instead - cruder, but it does not pretend to know a tail it has not seen.
MIN_QUANTILE_SAMPLES = 100


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def _binarise(gray):
    """Ink mask (255 = ink) for one line crop."""
    _, ink = _cv2.threshold(gray, 0, 255,
                            _cv2.THRESH_BINARY_INV + _cv2.THRESH_OTSU)
    return ink


def _cv(values: Sequence[float]) -> float:
    """Coefficient of variation - scale-free, so DPI does not change it."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def _components(ink, crop_height: float, crop_width: float) -> List[tuple]:
    """
    Character-sized connected components as (x, y, w, h, centre_x).

    Specks are dropped as noise, and anything spanning almost the whole crop
    is dropped as a rule line or a cell border - a land-record form is made
    of those, and counting one as a character wrecks the height statistics of
    the row it rules.
    """
    count, _labels, stats, centroids = _cv2.connectedComponentsWithStats(ink, 8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 8 or h < crop_height * 0.15 or w > crop_width * 0.9:
            continue
        boxes.append((float(x), float(y), float(w), float(h), float(centroids[i][0])))
    boxes.sort(key=lambda b: b[0])
    return boxes


def split_segments(boxes: List[tuple], gap_multiple: float = 3.0) -> List[List[tuple]]:
    """
    Break a line where it visibly breaks: at gaps far wider than its own
    typical inter-character spacing.

    THIS IS WHAT MAKES MIXED RECORDS WORK, and it fixes a real
    false positive. An old khatauni row is a printed label, a wide blank, and
    then the Patwari's handwritten entry. Measured across the whole row, two
    things go wrong: the printed label's perfect regularity dilutes the
    handwriting's irregularity, and the label-to-value gap by itself makes
    the spacing look wildly irregular - an ENTIRELY PRINTED form row scored
    9 to 15 standard deviations from print on gap variation alone, purely
    because of its layout.

    Splitting at that gap fixes both. Spacing is then measured only WITHIN a
    run of text, where it is genuinely uniform for print, and the printed
    label and the handwritten value are judged separately instead of being
    averaged into one verdict that describes neither.
    """
    if len(boxes) < 2:
        return [boxes] if boxes else []
    gaps = []
    for prev, nxt in zip(boxes, boxes[1:]):
        gaps.append(max(0.0, nxt[0] - (prev[0] + prev[2])))
    positive = [g for g in gaps if g > 0]
    if not positive:
        return [boxes]
    median_gap = float(_np.median(positive))
    # An absolute floor as well as a relative one: on a tightly-set line the
    # median gap can be a pixel or two, and a purely relative rule would then
    # split at every ordinary word space.
    threshold = max(median_gap * gap_multiple, _np.median([b[3] for b in boxes]) * 0.8)

    segments: List[List[tuple]] = [[boxes[0]]]
    for gap, box in zip(gaps, boxes[1:]):
        if gap > threshold:
            segments.append([box])
        else:
            segments[-1].append(box)
    return segments


def _features_from(gray, ink, boxes: List[tuple]) -> Optional[Dict[str, float]]:
    """The four features over one run of components."""
    if len(boxes) < MIN_COMPONENTS:
        return None
    height = float(gray.shape[0])

    # Stroke width, measured only under these components.
    x0 = int(max(0, min(b[0] for b in boxes) - 1))
    x1 = int(min(gray.shape[1], max(b[0] + b[2] for b in boxes) + 1))
    if x1 - x0 < 4:
        return None
    distance = _cv2.distanceTransform(ink[:, x0:x1], _cv2.DIST_L2, 5)
    ridge = distance[distance > 0]
    if ridge.size < 10:
        return None
    cutoff = float(_np.percentile(ridge, 70.0))
    spine = ridge[ridge >= cutoff]
    mean_spine = float(_np.mean(spine)) if spine.size else 0.0
    stroke_width_cv = float(_np.std(spine) / mean_spine) if mean_spine > 0 else 0.0

    xs = _np.array([b[4] for b in boxes], dtype=float)
    bottoms = _np.array([b[1] + b[3] for b in boxes], dtype=float)
    if xs.size >= 3 and float(_np.ptp(xs)) > 1e-6:
        slope, intercept = _np.polyfit(xs, bottoms, 1)
        residual = bottoms - (slope * xs + intercept)
    else:
        residual = bottoms - float(_np.mean(bottoms))
    baseline_residual = float(_np.std(residual) / height)

    gaps = []
    for prev, nxt in zip(boxes, boxes[1:]):
        gaps.append(max(0.0, nxt[0] - (prev[0] + prev[2])))

    return {
        "stroke_width_cv": stroke_width_cv,
        "baseline_residual": baseline_residual,
        "height_cv": _cv([b[3] for b in boxes]),
        "gap_cv": _cv(gaps) if len(gaps) >= 2 else 0.0,
        "components": float(len(boxes)),
    }


def line_features(gray) -> Optional[Dict[str, float]]:
    """
    The four features over a whole line crop.

    Kept for the case where a line really is one run of text. Scoring a FORM
    ROW with this is a mistake - use segment_features, which is what
    inspect_line does.
    """
    if not CV_AVAILABLE or gray is None:
        return None
    if gray.ndim != 2 or gray.shape[0] < MIN_LINE_HEIGHT or gray.shape[1] < MIN_LINE_HEIGHT:
        return None
    ink = _binarise(gray)
    if int((ink > 0).sum()) < 20:
        return None
    boxes = _components(ink, float(gray.shape[0]), float(gray.shape[1]))
    return _features_from(gray, ink, boxes)


def segment_features(gray) -> List[Dict[str, float]]:
    """
    One feature set per run of text on the line, left to right.

    Each carries `x0`/`x1` so a caller can say WHICH part of the row looked
    handwritten - which is what lets a printed label and a pen-filled value
    on the same row be reported separately.
    """
    if not CV_AVAILABLE or gray is None:
        return []
    if gray.ndim != 2 or gray.shape[0] < MIN_LINE_HEIGHT or gray.shape[1] < MIN_LINE_HEIGHT:
        return []
    ink = _binarise(gray)
    if int((ink > 0).sum()) < 20:
        return []
    boxes = _components(ink, float(gray.shape[0]), float(gray.shape[1]))
    out: List[Dict[str, float]] = []
    for segment in split_segments(boxes):
        features = _features_from(gray, ink, segment)
        if features is None:
            continue
        features["x0"] = float(min(b[0] for b in segment))
        features["x1"] = float(max(b[0] + b[2] for b in segment))
        out.append(features)
    return out


# --------------------------------------------------------------------------
# Print profile
# --------------------------------------------------------------------------

def fit_profile(samples: List[Dict[str, float]]) -> dict:
    """
    Build the print profile: mean and standard deviation of each feature over
    known-printed lines.

    A minimum standard deviation is imposed per feature. Without it a feature
    that happens to be nearly constant across the training corpus produces a
    near-zero denominator, and then any line at all scores an enormous
    z-score - the profile would flag everything, most confidently on the
    cleanest pages.
    """
    profile: Dict[str, Dict[str, float]] = {}
    for name in FEATURE_NAMES:
        values = sorted(s[name] for s in samples if name in s)
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        std = math.sqrt(variance)
        profile[name] = {
            "mean": mean,
            # Floor at 5% of the mean, or a small absolute value for features
            # whose mean is itself near zero.
            "std": max(std, abs(mean) * 0.05, 1e-3),
            "observed_std": std,
            # The EMPIRICAL distribution of this feature over printed text.
            # This, not mean+k*std, is what the threshold is read from.
            #
            # Mean and standard deviation assume a roughly Gaussian spread,
            # and these features are not remotely Gaussian on real scans:
            # fitted on the degraded corpus, baseline_residual came out with a
            # standard deviation LARGER than its mean, so a 4-sigma bar landed
            # at 0.41 while handwriting measures about 0.065. The detector was
            # inert - it flagged nothing at all, and the 1.4% false-positive
            # rate looked excellent for exactly that reason.
            #
            # A quantile makes no distributional assumption and states the
            # false-positive rate directly: a threshold at the 0.995 quantile
            # of print mis-flags 0.5% of printed runs on this feature, by
            # construction.
            "quantiles": {str(q): _quantile(values, q) for q in PROFILE_QUANTILES},
        }
    return {"version": 1, "lines": len(samples), "features": profile}


def save_profile(profile: dict, path: str = PROFILE_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)


def load_profile(path: str = PROFILE_PATH) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            profile = json.load(fh)
    except Exception:
        return None
    if not isinstance(profile, dict) or not profile.get("features"):
        return None
    return profile


_CACHED: Optional[dict] = None
_TRIED = False


def _profile(path: str = PROFILE_PATH) -> Optional[dict]:
    global _CACHED, _TRIED
    if not _TRIED:
        _CACHED = load_profile(path)
        _TRIED = True
    return _CACHED


def reset_cache() -> None:
    global _CACHED, _TRIED
    _CACHED, _TRIED = None, False


def available(path: str = PROFILE_PATH) -> bool:
    return CV_AVAILABLE and _profile(path) is not None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score(features: Dict[str, float], profile: Optional[dict] = None,
          quantile: float = DEFAULT_QUANTILE,
          z_threshold: Optional[float] = None) -> Optional[dict]:
    """
    How far this run of text sits outside the printed-text distribution.

    The verdict is decided by QUANTILE, not by sigmas: a feature counts as
    anomalous when it exceeds the given quantile of the printed distribution.
    That states the cost directly - at 0.995, half a percent of printed runs
    exceed it on that feature, by definition - and it makes no assumption
    about the shape of the distribution. These features are heavy-tailed on
    real scans, and a mean-plus-4-sigma rule put the bar so far out that the
    detector flagged nothing whatsoever.

    Only ONE-SIDED deviations count. Every feature measures IRREGULARITY, so
    a run more regular than print is not evidence of handwriting - it is
    evidence of very clean print. Scoring absolute deviation would flag the
    cleanest pages in the corpus.

    z-scores are still reported, because "3.4 standard deviations from print"
    is the phrasing a reviewer understands - but they no longer decide
    anything. Passing `z_threshold` restores the old sigma rule, which exists
    only so the historical behaviour remains testable.
    """
    profile = profile if profile is not None else _profile()
    if not profile or not features:
        return None
    stats = profile.get("features") or {}

    # Enough printed samples behind this profile to trust its tail?
    use_quantiles = int(profile.get("lines") or 0) >= MIN_QUANTILE_SAMPLES

    z_scores: Dict[str, float] = {}
    exceeded: Dict[str, float] = {}
    ratios: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name not in features or name not in stats:
            continue
        entry = stats[name]
        value = features[name]
        mean = entry.get("mean", 0.0)
        std = entry.get("std") or 1e-3
        z_scores[name] = (value - mean) / std

        if not use_quantiles:
            continue
        quantiles = entry.get("quantiles") or {}
        bar = quantiles.get(str(quantile))
        median = quantiles.get("0.5")
        if bar is None or median is None:
            continue
        # A feature on which printed text showed no spread at all carries no
        # information, and using it to flag is unjustified: the bar would sit
        # exactly on the constant, so any value a hair above it "exceeds" the
        # 99.5th percentile of print. Skip the feature rather than invent a
        # margin for it - a detector with three usable features is honest,
        # one with a fabricated fourth is not.
        if float(bar) <= float(median) * (1.0 + 1e-6) + 1e-9:
            continue
        bar = float(bar)
        ratios[name] = value / bar if bar > 0 else 0.0
        if value > bar:
            exceeded[name] = round(value / bar, 2)

    if not z_scores:
        return None

    # The feature that most exceeds its bar decides, falling back to the
    # largest z-score when the profile carries no quantiles (an older file).
    if ratios:
        worst_name = max(ratios, key=lambda k: ratios[k])
    else:
        worst_name = max(z_scores, key=lambda k: z_scores[k])

    # Decide. Quantiles when the profile carries them; the sigma rule when it
    # does not, so a profile written before quantiles existed still works
    # rather than silently flagging nothing - which is the worst possible
    # degradation for a detector, since it looks exactly like "all clear".
    if z_threshold is not None or not ratios:
        effective_z = z_threshold if z_threshold is not None else DEFAULT_Z_THRESHOLD
        worst_name = max(z_scores, key=lambda k: z_scores[k])
        suspected = z_scores[worst_name] >= effective_z
        decided_by = "sigma"
    else:
        suspected = bool(exceeded)
        decided_by = "quantile"

    return {
        "z_scores": {k: round(v, 2) for k, v in z_scores.items()},
        "worst_feature": worst_name,
        "worst_z": round(z_scores.get(worst_name, 0.0), 2),
        "quantile": quantile,
        "exceeded": exceeded,
        "worst_ratio": round(ratios.get(worst_name, 0.0), 2),
        "threshold": z_threshold if z_threshold is not None else quantile,
        "decided_by": decided_by,
        "is_handwriting_suspected": bool(suspected),
    }


_FEATURE_EXPLANATIONS = {
    "stroke_width_cv": "the stroke weight varies along the line, as a pen does "
                       "and a printer does not",
    "baseline_residual": "the characters do not sit on a straight baseline",
    "height_cv": "the characters vary in height across the line",
    "gap_cv": "the spacing between characters is irregular",
}


def explain(result: dict) -> str:
    """A sentence a reviewer can act on, naming the property that stood out."""
    if not result or not result.get("is_handwriting_suspected"):
        return ""
    name = result.get("worst_feature", "")
    why = _FEATURE_EXPLANATIONS.get(name, "it does not match printed text")
    where = ""
    if result.get("x0") is not None and result.get("segments", 1) > 1:
        where = (f" The suspect part runs from x={int(result['x0'])} to "
                 f"x={int(result['x1'])} of {result['segments']} text runs on "
                 f"this line, which on a printed form is normally the "
                 f"filled-in entry rather than the label.")
    return (f"This line may be handwritten: {why} "
            f"({name} is {result.get('worst_z')} standard deviations from "
            f"printed text)." + where + " Character recognition is trained on "
            f"print, so text read here is unreliable and has not been "
            f"trusted. Check it against the original document.")


def inspect_line(gray, profile: Optional[dict] = None,
                 quantile: float = DEFAULT_QUANTILE) -> Optional[dict]:
    """
    Judge one line crop, segment by segment, and report the worst part.

    Scores each run of text on the row SEPARATELY rather than the row as a
    whole. On an old land record a row is typically a printed label followed
    by a handwritten entry, and a single verdict for the row describes
    neither: the label's regularity hides the handwriting, while the blank
    between them makes even an all-printed row look irregular. Judging the
    parts separately reports the handwriting where it actually is - and the
    returned `x0`/`x1` say which part of the row that was, so it can be lined
    up with the value the extractor read.

    Returns None when nothing on the line could be judged, which is NOT the
    same as a verdict of "printed".
    """
    segments = segment_features(gray)
    if not segments:
        return None

    scored = []
    for features in segments:
        verdict = score(features, profile, quantile)
        if verdict is None:
            continue
        verdict["features"] = {k: round(v, 4) for k, v in features.items()}
        verdict["x0"] = features.get("x0")
        verdict["x1"] = features.get("x1")
        scored.append(verdict)
    if not scored:
        return None

    # The worst segment decides the line. One handwritten entry in a row of
    # printed labels has to flag the row, or the entry we actually extract is
    # the one part nobody was warned about.
    worst = max(scored, key=lambda v: (v["is_handwriting_suspected"],
                                       v.get("worst_ratio", 0.0)))
    worst["segments"] = len(scored)
    worst["flagged_segments"] = sum(1 for v in scored
                                    if v["is_handwriting_suspected"])
    return worst


def handwritten_segments(gray, profile: Optional[dict] = None,
                         quantile: float = DEFAULT_QUANTILE,
                         pad: int = 4) -> List[dict]:
    """
    The handwritten runs inside one line crop, as sub-crops with offsets.

    A land-record row is a printed label, a wide blank, then the Patwari's
    entry. Handing a recogniser the WHOLE row is what makes it fail: measured
    on five handwritten fields, TrOCR given the full row returned
    "owner 1887 1888" and "share # and #", and given a tight crop of the same
    value returned "Ramesh Kumar Yadav" and "Shri Mohan Lal Yadav" exactly.
    Same model, same pixels, 0 of 5 against 3 of 5.

    So this splits at the label-to-value gap - the same split that stopped
    printed form rows being flagged as handwriting - and returns only the
    runs that score as handwritten, each with the x-offset needed to map it
    back into the caller's line.
    """
    if not CV_AVAILABLE or gray is None or gray.size == 0:
        return []
    h, w = gray.shape[:2]
    if h < MIN_LINE_HEIGHT:
        return []
    ink = _binarise(gray)
    boxes = _components(ink, h, w)
    if len(boxes) < MIN_COMPONENTS:
        return []

    out: List[dict] = []
    for seg in split_segments(boxes):
        if len(seg) < MIN_COMPONENTS:
            continue
        feats = _features_from(gray, ink, seg)
        if not feats:
            continue
        verdict = score(feats, profile, quantile=quantile)
        if not verdict or not verdict.get("is_handwriting_suspected"):
            continue
        # int(): component boxes can carry float coordinates, and a float
        # slice index raises rather than rounding.
        x0 = int(max(0, min(b[0] for b in seg) - pad))
        x1 = int(min(w, max(b[0] + b[2] for b in seg) + pad))
        y0 = int(max(0, min(b[1] for b in seg) - pad))
        y1 = int(min(h, max(b[1] + b[3] for b in seg) + pad))
        if x1 - x0 < 8 or y1 - y0 < MIN_LINE_HEIGHT:
            continue
        out.append({
            "crop": gray[y0:y1, x0:x1],
            "x_offset": int(x0), "y_offset": int(y0),
            "width": int(x1 - x0), "height": int(y1 - y0),
            "score": verdict.get("score"),
        })
    return out


def crop_line(page_gray, bbox) -> Optional["_np.ndarray"]:
    """
    The image under a Tesseract line box, with a small vertical margin.

    Tesseract's line boxes clip tight to the ink, which cuts off descenders
    and the Devanagari headline - both of which the baseline and height
    features depend on - so a little context is included.
    """
    if not CV_AVAILABLE or page_gray is None or not bbox:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if x1 <= x0 or y1 <= y0:
        return None
    pad = max(2.0, (y1 - y0) * 0.15)
    h, w = page_gray.shape[:2]
    a = max(0, int(round(y0 - pad)))
    b = min(h, int(round(y1 + pad)))
    c = max(0, int(round(x0)))
    d = min(w, int(round(x1)))
    if b - a < MIN_LINE_HEIGHT or d - c < MIN_LINE_HEIGHT:
        return None
    return page_gray[a:b, c:d]
