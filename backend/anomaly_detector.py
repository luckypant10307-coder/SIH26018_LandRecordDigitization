"""
One-class anomaly detection over a document's own pipeline signals.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Why one-class, not a fake/genuine classifier: a classifier needs labelled
examples of both classes, and no labelled forged-document dataset exists
for Indian land records - manufacturing one would mean fabricating forged
government documents, which this project will not do. scikit-learn's
IsolationForest instead learns what "normal" looks like from documents that
have already been through human review and approved, using features the
rest of this pipeline already computes (validator trust score,
field-extraction confidence and completeness, OCR quality, the seal/
signature/stamp-paper presence signals from document_authenticity.py) - no
new extraction step, and no training data beyond what real usage naturally
accumulates over time, the same self-improving pattern learning.py already
uses for confidence recalibration.

SCOPE, STATED AS PLAINLY AS EVERYWHERE ELSE IN THIS PROJECT: this is
anomaly detection, not forgery detection. A document scored as anomalous is
statistically unusual relative to the documents this office has approved
before - it may be a genuine but unusual record (a very old deed, a
multi-owner khata with an atypical share pattern), not a forged one. A
document scored as normal is not thereby proven authentic. Treat an
ANOMALY_DETECTED flag as "worth a second look", never as a fraud verdict -
the same distinction document_authenticity.py draws for seals and
signatures.

Cold start is handled honestly: with fewer than MIN_TRAINING_SAMPLES
approved documents, there is no reliable baseline to compare against, and
retrain() says so explicitly rather than training on too little data and
reporting false confidence.

A further real sensitivity, found while testing this module rather than
assumed: IsolationForest cannot build a meaningful decision boundary when
most training features are an exact constant across every approved
document - a synthetic test fixture that varied only one of nine features
scored a wildly different document identically to a normal one (predict=1
for both), while the same features with realistic small variance on every
dimension correctly separated them (predict=1 vs -1). In practice this
means the baseline is only as useful as the natural variation in the
approved-document history it is trained on; an office whose approved
documents are unusually uniform will get a correspondingly weak baseline,
not a broken one - this is inherent to one-class anomaly detection on
near-uniform "normal" data, not a defect to fix.

`train()`'s `contamination` parameter (default 0.1) is not a tuning knob to
ignore: it tells IsolationForest to assume that fraction of *any* scored
set - including the approved-document training set itself - are outliers,
and it will flag roughly that many even among documents already approved
by a human. Confirmed directly: two of thirty synthetic training documents
scored as anomalous purely for sitting at the edge of the fixture's value
range, not for anything resembling a real problem. Lower it if 1-in-10
review flags on an office's own approved history is too aggressive for how
this gets used downstream.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


_sklearn_ensemble = _try_import("sklearn.ensemble")
_numpy = _try_import("numpy")

SKLEARN_AVAILABLE = _sklearn_ensemble is not None and _numpy is not None

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "storage", "anomaly_model.pkl")

# Set from a direct measurement, not a guess: with an obviously extreme
# document (trust_score 12 against a training range of 90-97) held out and
# scored across 5 random seeds at each size, 15 samples caught it 4/5 times,
# 20 samples caught it only 2/5 - worse, non-monotonically - and 25+ caught
# it 5/5 every time. A subtler real anomaly needs more margin than an
# obvious one, so this sits well above the 25 where reliability first
# appeared. A baseline built from too few documents is worse than no
# baseline at all, because it reports confidence it has not earned.
MIN_TRAINING_SAMPLES = 50

FEATURE_NAMES = [
    "trust_score", "error_count", "warning_count",
    "mean_field_confidence", "completeness", "legibility_score",
    "has_seal", "has_signature", "has_stamp_paper",
]


def extract_features(document: dict) -> List[float]:
    """
    Build a numeric feature vector from a processed document's own record.
    `document` is the shape returned by db.py's get_document(): trust_score,
    error_count, warning_count, summary (dict), quality (dict), issues
    (list of {rule, severity, ...} dicts) - every value here already exists
    elsewhere in the pipeline, nothing is computed freshly for this.
    """
    summary = document.get("summary") or {}
    quality = document.get("quality") or {}
    issues = document.get("issues") or []
    rule_names = {i.get("rule") for i in issues}

    return [
        float(document.get("trust_score") or 0.0),
        float(document.get("error_count") or 0),
        float(document.get("warning_count") or 0),
        float(summary.get("mean_field_confidence") or 0.0),
        float(summary.get("completeness") or 0.0),
        float(quality.get("legibility_score") if quality.get("legibility_score") is not None else 100.0),
        1.0 if "SEAL_DETECTED" in rule_names else 0.0,
        1.0 if "SIGNATURE_DETECTED" in rule_names else 0.0,
        1.0 if "STAMP_PAPER_DETECTED" in rule_names else 0.0,
    ]


def train(feature_rows: List[List[float]], contamination: float = 0.1):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn/numpy not installed - anomaly detection requires them.")
    if len(feature_rows) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_SAMPLES} approved documents to train a "
            f"baseline; have {len(feature_rows)}.")
    np = _numpy
    from sklearn.ensemble import IsolationForest
    X = np.array(feature_rows, dtype=float)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X)
    return model


def save_model(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load_model(path: str = MODEL_PATH):
    if not SKLEARN_AVAILABLE:
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def score(model, features: List[float]) -> Tuple[bool, float]:
    """Returns (is_anomaly, raw_score) - higher raw_score means more normal."""
    np = _numpy
    X = np.array([features], dtype=float)
    is_anomaly = bool(model.predict(X)[0] == -1)
    raw_score = float(model.score_samples(X)[0])
    return is_anomaly, raw_score


def retrain(db, path: str = MODEL_PATH) -> dict:
    """
    Rebuild and persist the anomaly baseline from every approved document's
    own already-computed pipeline signals. Cheap enough to run on demand,
    the same way learning.retrain(db) is.
    """
    if not SKLEARN_AVAILABLE:
        return {"trained": False, "reason": "scikit-learn/numpy not installed.", "sample_count": 0}

    approved_ids = [d["id"] for d in db.list_documents(status="approved", limit=100000)]
    feature_rows = []
    for doc_id in approved_ids:
        doc = db.get_document(doc_id)
        if doc:
            feature_rows.append(extract_features(doc))

    if len(feature_rows) < MIN_TRAINING_SAMPLES:
        return {
            "trained": False,
            "reason": f"Only {len(feature_rows)} approved documents; "
                     f"need at least {MIN_TRAINING_SAMPLES} to establish a baseline.",
            "sample_count": len(feature_rows),
        }

    model = train(feature_rows)
    save_model(model, path=path)
    return {"trained": True, "sample_count": len(feature_rows), "features": FEATURE_NAMES}
