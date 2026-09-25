#!/usr/bin/env python3
"""
Fit the printed-text profile used by backend/handwriting.py, and measure how
often it is wrong about print.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Handwriting detection here is NOVELTY detection: print is characterised from
the sample corpus, and a line far from that profile is flagged as not-print.
That design makes exactly one thing measurable, and this tool measures it.

  * FALSE POSITIVES - printed lines wrongly flagged as handwriting - ARE
    measured, on lines from documents the profile was not fitted on. This is
    the number that decides whether the feature is usable: every false
    positive sends a perfectly good printed field to a human for no reason,
    and enough of them make the review queue worthless.

  * TRUE POSITIVES - real Patwari handwriting actually being caught - are NOT
    measured here and cannot be, because the corpus contains no handwriting.
    Nothing this tool prints should be read as evidence of recall.

The split is by SAMPLE, not by line: lines from one document share a typeface,
a scan and a degradation, so fitting on some lines of a page and testing on
its others would measure memorisation of that page.

Usage:
    python3 tools/fit_print_profile.py                # measure, do not install
    python3 tools/fit_print_profile.py --install      # save the profile
    python3 tools/fit_print_profile.py --sweep        # FP rate vs threshold
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import handwriting as hw              # noqa: E402
import ocr_engine                     # noqa: E402

CORPUS = os.path.join(ROOT, "storage", "eval")
FALLBACK_CORPUS = os.path.join(ROOT, "samples")

# Same split as tools/train_learning_loop.py and tools/train_denoiser.py, so
# "held out" means the same documents everywhere in this project.
TRAIN_SAMPLES = {1, 2, 3, 4, 5, 6}


def sample_index(name: str) -> int:
    for part in name.split("_"):
        if part.isdigit():
            return int(part)
    return -1


def scan_paths() -> List[str]:
    folder = CORPUS if os.path.isdir(CORPUS) else FALLBACK_CORPUS
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.startswith("scan_") and f.lower().endswith(".png")]


def lines_of(path: str) -> List[Dict[str, float]]:
    """
    Features for every printed text line on one scan.

    Deliberately runs the REAL pipeline - assess_and_preprocess then
    Tesseract - and crops from the preprocessed image, because that is the
    image Tesseract's line boxes are expressed in and the image handwriting
    detection will see in production. Measuring on the raw scan instead would
    profile a picture the detector never gets.
    """
    work = tempfile.mkdtemp(prefix="hwprof_")
    pre_path, _metrics, _warnings = ocr_engine.assess_and_preprocess(path, work)
    target = pre_path or path
    page = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    if page is None:
        return []

    languages, _diag = ocr_engine.select_languages(
        path, ocr_engine.tesseract_languages(), work)
    tsv = ocr_engine._ocr_page(target, languages or "eng")
    if tsv is None:
        return []

    grouped: Dict[tuple, List[int]] = defaultdict(list)
    texts = tsv.get("text", [])
    for i in range(len(texts)):
        if not (texts[i] or "").strip():
            continue
        grouped[(tsv["block_num"][i], tsv["par_num"][i], tsv["line_num"][i])].append(i)

    out: List[Dict[str, float]] = []
    for key in sorted(grouped):
        idx = grouped[key]
        x0 = min(tsv["left"][i] for i in idx)
        y0 = min(tsv["top"][i] for i in idx)
        x1 = max(tsv["left"][i] + tsv["width"][i] for i in idx)
        y1 = max(tsv["top"][i] + tsv["height"][i] for i in idx)
        crop = hw.crop_line(page, (x0, y0, x1, y1))
        if crop is None:
            continue
        # SEGMENTS, not whole lines. Scoring works per text run (see
        # handwriting.split_segments), so the profile has to describe a text
        # run too. Fitting on whole lines and scoring segments would compare
        # each value against statistics of a different kind of object - gap
        # variation especially is far larger across a form row than within
        # one run of text, so every segment would look suspiciously regular
        # and nothing would ever be flagged.
        out.extend(hw.segment_features(crop))
    return out


def false_positive_rate(profile: dict, runs: List[Dict[str, float]],
                        quantile: float) -> Tuple[int, int]:
    """
    (flagged, total) over text runs that are all known to be PRINTED.

    This is the COMBINED rate across all four features, which is the number
    that matters and is not the same as the per-feature quantile: a run is
    flagged if ANY feature exceeds its bar, so four features each mis-firing
    on 0.5% of print give somewhere between 0.5% and 2% overall depending on
    how correlated they are. Measuring beats reasoning about it.
    """
    flagged = 0
    for features in runs:
        verdict = hw.score(features, profile, quantile)
        if verdict and verdict["is_handwriting_suspected"]:
            flagged += 1
    return (flagged, len(runs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help="save the profile to storage/print_profile.json")
    ap.add_argument("--sweep", action="store_true",
                    help="report the false-positive rate across quantiles")
    args = ap.parse_args()

    paths = scan_paths()
    if not paths:
        print("No scans found. Run tools/make_samples.py (and "
              "tools/train_learning_loop.py --keep-corpus for the full set).")
        return 1

    print("1. measuring printed lines on %d scans ..." % len(paths))
    train: List[Dict[str, float]] = []
    held: List[Dict[str, float]] = []
    for path in paths:
        name = os.path.basename(path)
        features = lines_of(path)
        bucket = train if sample_index(name) in TRAIN_SAMPLES else held
        bucket.extend(features)
        print("   %-46s %3d text runs" % (name[:46], len(features)), flush=True)

    print("\n   %d text runs to fit on, %d held out" % (len(train), len(held)))
    if len(train) < 30 or not held:
        print("   ! not enough lines to fit a profile honestly.")
        return 1

    print("\n2. fitting the print profile ...")
    profile = hw.fit_profile(train)
    for name, stats in profile["features"].items():
        q = stats.get("quantiles") or {}
        print("     %-20s median %7.4f  p99 %7.4f  p99.5 %7.4f  max %7.4f"
              % (name, q.get("0.5", 0.0), q.get("0.99", 0.0),
                 q.get("0.995", 0.0), q.get("1.0", 0.0)))

    print("\n3. FALSE POSITIVES on held-out PRINTED text runs")
    print("   (a flag here is always wrong - every one of these is print)")
    candidates = ([q for q in hw.PROFILE_QUANTILES if q >= 0.9] if args.sweep
                  else [hw.DEFAULT_QUANTILE])
    for quantile in candidates:
        flagged, total = false_positive_rate(profile, held, quantile)
        rate = flagged / total * 100 if total else 0.0
        marker = "  <-- default" if quantile == hw.DEFAULT_QUANTILE else ""
        print("     quantile %-6s -> %3d/%3d flagged  (%5.2f%%)%s"
              % (quantile, flagged, total, rate, marker))

    print("\n   NOT MEASURED: whether real handwriting is caught. This corpus")
    print("   has no handwriting in it, so recall is unknown by construction.")

    if args.install:
        hw.save_profile(profile)
        hw.reset_cache()
        print("\n  Profile saved to %s"
              % os.path.relpath(hw.PROFILE_PATH, ROOT))
    else:
        print("\n  Profile NOT saved (pass --install to keep it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
