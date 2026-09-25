#!/usr/bin/env python3
"""
Per-field extraction accuracy harness.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The problem statement asks for a system whose accuracy improves with use. That
claim is unfalsifiable without a number to improve, so this measures one: it
runs the real pipeline (ocr_engine -> field_extractor) over the generated
sample corpus and scores every field against known ground truth.

Ground truth comes from tools/make_samples.py, which is what *printed* the
documents, so the expected value of each field is known exactly rather than
eyeballed. The label -> field-key mapping below is written out by hand on
purpose: reusing field_extractor's own label aliases would make the harness
grade the extractor with the extractor's own answer key, and a broken alias
list would then score 100%.

Accuracy is measured against what is PRINTED on the page, not against what is
valid. sample_05 prints an area with no unit and sample_06 prints an
out-of-range date; reading those exactly as printed is correct extraction -
flagging them is validator.py's job, and is scored separately by the test
suite.

Usage:
    python3 tools/measure_accuracy.py              # digital PDFs (fast)
    python3 tools/measure_accuracy.py --scans      # include scanned PNGs (slow)
    python3 tools/measure_accuracy.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import field_extractor as fe          # noqa: E402
import ocr_engine                     # noqa: E402
import make_samples                   # noqa: E402

SAMPLES_DIR = os.path.join(ROOT, "samples")

# Hand-written answer key: every label string used by make_samples.SAMPLES,
# mapped to the schema field it denotes. Independent of field_extractor's
# FieldSpec aliases by design (see module docstring).
LABEL_TO_KEY: Dict[str, Optional[str]] = {
    "खाता संख्या / Khata Number": "khata_number",
    "खाते क्रमांक / Khata Number": "khata_number",
    "खाता क्रमांक": "khata_number",
    "खाता संख्या": "khata_number",
    "खसरा संख्या / Khasra Number": "khasra_number",
    "खसरा क्रमांक": "khasra_number",
    "खसरा संख्या": "khasra_number",
    "सर्वे क्रमांक / Survey Number": "survey_number",
    "ULPIN": "ulpin",
    "खातेदार का नाम / Owner Name": "owner_name",
    "भोगवटादार / Owner Name": "owner_name",
    "कृषक का नाम": "owner_name",
    "काश्तकार का नाम": "owner_name",
    "पिता का नाम / Father's Name": "father_name",
    "वडिलांचे नाव / Father's Name": "father_name",
    "पिता का नाम": "father_name",
    "पति का नाम": "father_name",
    "अंश / Share": "share",
    "हिस्सा / Share": "share",
    "हिस्सा": "share",
    "क्षेत्रफल / Area": "area",
    "क्षेत्र / Area": "area",
    "रकबा": "area",
    "भूमि श्रेणी / Land Classification": "land_classification",
    "जमिनीचा प्रकार / Land Classification": "land_classification",
    "भूमि का प्रकार": "land_classification",
    "किस्म भूमि": "land_classification",
    "ग्राम / Village": "village",
    "गाव / Village": "village",
    "ग्राम": "village",
    "तहसील / Tehsil": "tehsil",
    "तालुका / Tehsil": "tehsil",
    "तहसील": "tehsil",
    "जनपद / District": "district",
    "जिल्हा / District": "district",
    "जिला": "district",
    "राज्य / State": "state",
    "राज्य": "state",
    "नामांतरण संख्या / Mutation Number": "mutation_number",
    "फेरफार क्रमांक / Mutation Number": "mutation_number",
    "नामांतरण क्रमांक": "mutation_number",
    "नामांतरण संख्या": "mutation_number",
    "नामांतरण दिनांक / Mutation Date": "mutation_date",
    "फेरफार दिनांक / Mutation Date": "mutation_date",
    "नामांतरण दिनांक": "mutation_date",
    "पंजीकरण संख्या / Registration Number": "registration_number",
    "पंजीकरण संख्या": "registration_number",
    "पंजीकरण दिनांक / Registration Date": "registration_date",
    "पंजीकरण दिनांक": "registration_date",
    # Printed on the Rajasthan jamabandi but outside the extraction schema;
    # listed so an unmapped label is a real error, not an oversight.
    "जमाबंदी वर्ष": None,
}

DATE_KEYS = {"mutation_date", "registration_date"}

# The extractor deliberately canonicalises land classification to a controlled
# vocabulary, so the printed Hindi/Marathi term and a correct extraction never
# compare equal as strings. The expected CODE for every term the corpus prints
# is therefore written out here by hand - not derived from the extractor's own
# LAND_CLASSES table, which is the thing being graded. Writing this by hand is
# what exposed the substring inversion where 'असिंचित' (unirrigated) scored as
# irrigated_agricultural, because 'सिंचित' is a substring of it.
EXPECTED_CLASS: Dict[str, str] = {
    "सिंचित कृषि भूमि": "irrigated_agricultural",   # irrigated agricultural
    "सिंचित": "irrigated_agricultural",              # irrigated
    "चाही": "irrigated_agricultural",                # chahi - well-irrigated (Rajasthan)
    "असिंचित": "unirrigated_agricultural",           # unirrigated
    "बारानी": "unirrigated_agricultural",            # barani - rain-fed
    "जिरायत": "unirrigated_agricultural",            # jirayat - dry crop (Maharashtra)
    "आबादी": "residential",                          # abadi - inhabited
    "बंजर": "barren",                                # banjar - barren
    # Genuinely outside the controlled vocabulary: the correct behaviour is to
    # pass the raw text through with low confidence, not to force a code.
    "मिश्रित प्रकार": "मिश्रित प्रकार",
}

# Honorifics the extractor strips from person names. Stripping them is correct,
# so the harness strips them from the expected value too before comparing.
HONORIFICS = ("श्री", "श्रीमती", "shri", "sri", "smt", "mr", "mrs")


def _norm(text: Optional[str]) -> str:
    """Comparison form: Indic digits folded to ASCII, case and spacing dropped."""
    if not text:
        return ""
    return " ".join(fe.normalise_digits(fe.normalise(str(text))).casefold().split())


def _strip_honorific(name: str) -> str:
    low = name.strip()
    for h in HONORIFICS:
        if low.lower().startswith(h.lower() + " "):
            return low[len(h):].strip()
    return low


def values_agree(key: str, expected: str, got: Optional[str]) -> bool:
    """
    True when `got` conveys the same fact as `expected`.

    The extractor canonicalises dates to ISO and areas to square metres, so a
    literal string comparison would score correct extractions as wrong. Both
    sides are put through the same parser and compared on meaning; everything
    else compares as normalised text.
    """
    if not got:
        return False
    if key == "land_classification":
        want = EXPECTED_CLASS.get(expected.strip())
        if want is None:
            raise KeyError(f"{expected!r} has no entry in EXPECTED_CLASS")
        return _norm(want) == _norm(got)
    if key in ("owner_name", "father_name"):
        expected = _strip_honorific(expected)
    if _norm(expected) == _norm(got):
        return True
    if key in DATE_KEYS:
        a, b = fe.parse_date(expected), fe.parse_date(got)
        if a and b and a.get("iso") and a.get("iso") == b.get("iso"):
            return True
    if key == "area":
        a, b = fe.parse_area(expected), fe.parse_area(got)
        if a and b and a.get("sqm") and b.get("sqm"):
            return abs(a["sqm"] - b["sqm"]) <= max(1.0, 0.005 * a["sqm"])
    return False


def ground_truth() -> Dict[str, Dict[str, str]]:
    """{pdf filename: {field key: printed value}} straight from the generator."""
    truth: Dict[str, Dict[str, str]] = {}
    for sample in make_samples.SAMPLES:
        expected: Dict[str, str] = {}
        for label, value in sample["rows"]:
            if not label:
                continue
            if label not in LABEL_TO_KEY:
                raise KeyError(
                    f"{label!r} in {sample['name']} has no entry in LABEL_TO_KEY. "
                    f"Add it, or map it to None if it is outside the schema."
                )
            key = LABEL_TO_KEY[label]
            if key:
                expected[key] = value
        truth[sample["name"]] = expected
    return truth


# path -> ocr_engine.ExtractionResult, populated by score_document().
_OCR_CACHE: Dict[str, object] = {}


def score_document(path: str, expected: Dict[str, str],
                   apply_learned: bool = False) -> dict:
    started = time.time()
    # OCR is the expensive step and is deterministic for a given file, so a
    # before/after comparison of the learned model does not pay for it twice.
    result = _OCR_CACHE.get(path)
    if result is None:
        work = tempfile.mkdtemp(prefix="lrdv_acc_")
        result = ocr_engine.extract(path, work_dir=work)
        _OCR_CACHE[path] = result
    fields = fe.extract_fields(result.lines)

    # The same learned model, applied through the same call the server makes,
    # so an improvement measured here is an improvement a user would get.
    learned: List[dict] = []
    if apply_learned:
        import learning
        import server
        learned = learning.apply_model(
            fields, corroborate=server._make_corroborator(fields))
    elapsed = time.time() - started

    by_key = {f.key: f for f in fields}
    rows = []
    for key, want in expected.items():
        f = by_key.get(key)
        got = f.value if f else None
        conf = float(f.confidence) if f else 0.0
        if got is None:
            outcome = "missed"           # printed on the page, not extracted
        elif values_agree(key, want, got):
            outcome = "correct"
        else:
            outcome = "wrong"            # extracted, but not what is printed
        rows.append({"key": key, "expected": want, "got": got,
                     "confidence": conf, "outcome": outcome})

    # A value produced for a field the document never printed. These matter:
    # a confident wrong answer is worse than an admitted blank.
    for key, f in by_key.items():
        if key not in expected and f.value:
            rows.append({"key": key, "expected": None, "got": f.value,
                         "confidence": float(f.confidence), "outcome": "spurious"})

    return {"document": os.path.basename(path), "engine": result.engine,
            "seconds": round(elapsed, 1), "rows": rows, "learned": learned,
            "needs_review": sum(1 for f in fields if f.status == "needs_review")}


def summarise(docs: List[dict]) -> dict:
    per_field: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: Dict[str, int] = defaultdict(int)
    # Calibration: does a high confidence score actually mean a right answer?
    buckets = {"0.90-1.00": [0, 0], "0.80-0.90": [0, 0],
               "0.60-0.80": [0, 0], "0.00-0.60": [0, 0]}

    for doc in docs:
        for row in doc["rows"]:
            per_field[row["key"]][row["outcome"]] += 1
            totals[row["outcome"]] += 1
            if row["outcome"] in ("correct", "wrong", "spurious"):
                c = row["confidence"]
                name = ("0.90-1.00" if c >= 0.90 else "0.80-0.90" if c >= 0.80
                        else "0.60-0.80" if c >= 0.60 else "0.00-0.60")
                buckets[name][0] += 1
                buckets[name][1] += 1 if row["outcome"] == "correct" else 0

    attempted = totals["correct"] + totals["wrong"] + totals["spurious"]
    present = totals["correct"] + totals["wrong"] + totals["missed"]
    return {
        "totals": dict(totals),
        # Of the values it produced, how many were right.
        "precision": round(totals["correct"] / attempted, 4) if attempted else 0.0,
        # Of the values printed on the pages, how many it got right.
        "recall": round(totals["correct"] / present, 4) if present else 0.0,
        "per_field": {k: dict(v) for k, v in sorted(per_field.items())},
        "calibration": {k: {"n": v[0], "accuracy": round(v[1] / v[0], 3) if v[0] else None}
                        for k, v in buckets.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", action="store_true",
                    help="also score the scanned PNGs (needs Tesseract; slow)")
    ap.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    ap.add_argument("--learned", action="store_true",
                    help="apply the trained model (storage/learned_model.json) "
                         "before scoring, to measure what the learning loop bought")
    args = ap.parse_args()

    truth = ground_truth()
    targets = []
    for name, expected in truth.items():
        path = os.path.join(SAMPLES_DIR, name)
        if os.path.exists(path):
            targets.append((path, expected))
        else:
            print(f"  ! {name} not generated yet - run tools/make_samples.py")

    if args.scans:
        for scan in sorted(f for f in os.listdir(SAMPLES_DIR) if f.startswith("scan_")):
            stem = scan[len("scan_"):].rsplit("_", 1)[0]
            match = next((n for n in truth if n.startswith("sample_" + stem)), None)
            if match:
                targets.append((os.path.join(SAMPLES_DIR, scan), truth[match]))

    if not targets:
        print("No sample documents found. Run: python3 tools/make_samples.py")
        return 1

    docs = []
    for path, expected in targets:
        print(f"  scoring {os.path.basename(path)} ...", flush=True)
        docs.append(score_document(path, expected, apply_learned=args.learned))

    report = summarise(docs)
    report["documents"] = docs

    print("\n" + "=" * 68)
    print("FIELD EXTRACTION ACCURACY"
          + ("  [learned model applied]" if args.learned else ""))
    print("=" * 68)
    t = report["totals"]
    print(f"  correct {t.get('correct', 0):4d}   wrong {t.get('wrong', 0):4d}   "
          f"missed {t.get('missed', 0):4d}   spurious {t.get('spurious', 0):4d}")
    print(f"  precision {report['precision'] * 100:5.1f}%   "
          f"recall {report['recall'] * 100:5.1f}%")

    print("\n  Weakest fields (recall):")
    scored = []
    for key, counts in report["per_field"].items():
        present = counts.get("correct", 0) + counts.get("wrong", 0) + counts.get("missed", 0)
        if present:
            scored.append((counts.get("correct", 0) / present, key, counts, present))
    for rate, key, counts, present in sorted(scored)[:8]:
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"    {rate * 100:5.1f}%  {key:22s} ({detail})")

    print("\n  Confidence calibration (is a high score actually more accurate?):")
    for name, stats in report["calibration"].items():
        if stats["n"]:
            print(f"    conf {name}:  n={stats['n']:3d}  accuracy={stats['accuracy'] * 100:5.1f}%")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n  Full report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
