#!/usr/bin/env python3
"""
End-to-end exercise of the AI learning loop, measured on held-out documents.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The problem statement asks for extraction that "improves accuracy over time".
tools/measure_accuracy.py established the number; this closes the loop around
it and shows whether the number actually moves:

    1. render a larger corpus (every sample x every scan degradation)
    2. SPLIT IT BY SAMPLE into a training half and a held-out half
    3. score the held-out half with no learned model  -> the baseline
    4. review the training half against ground truth and write the
       disagreements through Database.update_field(), the same call the
       verifier UI makes, so they land in the corrections table
    5. learning.retrain() mines confusions, aliases and calibration from them
    6. score the held-out half again with that model applied

Why the split is by SAMPLE and not by file: the three scans of sample_09 are
degraded renders of one document with one set of field values. Training on the
skewed scan and testing on the faded one would let a learned alias memorise
that document's owner name and score it as generalisation. Splitting by sample
keeps every value in the test half genuinely unseen.

The improvement this reports is therefore what a real deployment would get
after a verifier corrected a few dozen fields - not a model graded on its own
training data.

Usage:
    python3 tools/train_learning_loop.py            # measure, leave model alone
    python3 tools/train_learning_loop.py --install  # also save the trained model
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import field_extractor as fe          # noqa: E402
import learning                       # noqa: E402
import make_samples                   # noqa: E402
import measure_accuracy as ma         # noqa: E402
from db import Database               # noqa: E402

# Deliberately short: the corpus filenames are long, and Windows still caps a
# path at 260 characters. "eval_corpus" was enough to push exactly one scan
# over the limit and drop it from the benchmark without any error.
CORPUS = os.path.join(ROOT, "storage", "eval")
SCAN_STYLES = ("skew_blur", "faded", "heavy")

# Samples 1-6 train, 7-11 are held out. Chosen before any measurement was
# taken, and left alone since: picking the split after seeing which half
# scores better is how a benchmark becomes a marketing number.
TRAIN_SAMPLES = {1, 2, 3, 4, 5, 6}


def sample_index(name: str) -> int:
    return int(name.split("_")[1])


def scan_name(sample: str, style: str) -> str:
    stem = os.path.splitext(sample)[0].replace("sample_", "")
    return "scan_" + stem + "_" + style + ".png"


def build_corpus() -> None:
    """Render every sample as a PDF plus one scan per degradation style."""
    os.makedirs(CORPUS, exist_ok=True)
    real_out = make_samples.OUT
    make_samples.OUT = CORPUS
    try:
        for spec in make_samples.SAMPLES:
            target = os.path.join(CORPUS, spec["name"])
            if not os.path.exists(target):
                make_samples.draw_record(target, spec["header"], spec["rows"],
                                         spec["footer"], seed=spec["seed"])
        pending = []
        for spec in make_samples.SAMPLES:
            for style in SCAN_STYLES:
                if not os.path.exists(os.path.join(CORPUS, scan_name(spec["name"], style))):
                    pending.append((spec["name"], style))
        if pending:
            make_samples.make_scans(pending)
    finally:
        make_samples.OUT = real_out


def corpus_documents(truth: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, Dict[str, str]]]:
    """[(sample name, file path, expected values)] for everything rendered."""
    out = []
    for name, expected in truth.items():
        candidates = [name] + [scan_name(name, s) for s in SCAN_STYLES]
        for candidate in candidates:
            path = os.path.join(CORPUS, candidate)
            if os.path.exists(path):
                out.append((name, path, expected))
    return out


def review_into_db(db: Database, user: dict,
                   docs: List[Tuple[str, str, Dict[str, str]]]) -> dict:
    """
    Play the verifier: run the pipeline over the training documents and record
    a confirmation or a correction for every field, exactly as the review UI
    does. This is the only place ground truth is allowed to touch the model.
    """
    confirmed = corrected = 0
    for name, path, expected in docs:
        print("    reviewing " + os.path.basename(path) + " ...", flush=True)
        scored = ma.score_document(path, expected)
        doc_id = db.insert_document(filename=os.path.basename(path), stored_path=path,
                                    sha256=Database.file_hash(path),
                                    file_size=os.path.getsize(path),
                                    status="needs_review", ocr_engine=scored["engine"])
        rows = [r for r in scored["rows"] if r["expected"] is not None]
        for row in rows:
            spec = fe.FIELD_BY_KEY.get(row["key"])
            db.insert_field(doc_id, {
                "key": row["key"], "display": spec.display if spec else row["key"],
                "value": row["got"], "confidence": row["confidence"],
                "status": "extracted", "page": 1, "source_line": "",
            })
        for row in rows:
            if row["outcome"] == "correct":
                db.update_field(doc_id, row["key"], row["got"], user, confirm_only=True)
                confirmed += 1
            else:
                # A verifier types the value that is printed on the page.
                db.update_field(doc_id, row["key"], row["expected"], user)
                corrected += 1
    return {"documents": len(docs), "confirmed": confirmed, "corrected": corrected}


def score_all(docs, apply_learned: bool) -> dict:
    scored = []
    for name, path, expected in docs:
        tag = " [learned]" if apply_learned else ""
        print("    scoring " + os.path.basename(path) + tag + " ...", flush=True)
        scored.append(ma.score_document(path, expected, apply_learned=apply_learned))
    report = ma.summarise(scored)
    report["documents"] = scored
    return report


def unreviewed_errors(report: dict) -> int:
    """
    Wrong values that nobody would have been asked to look at: the operational
    failure this system exists to prevent. A wrong value routed to review is a
    caught error; a wrong value auto-accepted is a corrupted land record.
    """
    n = 0
    for doc in report["documents"]:
        for row in doc["rows"]:
            if (row["outcome"] in ("wrong", "spurious")
                    and row["confidence"] >= fe.REVIEW_THRESHOLD):
                n += 1
    return n


def delta_line(label: str, before: float, after: float) -> str:
    diff = (after - before) * 100
    arrow = "+" if diff > 0 else ""
    return ("  " + label.ljust(26) + "%6.1f" % (before * 100) + "  ->"
            + "%6.1f" % (after * 100) + "   (" + arrow + "%.1f" % diff + ")")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help="save the trained model to storage/learned_model.json")
    ap.add_argument("--keep-corpus", action="store_true",
                    help="do not delete the rendered evaluation corpus afterwards")
    args = ap.parse_args()

    print("1. building evaluation corpus ...")
    build_corpus()
    truth = ma.ground_truth()
    everything = corpus_documents(truth)
    train = [d for d in everything if sample_index(d[0]) in TRAIN_SAMPLES]
    test = [d for d in everything if sample_index(d[0]) not in TRAIN_SAMPLES]
    print("   %d documents: %d train, %d held out" % (len(everything), len(train), len(test)))
    if not train or not test:
        print("   ! corpus incomplete - is pymupdf/OpenCV installed?")
        return 1

    print("\n2. baseline on held-out documents (no learned model) ...")
    before = score_all(test, apply_learned=False)

    tmpdir = tempfile.mkdtemp(prefix="lrdv_loop_")
    db = Database(os.path.join(tmpdir, "train.db"))
    row = db.get_user("verifier1")
    if row is None:
        raise SystemExit("no verifier account in the freshly seeded database")
    user = dict(row)

    print("\n3. verifier reviews the training half ...")
    review = review_into_db(db, user, train)
    print("   %d fields confirmed, %d corrected across %d documents"
          % (review["confirmed"], review["corrected"], review["documents"]))

    print("\n4. retraining ...")
    real_path = learning.MODEL_PATH
    learning.MODEL_PATH = os.path.join(tmpdir, "learned_model.json")
    model = learning.retrain(db)
    print("   %d correction samples -> %d active rules"
          % (model["samples"], model["active_rules"]))
    for c in model["confusions"][:8]:
        print("     confusion '%s' -> '%s'  support=%d  auto_apply=%s  kinds=%s"
              % (c["from"], c["to"], c["support"], c["auto_apply"],
                 ",".join(c["field_kinds"]) or "-"))
    for a in model["aliases"][:8]:
        print("     alias %s: '%s' -> '%s'  support=%d  auto_apply=%s"
              % (a["field_key"], a["wrong_value"], a["corrected_value"],
                 a["support"], a["auto_apply"]))
    for c in model["calibration"][:8]:
        print("     calibrate %-20s stated=%.2f observed=%.2f x%.3f (%s)"
              % (c["field_key"], c["stated_confidence"], c["observed_precision"],
                 c["multiplier"], c["direction"]))

    print("\n5. re-scoring the SAME held-out documents with the model ...")
    after = score_all(test, apply_learned=True)

    print("\n" + "=" * 68)
    print("LEARNING LOOP RESULT  (held-out documents only)")
    print("=" * 68)
    print(delta_line("precision %", before["precision"], after["precision"]))
    print(delta_line("recall %", before["recall"], after["recall"]))
    b, a = before["totals"], after["totals"]
    for outcome in ("correct", "wrong", "missed", "spurious"):
        print("  %-26s %6d  ->%6d" % (outcome, b.get(outcome, 0), a.get(outcome, 0)))

    rev_b = sum(d["needs_review"] for d in before["documents"])
    rev_a = sum(d["needs_review"] for d in after["documents"])
    print("  %-26s %6d  ->%6d" % ("fields sent to review", rev_b, rev_a))
    print("  %-26s %6d  ->%6d" % ("wrong + auto-accepted",
                                  unreviewed_errors(before), unreviewed_errors(after)))

    changes = [c for d in after["documents"] for c in d["learned"]]
    value_changes = [c for c in changes if c["type"] != "calibration"]
    if value_changes:
        print("\n  What the model did to values on held-out documents:")
        for c in value_changes:
            if c["type"] == "confusion_ambiguous":
                print("    %-19s %-20s '%s' -> review (%s)"
                      % (c["type"], c["field_key"], c["from"], ", ".join(c["candidates"])))
            else:
                print("    %-19s %-20s '%s' -> '%s'"
                      % (c["type"], c["field_key"], c["from"], c["to"]))
    else:
        print("\n  The model changed no values on held-out documents.")

    if args.install:
        learning.save_model(model, real_path)
        print("\n  Model installed to " + os.path.relpath(real_path, ROOT))
    else:
        print("\n  Model NOT installed (pass --install to keep it).")

    db.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    if not args.keep_corpus:
        shutil.rmtree(CORPUS, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
