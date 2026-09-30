#!/usr/bin/env python3
"""
Build a LayoutXLM training set from documents this system has already read.

WHY THIS IS THE UNBLOCKING PIECE

LayoutXLM is the right model for these documents and cannot be used yet.
Measured: LayoutLMv3's tokenizer turns "भूमि का विवरण" into 61 fragments of
raw UTF-8 and its Devanagari predictions are incoherent, while LayoutXLM
turns the same text into 13 real words. But layoutxlm-base ships with no
task head - it classifies nothing until fine-tuned, and fine-tuning needs
token-level labels with bounding boxes, which nobody has for Indian land
records.

Two things this project already produces make that dataset generatable
rather than hand-annotated:

  * Sarvam returns grounded ground truth - 16 of 16 co-owners with the
    correct father's name each, every value checked to appear verbatim in
    the source before it is kept.
  * The documents carry word-level boxes. A digital PDF gives them exactly,
    from its own text layer.

So the labels can be ALIGNED rather than drawn: take a value Sarvam
extracted, find the run of words that spells it, and tag that run. Every
document run through the pipeline becomes a training example.

WHAT THIS IS NOT

It is not a substitute for checking the result. The alignment is exact - a
value must be found as a contiguous run of words or it is skipped, never
approximated - but Sarvam is the teacher here, so any systematic error it
makes is taught to the student. Spot-check the output before training on it.

OUTPUT is JSONL, one document per line:
    {"id", "words": [...], "bboxes": [[x0,y0,x1,y1] 0-1000], "labels": [...]}

Run:
    python3 tools/make_layout_dataset.py <pdf-or-folder> [-o dataset.jsonl]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# .env BEFORE importing llm_extractor, not after.
#
# run.py loads it; a tool that imports the backend directly does not, and
# llm_extractor reads os.environ at IMPORT time to decide whether it has a
# provider. Without this the key in .env is invisible and the module reports
# no teacher available - which looks identical to the model returning
# nothing, and cost a debugging round to tell apart.
def _load_env(path=os.path.join(ROOT, ".env")):
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())
    except FileNotFoundError:
        pass


_load_env()

import ocr_engine        # noqa: E402
import llm_extractor     # noqa: E402

CACHE_PATH = os.path.join(ROOT, "storage", "layout_teacher_cache.json")

# The tag set. Deliberately small: these are the fields the rules get wrong
# on real documents, which is the only reason to train a model at all.
# Adding a tag for khasra or area would be teaching a student to do what the
# teacher's teacher already does at 0.94.
FIELD_TAGS = ("OWNER", "FATHER", "VILLAGE", "TEHSIL", "DISTRICT")
LABELS = ["O"] + [f"{p}-{t}" for t in FIELD_TAGS for p in ("B", "I")]


def normalise(text: str) -> str:
    """Compare on what is written, not how it is joined or spaced."""
    text = (text or "").translate(dict.fromkeys(map(ord, "​‌‍﻿")))
    return re.sub(r"\s+", "", text).casefold()


def pdf_words(path: str):
    """
    Word boxes from a PDF's own text layer, as (word, x0, y0, x1, y1, page).

    Exact rather than interpolated. The Line objects the pipeline carries
    hold only LINE boxes, and splitting a line into words by character width
    would put every box slightly wrong - on a model whose whole advantage is
    knowing where things sit, approximated geometry is the one shortcut not
    worth taking.
    """
    try:
        import pymupdf
    except Exception:
        try:
            import fitz as pymupdf
        except Exception:
            return []
    out = []
    try:
        doc = pymupdf.open(path)
    except Exception:
        return []
    try:
        for number in range(doc.page_count):
            page = doc.load_page(number)
            width, height = page.rect.width or 1, page.rect.height or 1
            for x0, y0, x1, y1, word, *_ in page.get_text("words"):
                if not word.strip():
                    continue
                out.append((word.strip(),
                            int(1000 * x0 / width), int(1000 * y0 / height),
                            int(1000 * x1 / width), int(1000 * y1 / height),
                            number))
    finally:
        doc.close()
    return out


def find_span(words, value: str, tags=None):
    """
    The first UNCLAIMED contiguous run of words spelling `value`, or None.

    Unclaimed matters more than it sounds. A father's name is shared by his
    children, so "हरिप्रसाद" is the father of four different owners on one
    page. Returning the first occurrence every time meant the second, third
    and fourth lookups all landed on a span that was already tagged and were
    discarded - measured: 3 FATHER labels for 15 owners. Skipping past
    claimed spans recovers the rest.

    Contiguity is required. A value whose words are scattered is not a span,
    and tagging the gap would teach the model that everything between two
    names belongs to the field.
    """
    target = normalise(value)
    if not target:
        return None
    for start in range(len(words)):
        joined = ""
        for end in range(start, min(start + 12, len(words))):
            joined += normalise(words[end][0])
            if joined == target:
                if tags is None or all(t == "O" for t in tags[start:end + 1]):
                    return (start, end)
                break                   # claimed - keep looking further on
            if len(joined) > len(target):
                break
    return None


def teacher_values(path: str, text: str, cache: dict) -> dict:
    """
    Ground truth from Sarvam, cached - each call takes about 90 seconds.
    """
    key = os.path.basename(path)
    if key in cache:
        return cache[key] or {}
    if not llm_extractor.LLM_AVAILABLE:
        return {}
    try:
        structured = llm_extractor.extract_structured(text)
    except Exception:
        structured = None
    cache[key] = structured or {}
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return structured or {}


def label_document(path: str, cache: dict):
    """One JSONL record, or None when nothing could be labelled."""
    words = pdf_words(path)
    if not words:
        return None, "no word boxes (scanned PDF - not supported yet)"

    result = ocr_engine.extract(path)
    text = "\n".join(getattr(l, "text", "") for l in result.lines)
    truth = teacher_values(path, text, cache)
    if not truth:
        return None, "no teacher output"

    tags = ["O"] * len(words)
    tagged = 0

    def apply(value, tag):
        nonlocal tagged
        span = find_span(words, value, tags)
        if not span:
            return
        start, end = span
        tags[start] = f"B-{tag}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{tag}"
        tagged += 1

    for owner in (truth.get("owners") or []):
        apply(owner.get("name"), "OWNER")
        apply(owner.get("father_name"), "FATHER")
    for key, tag in (("village", "VILLAGE"), ("tehsil", "TEHSIL"),
                     ("district", "DISTRICT")):
        apply(truth.get(key), tag)

    if not tagged:
        return None, "teacher values found, none alignable to word spans"

    # ONE RECORD PER PAGE, not per document.
    #
    # Boxes are normalised to 0-1000 within their OWN page, so a word at the
    # top of page 2 and a word at the top of page 1 both carry y=2. Putting
    # them in one sequence tells the model those two words sit in the same
    # place - the opposite of what a layout model is for. Measured on a real
    # report: the owner list ran across a page break and the second page's
    # coordinates restarted at the top.
    by_page = {}
    for index, word in enumerate(words):
        by_page.setdefault(word[5], []).append(index)

    records = []
    for page in sorted(by_page):
        indexes = by_page[page]
        page_tags = [tags[i] for i in indexes]
        if all(t == "O" for t in page_tags):
            continue                    # nothing to learn from this page
        # A page may begin part-way through a labelled run. Promote its
        # first token to B- so the sequence stays valid BIO - a sequence
        # opening on I- is malformed and most trainers reject or mislabel it.
        if page_tags[0].startswith("I-"):
            page_tags[0] = "B-" + page_tags[0][2:]
        records.append({
            "id": f"{os.path.basename(path)}#p{page}",
            "words": [words[i][0] for i in indexes],
            "bboxes": [[words[i][1], words[i][2], words[i][3], words[i][4]]
                       for i in indexes],
            "labels": page_tags,
        })
    if not records:
        return None, "labels found but none on a page with usable geometry"
    return records, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("source", help="a PDF, or a folder of them")
    ap.add_argument("-o", "--out", default="layout_dataset.jsonl")
    args = ap.parse_args()

    paths = ([args.source] if args.source.lower().endswith(".pdf")
             else sorted(glob.glob(os.path.join(args.source, "*.pdf"))))
    if not paths:
        print("No PDFs found.")
        return 1

    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    written = 0
    spans = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in paths:
            records, why = label_document(path, cache)
            name = os.path.basename(path)[:44]
            if not records:
                print(f"  skip  {name:46} {why}")
                continue
            for record in records:
                # Counted from the labels themselves rather than carried in
                # the record: a B- tag is one span by definition, and a
                # count passed alongside can drift from what was written.
                spans += sum(1 for t in record["labels"] if t.startswith("B-"))
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            print(f"  ok    {name:46} {len(records)} page(s), "
                  f"{sum(len(r['words']) for r in records)} words")

    print()
    print(f"{written} page records from {len(paths)} document(s) -> {args.out}")
    print(f"{spans} labelled spans across {len(LABELS)} tags")
    if written:
        print()
        print("Next: fine-tune microsoft/layoutxlm-base on this with "
              "AutoModelForTokenClassification")
        print("on Kaggle (no CUDA here). Spot-check the labels first - "
              "Sarvam is the teacher,")
        print("so any systematic mistake it makes is what the student will "
              "learn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
