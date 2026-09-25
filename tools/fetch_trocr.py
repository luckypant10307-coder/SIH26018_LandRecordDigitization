#!/usr/bin/env python3
"""
Download the ONNX TrOCR weights used by backend/trocr_htr.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Weights are NOT committed - they are 600 MB - so this fetches them on demand
from the transformers.js export of microsoft/trocr-base-handwritten.

WHICH PRECISIONS, AND WHY NOT BOTH QUANTISED
--------------------------------------------
The encoder is fetched at full precision and the decoder at int8, which looks
inconsistent and is the result of measuring rather than assuming. On the
canonical IAM fixture (a cursive "industrie"):

    int8 encoder + int8 decoder -> "insalums true"    unusable
    fp32 encoder + int8 decoder -> "indus the"
    fp32 encoder + fp32 decoder -> "indus the"        no better

int8 quantisation of the VISION tower destroys the reading; the same
quantisation of the language decoder costs nothing measurable. Taking the
obvious route - quantise both, they are one model - would have shipped a
recogniser that emits fluent nonsense at ordinary confidence.

Usage:
    python3 tools/fetch_trocr.py            # fetch what is missing
    python3 tools/fetch_trocr.py --all      # also fetch the fp32 decoder
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

REPO = "Xenova/trocr-base-handwritten"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
TARGET = os.path.join(ROOT, "storage", "models", "trocr")

REQUIRED = [
    ("tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
    ("generation_config.json", "generation_config.json"),
    ("onnx/encoder_model.onnx", "encoder_model.onnx"),
    ("onnx/decoder_model_quantized.onnx", "decoder_model_quantized.onnx"),
]
OPTIONAL = [("onnx/decoder_model.onnx", "decoder_model.onnx")]


def fetch(remote: str, local: str) -> None:
    path = os.path.join(TARGET, local)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("  have %-32s %8.1f MB" % (local, os.path.getsize(path) / 1e6))
        return
    started = time.time()
    tmp = path + ".part"
    urllib.request.urlretrieve(BASE + remote, tmp)
    os.replace(tmp, path)
    print("  got  %-32s %8.1f MB  %5.0fs"
          % (local, os.path.getsize(path) / 1e6, time.time() - started))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="also fetch the full-precision decoder (990 MB, no "
                         "measured accuracy gain)")
    args = ap.parse_args()

    os.makedirs(TARGET, exist_ok=True)
    print("fetching TrOCR ONNX weights from %s" % REPO)
    for remote, local in REQUIRED + (OPTIONAL if args.all else []):
        fetch(remote, local)

    import trocr_htr
    trocr_htr.reset_cache()
    print("\n%s" % trocr_htr.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
