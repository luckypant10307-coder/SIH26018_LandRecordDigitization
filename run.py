#!/usr/bin/env python3
"""
Launcher for the Intelligent Land Record Digitization and Validation System.
SIH 2026 | PS 26018 | Ministry of Rural Development (DoLR)

Usage:
    python3 run.py                # start on http://127.0.0.1:8000
    python3 run.py --port 9000    # start on another port
    python3 run.py --samples      # (re)generate the sample document corpus
    python3 run.py --check        # print an environment capability report
    python3 run.py --host 0.0.0.0 # expose on the LAN (for a projector/demo laptop)

There is deliberately nothing to install. The backend runs on the Python
standard library (http.server + sqlite3) and the front-end is plain HTML, CSS
and JavaScript with no build step. Optional libraries (PyMuPDF, OpenCV,
pytesseract) are used when present and cleanly degraded when absent, so a
failed `pip install` or a machine with no internet can never block a demo.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

MIN_PYTHON = (3, 8)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
                 f"(found {sys.version.split()[0]}).")


def report() -> int:
    """Print what this machine can actually do, honestly."""
    import fact_checker
    import llm_extractor
    import ocr_engine
    import table_structure
    import validator

    caps = ocr_engine.capabilities()
    print("\nEnvironment capability report")
    print("-" * 58)
    print(f"  Python                : {sys.version.split()[0]}")
    print(f"  Native PDF text layer : {'yes' if caps['pdf_text_layer'] else 'no (install PyMuPDF)'}")
    print(f"  Image preprocessing   : {'yes' if caps['image_preprocessing'] else 'no (install opencv-python)'}")
    print(f"  Tesseract OCR         : {'yes' if caps['tesseract'] else 'no'}")
    if caps["tesseract"]:
        langs = caps["tesseract_languages"]
        print(f"    Languages           : {', '.join(langs[:12])}"
              + (" ..." if len(langs) > 12 else ""))
        for need in ("hin", "eng"):
            if need not in langs:
                print(f"    ! Language pack '{need}' is missing.")
    else:
        print("    Scanned images will be queued for manual entry rather than")
        print("    guessed at. Install tesseract-ocr + the hin/eng language")
        print("    packs to enable the OCR path.")
    print(f"  LGD admin master      : {'loaded' if validator._MASTER.loaded else 'MISSING'}")
    print(f"  Fact-check ML engine  : {'available (scikit-learn)' if fact_checker.SKLEARN_AVAILABLE else 'no (install scikit-learn)'}")
    print(f"  Fact-check registry   : {'loaded' if fact_checker._REGISTRY.loaded else 'MISSING'}")
    if table_structure.PADDLE_STRUCTURE_AVAILABLE:
        print("  Table/layout (PaddleOCR): available (heavy - seconds/page on CPU, no GPU here)")
    elif table_structure._PYTHON_TOO_NEW:
        print(f"  Table/layout (PaddleOCR): no - PaddlePaddle does not yet support "
              f"Python {sys.version_info.major}.{sys.version_info.minor}; needs 3.8-3.12")
    else:
        print("  Table/layout (PaddleOCR): no (optional, install paddleocr[doc-parser] + paddlepaddle)")
    if llm_extractor.LLM_AVAILABLE:
        provider = ("Anthropic" if llm_extractor.ANTHROPIC_API_KEY
                    else "OpenAI" if llm_extractor.OPENAI_API_KEY else "NVIDIA")
        print(f"  LLM field suggestions : ACTIVE ({provider}) - document text is sent "
              "to a third party for fields the rule-based extractor could not read")
    else:
        print(f"  LLM field suggestions : off ({llm_extractor.unavailable_reason()})")

    samples = os.path.join(ROOT, "samples")
    count = len([f for f in os.listdir(samples)
                 if not f.startswith("_")]) if os.path.isdir(samples) else 0
    print(f"  Sample documents      : {count}")
    print("-" * 58)
    if not caps["tesseract"]:
        print("  Verdict: fully usable. Digital PDFs extract at full accuracy;")
        print("           scanned images route to human verification.\n")
    else:
        print("  Verdict: all three extraction paths available.\n")
    return 0


def build_samples() -> int:
    script = os.path.join(ROOT, "tools", "make_samples.py")
    if not os.path.exists(script):
        sys.exit("tools/make_samples.py not found.")
    import runpy
    sys.argv = [script]
    runpy.run_path(script, run_name="__main__")
    return 0


def main() -> int:
    check_python()

    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000, help="port to listen on")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind")
    ap.add_argument("--samples", action="store_true", help="regenerate sample documents and exit")
    ap.add_argument("--check", action="store_true", help="print a capability report and exit")
    args = ap.parse_args()

    if args.check:
        return report()
    if args.samples:
        return build_samples()

    # Generate the corpus on first run so the demo is never empty.
    samples = os.path.join(ROOT, "samples")
    if not os.path.isdir(samples) or not os.listdir(samples):
        print("No sample documents found - generating them first...")
        try:
            build_samples()
        except Exception as exc:
            print(f"  ! Sample generation failed ({exc}). Continuing anyway.")

    import server
    server.serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
