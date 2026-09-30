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


def load_env_file(path: str = None) -> int:
    """
    Read .env into the environment, if there is one. Returns how many names
    were set.

    This has to run BEFORE any backend module is imported, because several of
    them read os.environ at import time to decide whether an optional service
    is available - bhashini.py and llm_extractor.py both do. Every import in
    this file is inside a function for exactly that reason; moving one to the
    top would silently disable whatever the .env was meant to switch on.

    Written by hand rather than with python-dotenv because the whole point of
    this project is that `python3 run.py` works with nothing installed. An
    existing environment variable always wins, so an operator can override one
    for a single run without editing the file.
    """
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return 0
    applied = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                name, value = name.strip(), value.strip()
                # Quotes are stripped so a value with spaces can be written
                # either way; anything after a '#' is kept, because a key can
                # legitimately contain one.
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if name and name not in os.environ:
                    os.environ[name] = value
                    applied += 1
    except OSError:
        return applied
    return applied


load_env_file()


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
                 f"(found {sys.version.split()[0]}).")


def report() -> int:
    """Print what this machine can actually do, honestly."""
    import bhashini
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
        # Ask the module which provider it will ACTUALLY use rather than
        # re-deriving it here. The two answers diverged: this line reported
        # NVIDIA while _provider() had been changed to prefer Sarvam, so the
        # status line described a call that was never going to be made.
        provider = (llm_extractor._provider() or "none").capitalize()
        print(f"  LLM field suggestions : ACTIVE ({provider}) - document text is sent "
              "to a third party for fields the rule-based extractor could not read")
    else:
        print(f"  LLM field suggestions : off ({llm_extractor.unavailable_reason()})")

    import sam_fallback
    sam = sam_fallback.status()
    if sam["enabled"]:
        print(f"  Parcel fallback (SAM) : enabled ({sam['model']}) - fires ONLY when no")
        print("                          drawn boundary can be traced; output is approximate")
    else:
        print("  Parcel fallback (SAM) : off (SAM_FALLBACK != 1). Contour tracing is the")
        print("                          primary path and measured better on readable maps.")

    import ner_extractor
    if ner_extractor.ner_available():
        print("  NER cross-check (en)  : spaCy en_core_web_sm")
    else:
        print("  NER cross-check (en)  : off (pip install spacy; spacy download en_core_web_sm)")
    if ner_extractor.indic_ner_available():
        print(f"  NER cross-check (Indic): {ner_extractor.INDIC_NER_MODEL}")
        print("                          Devanagari owner/father names are now checked "
              "against the source line")
    else:
        status = ner_extractor.indic_ner_status()
        why = status["error"] or "model not installed"
        print(f"  NER cross-check (Indic): off ({why})")
        print("                          Devanagari person fields get no NER claim either way.")

    bh = bhashini.capabilities()
    if bh["available"]:
        print("  Script transliteration: ACTIVE (Bhashini) - Indic place names are "
              "sent to a government service to be")
        print("                          matched against the Latin admin master")
        print(f"    Exonym table        : {bh['exonyms']} names transliteration cannot derive")
        print(f"    Cached              : {bh['cached_names']} names (no repeat network calls)")
    else:
        print(f"  Script transliteration: off ({bh['reason']})")
        print("                          Devanagari place names will not match the "
              "Latin admin master.")

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
    # A managed host (Render, Fly, Hugging Face Spaces, Cloud Run) assigns the
    # port at runtime and requires the process to bind 0.0.0.0 - binding
    # localhost there produces a container that passes its own health check and
    # is unreachable from outside. So the environment supplies the default and
    # an explicit flag still wins, which keeps `python3 run.py` private on a
    # laptop exactly as before.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")),
                    help="port to listen on (default: $PORT, else 8000)")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="interface to bind (default: $HOST, else 127.0.0.1)")
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
