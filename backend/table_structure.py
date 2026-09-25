"""
Document layout / table-structure recognition via PaddleOCR's PP-StructureV3
pipeline.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

OPTIONAL, AND HEAVIER THAN EVERY OTHER OPTIONAL DEPENDENCY IN THIS PROJECT.
PyMuPDF, OpenCV, pytesseract, scikit-learn and spaCy are all small, fast,
CPU-friendly installs that degrade to "skip one feature" when absent.
PP-StructureV3 is not that: it pulls in the PaddlePaddle deep-learning
framework and downloads several hundred MB of model weights (seven separate
sub-models: layout detection, document/textline orientation, OCR detection
and recognition, table classification and cell detection, formula
recognition) the first time it runs. It is therefore never on the critical
path: ocr_engine.py's own three-tier ladder (native PDF text / Tesseract /
degraded) runs first and unconditionally, and this module only ever *adds*
a supplementary structure report on top of that, never gates or slows
anything else when it is unavailable.

MEASURED, NOT ESTIMATED: on this project's real reference machine (no GPU),
a single page took roughly 20 minutes to process end-to-end with the
pipeline correctly configured. That is not a typo and not a worst case -
it is what one real run against one real bundled sample scan actually took.
This is far beyond "slow but usable" for anything resembling a live demo or
an interactive review workflow; it only makes sense today as an offline,
batch, non-blocking annotation step, if used at all on CPU-only hardware.
GPU acceleration is the documented way PaddleOCR expects this pipeline to
be run at a usable speed, and this machine has none.

REAL COMPATIBILITY CONSTRAINT FOUND WHILE WIRING THIS IN: PaddlePaddle does
not currently support Python 3.13+ (checked against the project's own system
interpreter, Python 3.14 - `pip install paddleocr` fails outright there with
no matching distribution, it does not "install but silently misbehave"). A
Python 3.8-3.12 environment is required. This is exactly the kind of gap the
`run.py --check` capability report exists to surface honestly rather than
have a judge or teammate discover it as an unexplained ImportError.

WHAT THIS ADDS: per-page layout region labels (table / text / image / seal /
etc.) and, for detected tables, their structure as HTML - read off the
*rendered page image*, independent of and supplementary to the existing
rule-based field_extractor.py. It never writes into extracted field values
and never affects the validation decision; it is reported purely as
informational issues a reviewer can read, the same tier as STAMP_PAPER_DETECTED
in document_authenticity.py.

The PP-StructureV3 result object's exact attribute surface has changed
between PaddleOCR releases and is not part of this project's control, so
every access into it is defensive (dict-or-attribute, try/except around the
whole call) - a shape this code does not recognise degrades to "structure
analysis ran but found nothing to report", never a crash.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


_paddleocr = _try_import("paddleocr")
PADDLE_STRUCTURE_AVAILABLE = _paddleocr is not None

_PYTHON_TOO_NEW = None
if not PADDLE_STRUCTURE_AVAILABLE:
    import sys
    _PYTHON_TOO_NEW = sys.version_info >= (3, 13)

_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_load_failed = False


def _get_pipeline():
    """Lazily construct and cache the PP-StructureV3 pipeline (loads model
    weights from disk/network the first time - expensive, so this must only
    ever happen once per process, and only if a page actually needs it)."""
    global _pipeline, _pipeline_load_failed
    if _pipeline is not None or _pipeline_load_failed:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None or _pipeline_load_failed:
            return _pipeline
        try:
            from paddleocr import PPStructureV3
            # enable_mkldnn=False is required, not optional, on this
            # PaddlePaddle build: the default (mkldnn/oneDNN CPU backend)
            # crashed outright on real inference here with
            # "NotImplementedError: ConvertPirAttribute2RuntimeAttribute
            # not support [pir::ArrayAttribute<pir::DoubleAttribute>]" - a
            # known PaddlePaddle 3.3.0+ CPU/oneDNN regression (see
            # PaddlePaddle/Paddle#77340), not something this project can fix.
            # The kwarg itself is not guaranteed to exist across PaddleOCR
            # versions, so fall back to the plain constructor if it is
            # rejected rather than treating that as unavailability.
            try:
                _pipeline = PPStructureV3(enable_mkldnn=False)
            except TypeError:
                _pipeline = PPStructureV3()
        except Exception:
            _pipeline_load_failed = True
            _pipeline = None
    return _pipeline


@dataclass
class StructureResult:
    available: bool
    engine: str = "pp_structure_v3"
    table_count: int = 0
    layout_labels: Dict[str, int] = field(default_factory=dict)
    tables_html: List[str] = field(default_factory=list)
    markdown_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _markdown_text(res) -> str:
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        return md.get("markdown_texts") or md.get("markdown") or ""
    if isinstance(md, str):
        return md
    return ""


def _layout_label_counts(res) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    layout = getattr(res, "layout_det_res", None)
    if layout is None:
        return counts
    boxes = layout.get("boxes") if isinstance(layout, dict) else getattr(layout, "boxes", None)
    for box in boxes or []:
        label = box.get("label") if isinstance(box, dict) else getattr(box, "label", None)
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def analyze(image_path: str, work_dir: str) -> StructureResult:
    """
    Run PP-StructureV3 on one rendered page image. Returns available=False
    (never raises) if the library is missing, failed to load, or the page
    could not be processed - the same "report, never fabricate" discipline
    as the rest of this project.
    """
    if not PADDLE_STRUCTURE_AVAILABLE:
        return StructureResult(available=False)

    pipeline = _get_pipeline()
    if pipeline is None:
        return StructureResult(available=False, warnings=[
            "PP-StructureV3 pipeline failed to initialise (model weights "
            "missing and no network to fetch them, or an incompatible "
            "PaddlePaddle install)."
        ])

    try:
        outputs = list(pipeline.predict(image_path))
    except Exception as exc:
        return StructureResult(available=False, warnings=[
            f"PP-StructureV3 inference failed: {exc}"
        ])

    if not outputs:
        return StructureResult(available=True, engine="pp_structure_v3")

    res = outputs[0]
    markdown_text = _markdown_text(res)
    table_count = len(re.findall(r"<table[\s>]", markdown_text, flags=re.IGNORECASE))
    tables_html = re.findall(r"<table[\s\S]*?</table>", markdown_text, flags=re.IGNORECASE)
    layout_labels = _layout_label_counts(res)
    if table_count == 0 and "table" in layout_labels:
        table_count = layout_labels["table"]

    markdown_path = None
    if markdown_text.strip():
        try:
            os.makedirs(work_dir, exist_ok=True)
            markdown_path = os.path.join(work_dir, "structure.md")
            with open(markdown_path, "w", encoding="utf-8") as fh:
                fh.write(markdown_text)
        except Exception:
            markdown_path = None

    return StructureResult(
        available=True,
        engine="pp_structure_v3",
        table_count=table_count,
        layout_labels=layout_labels,
        tables_html=tables_html,
        markdown_path=markdown_path,
    )
