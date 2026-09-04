"""
Pluggable OCR / text-extraction layer.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Three extraction paths are attempted in order of reliability:

  1. NATIVE PDF TEXT LAYER (PyMuPDF)
     Highest confidence. Many DILRMP-era records are digital PDFs that already
     carry a text layer, so running OCR on them would only add noise.

  2. TESSERACT OCR (pytesseract + tesseract binary, hin+eng)
     Used for scanned images and image-only PDFs. Word-level confidences are
     read straight out of Tesseract's TSV output.

  3. DEGRADED MODE
     If no OCR engine is installed, image quality assessment still runs and the
     document is queued, but the operator is told plainly that no OCR engine is
     available. Text is never fabricated.

Design note: the pipeline deliberately reports its own uncertainty. Every line
carries a confidence value and every document carries quality metrics, because
the downstream verification workflow is only useful if it can tell a clean
record from a doubtful one.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Optional dependency probing. Nothing here is required for the app to boot.
# --------------------------------------------------------------------------

def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


_pymupdf = _try_import("pymupdf") or _try_import("fitz")
_cv2 = _try_import("cv2")
_numpy = _try_import("numpy")


def tesseract_available() -> bool:
    """True only if BOTH the python wrapper and the native binary exist."""
    if _try_import("pytesseract") is None:
        return False
    return shutil.which("tesseract") is not None


def tesseract_languages() -> List[str]:
    if shutil.which("tesseract") is None:
        return []
    try:
        out = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        return [l.strip() for l in out.splitlines()[1:] if l.strip()]
    except Exception:
        return []


def capabilities() -> dict:
    """Reported to the UI so the demo is always honest about what is running."""
    langs = tesseract_languages()
    return {
        "pdf_text_layer": _pymupdf is not None,
        "image_preprocessing": _cv2 is not None,
        "tesseract": tesseract_available(),
        "tesseract_languages": langs,
        "indic_ocr": any(l in langs for l in ("hin", "mar", "ben", "tam", "tel", "kan", "guj", "pan", "ori")),
    }


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Line:
    """One recognised line of text with provenance and confidence."""
    text: str
    confidence: float          # 0.0 - 1.0
    page: int = 1
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    source: str = "unknown"    # pdf_text | tesseract

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


@dataclass
class ExtractionResult:
    lines: List[Line] = field(default_factory=list)
    engine: str = "none"
    page_count: int = 0
    quality: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    render_path: Optional[str] = None   # preview image shown in the verifier UI

    @property
    def full_text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def mean_confidence(self) -> float:
        scored = [l.confidence for l in self.lines if l.text.strip()]
        return round(sum(scored) / len(scored), 4) if scored else 0.0

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "page_count": self.page_count,
            "quality": self.quality,
            "warnings": self.warnings,
            "mean_confidence": self.mean_confidence,
            "line_count": len(self.lines),
            "lines": [l.to_dict() for l in self.lines],
        }


# --------------------------------------------------------------------------
# Image quality assessment + preprocessing
# --------------------------------------------------------------------------

def assess_and_preprocess(image_path: str, out_dir: str) -> Tuple[Optional[str], dict, List[str]]:
    """
    Real, measurable quality assessment. These numbers drive the 'poor image
    quality' problem named in the problem statement: instead of failing
    silently on a faded or skewed page, the system scores it and warns.

    Returns (preprocessed_path, metrics, warnings).
    """
    warnings: List[str] = []
    if _cv2 is None or _numpy is None:
        return None, {}, ["OpenCV unavailable - image preprocessing skipped."]

    cv2, np = _cv2, _numpy
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, {}, ["Unreadable image file."]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # --- Sharpness: variance of Laplacian. Low value => blurred/soft scan.
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # --- Contrast: std-dev of intensities. Low value => faded ink.
    contrast = float(gray.std())

    # --- Ink coverage: proportion of dark pixels after Otsu.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float((otsu > 0).sum()) / float(h * w)

    # --- Skew: dominant text angle via minAreaRect over ink pixels.
    skew = 0.0
    coords = np.column_stack(np.where(otsu > 0))
    if coords.shape[0] > 100:
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        skew = float(angle)

    # --- Deskew + denoise + adaptive threshold (helps Tesseract materially).
    work = gray
    if abs(skew) > 0.4:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew, 1.0)
        work = cv2.warpAffine(work, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    work = cv2.fastNlMeansDenoising(work, None, 9, 7, 21)
    work = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 11)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, base + "__preprocessed.png")
    cv2.imwrite(out_path, work)

    # --- Composite 0-100 legibility score.
    sharp_c = min(1.0, blur_score / 350.0)
    contrast_c = min(1.0, contrast / 70.0)
    skew_c = max(0.0, 1.0 - abs(skew) / 12.0)
    ink_c = 1.0 - min(1.0, abs(ink_ratio - 0.10) / 0.25)
    legibility = 100.0 * (0.35 * sharp_c + 0.30 * contrast_c + 0.20 * skew_c + 0.15 * ink_c)

    if blur_score < 80:
        warnings.append("Low sharpness - page appears blurred or softly scanned.")
    if contrast < 30:
        warnings.append("Low contrast - ink may be faded.")
    if abs(skew) > 3:
        warnings.append(f"Page skewed by {skew:.1f} deg - auto-deskew applied.")
    if ink_ratio < 0.01:
        warnings.append("Very little ink detected - page may be blank or washed out.")

    metrics = {
        "width": w,
        "height": h,
        "sharpness": round(blur_score, 2),
        "contrast": round(contrast, 2),
        "skew_deg": round(skew, 2),
        "ink_ratio": round(ink_ratio, 4),
        "legibility_score": round(legibility, 1),
    }
    return out_path, metrics, warnings


# --------------------------------------------------------------------------
# Extraction paths
# --------------------------------------------------------------------------

def _extract_pdf_text_layer(path: str) -> Optional[ExtractionResult]:
    """Path 1: native PDF text. Returns None if the PDF has no usable text."""
    if _pymupdf is None:
        return None
    try:
        doc = _pymupdf.open(path)
    except Exception:
        return None

    lines: List[Line] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        blocks = page.get_text("dict").get("blocks", [])

        # PyMuPDF returns each drawn text fragment separately, so a form row
        # like "Khasra Number : 237/4" arrives as three unrelated fragments.
        # Land records are overwhelmingly label-value rows on one baseline, so
        # fragments are regrouped by baseline before any parsing happens.
        # Without this, every label would be orphaned from its own value.
        frags = []
        for blk in blocks:
            for ln in blk.get("lines", []):
                text = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = ln.get("bbox", (0, 0, 0, 0))
                frags.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})

        frags.sort(key=lambda f: (round(f["y0"], 1), f["x0"]))
        rows: List[List[dict]] = []
        for f in frags:
            placed = False
            for row in rows:
                # Same baseline if the vertical centres sit within ~55% of the
                # fragment height. Tolerant enough for the small baseline
                # jitter in scanned-then-OCRed forms, tight enough not to weld
                # adjacent rows together.
                h = max(1.0, min(f["y1"] - f["y0"], row[0]["y1"] - row[0]["y0"]))
                mid_f = (f["y0"] + f["y1"]) / 2.0
                mid_r = (row[0]["y0"] + row[0]["y1"]) / 2.0
                if abs(mid_f - mid_r) <= h * 0.55:
                    row.append(f)
                    placed = True
                    break
            if not placed:
                rows.append([f])

        for row in rows:
            row.sort(key=lambda f: f["x0"])
            text = " ".join(f["text"] for f in row)
            text = " ".join(text.split())
            if not text:
                continue
            lines.append(Line(
                text=text,
                confidence=0.99,      # embedded text is authoritative
                page=pno + 1,
                bbox=(min(f["x0"] for f in row), min(f["y0"] for f in row),
                      max(f["x1"] for f in row), max(f["y1"] for f in row)),
                source="pdf_text",
            ))

    page_count = doc.page_count
    if len("".join(l.text for l in lines).strip()) < 40:
        doc.close()
        return None   # image-only PDF -> fall through to OCR

    result = ExtractionResult(
        lines=lines,
        engine="pdf_text_layer",
        page_count=page_count,
        quality={"legibility_score": 100.0, "note": "Native digital text - no OCR required."},
    )
    try:
        pix = doc.load_page(0).get_pixmap(dpi=110)
        prev = os.path.join(tempfile.gettempdir(), os.path.basename(path) + ".preview.png")
        pix.save(prev)
        result.render_path = prev
    except Exception:
        pass
    doc.close()
    return result


def _pdf_to_images(path: str, out_dir: str, dpi: int = 220) -> List[str]:
    if _pymupdf is None:
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    try:
        doc = _pymupdf.open(path)
    except Exception:
        return []
    for pno in range(doc.page_count):
        pix = doc.load_page(pno).get_pixmap(dpi=dpi)
        p = os.path.join(out_dir, f"page_{pno + 1}.png")
        pix.save(p)
        paths.append(p)
    doc.close()
    return paths


def _extract_tesseract(image_paths: List[str], out_dir: str,
                       languages: str = "hin+eng") -> ExtractionResult:
    """Path 2: real Tesseract OCR with word-level confidence aggregation."""
    import pytesseract  # safe: caller checked availability
    from collections import defaultdict

    avail = tesseract_languages()
    requested = [l for l in languages.split("+") if l]
    usable = [l for l in requested if l in avail] or (["eng"] if "eng" in avail else [])
    lang_arg = "+".join(usable) if usable else "eng"

    result = ExtractionResult(engine=f"tesseract:{lang_arg}", page_count=len(image_paths))
    missing = [l for l in requested if l not in avail]
    if missing:
        result.warnings.append(
            "Tesseract language pack(s) not installed: " + ", ".join(missing)
            + ". Install tesseract-langpack-hin for Devanagari records."
        )

    agg_quality: List[dict] = []
    for idx, img_path in enumerate(image_paths, start=1):
        pre_path, metrics, warns = assess_and_preprocess(img_path, out_dir)
        if metrics:
            agg_quality.append(metrics)
        result.warnings.extend(f"p{idx}: {w}" for w in warns)
        target = pre_path or img_path
        if idx == 1:
            result.render_path = img_path

        try:
            tsv = pytesseract.image_to_data(
                target, lang=lang_arg,
                config="--oem 1 --psm 6",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            result.warnings.append(f"p{idx}: OCR failed - {exc}")
            continue

        # Group word boxes into lines using Tesseract's own line indices.
        grouped = defaultdict(list)
        n = len(tsv.get("text", []))
        for i in range(n):
            word = (tsv["text"][i] or "").strip()
            if not word:
                continue
            key = (tsv["block_num"][i], tsv["par_num"][i], tsv["line_num"][i])
            grouped[key].append({
                "text": word,
                "conf": max(0.0, float(tsv["conf"][i])) / 100.0,
                "left": tsv["left"][i], "top": tsv["top"][i],
                "width": tsv["width"][i], "height": tsv["height"][i],
            })

        for key in sorted(grouped.keys()):
            words = grouped[key]
            text = " ".join(w["text"] for w in words)
            confs = [w["conf"] for w in words]
            # Geometric-leaning mean: one bad word should drag the line down.
            conf = math.exp(sum(math.log(max(c, 0.01)) for c in confs) / len(confs))
            x0 = min(w["left"] for w in words)
            y0 = min(w["top"] for w in words)
            x1 = max(w["left"] + w["width"] for w in words)
            y1 = max(w["top"] + w["height"] for w in words)
            result.lines.append(Line(text=text, confidence=round(conf, 4),
                                     page=idx, bbox=(x0, y0, x1, y1),
                                     source="tesseract"))

    if agg_quality:
        result.quality = {
            "legibility_score": round(sum(q["legibility_score"] for q in agg_quality) / len(agg_quality), 1),
            "pages": agg_quality,
        }
    return result


def extract(path: str, work_dir: Optional[str] = None,
            languages: str = "hin+eng") -> ExtractionResult:
    """
    Main entry point. Chooses the best available extraction path for `path`.
    """
    work_dir = work_dir or tempfile.mkdtemp(prefix="lrdv_")
    os.makedirs(work_dir, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        native = _extract_pdf_text_layer(path)
        if native is not None:
            return native
        images = _pdf_to_images(path, os.path.join(work_dir, "pages"))
        if not images:
            return ExtractionResult(engine="none", warnings=[
                "PDF could not be rasterised (PyMuPDF unavailable)."])
        if tesseract_available():
            return _extract_tesseract(images, work_dir, languages)
        return _degraded(images, work_dir)

    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        if tesseract_available():
            return _extract_tesseract([path], work_dir, languages)
        return _degraded([path], work_dir)

    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read()
        lines = [Line(text=t.strip(), confidence=0.99, source="pdf_text")
                 for t in body.splitlines() if t.strip()]
        return ExtractionResult(lines=lines, engine="plain_text", page_count=1,
                                quality={"legibility_score": 100.0})

    return ExtractionResult(engine="none", warnings=[f"Unsupported file type: {ext}"])


def _degraded(image_paths: List[str], work_dir: str) -> ExtractionResult:
    """
    Path 3. No OCR engine installed. We still do the honest work we can:
    assess quality, preprocess, and queue the document for manual entry.
    We do NOT invent text.
    """
    result = ExtractionResult(engine="degraded_no_ocr", page_count=len(image_paths))
    result.warnings.append(
        "No OCR engine detected. Install Tesseract to enable automatic text "
        "extraction (see README). Document queued for manual entry."
    )
    q = []
    for idx, p in enumerate(image_paths, start=1):
        pre, metrics, warns = assess_and_preprocess(p, work_dir)
        if idx == 1:
            result.render_path = p
        if metrics:
            q.append(metrics)
        result.warnings.extend(f"p{idx}: {w}" for w in warns)
    if q:
        result.quality = {
            "legibility_score": round(sum(x["legibility_score"] for x in q) / len(q), 1),
            "pages": q,
        }
    return result
