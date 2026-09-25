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

import hashlib
import math
import re
from difflib import SequenceMatcher
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
    """
    Import a module by name, or None if it is unavailable.

    importlib.import_module, NOT __import__: for a dotted name __import__
    returns the TOP-LEVEL package rather than the submodule asked for, so
    _try_import("skimage.feature") handed back plain `skimage` and every
    attribute lookup on it failed. That mistake shipped once already and was
    invisible, because the caller caught the resulting AttributeError and
    quietly fell back - the scikit-image skew estimator was dead in the
    pipeline while measuring six points of recall on the bench.
    """
    import importlib
    try:
        return importlib.import_module(name)
    except Exception:
        return None


_pymupdf = _try_import("pymupdf") or _try_import("fitz")
_cv2 = _try_import("cv2")
_numpy = _try_import("numpy")
_skimage_filters = _try_import("skimage.filters")
_skimage_transform = _try_import("skimage.transform")
_skimage_feature = _try_import("skimage.feature")
_cnn_denoiser = _try_import("cnn_denoiser")
_handwriting = _try_import("handwriting")
# Handwriting RECOGNITION, as distinct from detection above. Optional
# and refuses non-Latin scripts before it runs - see trocr_htr.py.
_trocr = _try_import("trocr_htr")

def _has(module, attr: str) -> bool:
    """
    Does `module` really provide `attr`?

    Checked by ATTRIBUTE, not just by import success. A module object that
    lacks the functions needed is not availability, and treating it as such is
    how the dead-import bug above stayed hidden: every page silently fell back
    instead of the capability reporting itself as missing once.

    Wrapped in try/except because a bare hasattr() here is not safe.
    scikit-image uses lazy_loader, so the attribute access is what actually
    performs the import - and if that import fails, hasattr RAISES rather
    than returning False. Measured: this machine's Application Control policy
    began blocking scipy's compiled DLLs mid-project (_ufuncs_cxx,
    _ellip_harm_2 - the name varies by run), and because the check sat at
    module scope the exception propagated out of `import ocr_engine`. An
    OPTIONAL dependency being unavailable then took down the entire pipeline,
    which is the precise opposite of what this project's degradation contract
    promises.
    """
    if module is None:
        return False
    try:
        return hasattr(module, attr)
    except Exception:
        return False


SKIMAGE_AVAILABLE = (
    _has(_skimage_feature, "canny")
    and _has(_skimage_transform, "probabilistic_hough_line")
)


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
    # Handwriting verdict for this line, when it could be judged. None means
    # "not assessed" (no profile installed, or the crop was too small) and
    # must NOT be read as "this line is printed" - see backend/handwriting.py.
    handwriting: Optional[dict] = None
    # Handwriting-recogniser result, when one was attempted. Carries BOTH
    # readings: replacing Tesseract's text without keeping it would leave a
    # verifier comparing the new reading against nothing.
    htr: Optional[dict] = None

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

# The legibility-score band that gates how much a page's raw OCR confidence
# is trusted downstream. A word Tesseract reports as 95% confident is still
# only as trustworthy as the image it was read from; below QUALITY_GATE_POOR
# that confidence is discounted heavily rather than taken at face value.
QUALITY_GATE_OK = 62.0
QUALITY_GATE_POOR = 32.0
QUALITY_GATE_MIN_MULTIPLIER = 0.55


def quality_gate_multiplier(legibility_score: float) -> float:
    """Confidence multiplier implied by a page's legibility score (0-100)."""
    if legibility_score >= QUALITY_GATE_OK:
        return 1.0
    if legibility_score <= QUALITY_GATE_POOR:
        return QUALITY_GATE_MIN_MULTIPLIER
    frac = (legibility_score - QUALITY_GATE_POOR) / (QUALITY_GATE_OK - QUALITY_GATE_POOR)
    return QUALITY_GATE_MIN_MULTIPLIER + frac * (1.0 - QUALITY_GATE_MIN_MULTIPLIER)


def quality_gate_label(legibility_score: float) -> str:
    if legibility_score >= QUALITY_GATE_OK:
        return "ok"
    if legibility_score <= QUALITY_GATE_POOR:
        return "poor"
    return "marginal"


# Windows refuses to open a path of 260 characters or more through the normal
# API, and both OpenCV and the Tesseract binary go through that API. This
# project's own storage/work directory is already 209 characters deep on the
# machine it was developed on, so "<work>/<sha16>/<name>__preprocessed.png"
# lands at ~280 and neither tool can touch it. cv2.imwrite then fails by
# returning False (it does not raise), Tesseract is handed a path to a file
# that does not exist, and every scanned image silently extracts zero text
# while still reporting engine=tesseract. Found by end-to-end testing, not by
# the unit suite - the same MAX_PATH class of bug that once aborted api_seed's
# whole batch, in a different place.
#
# Writing to the system temp directory when the natural path is too long
# keeps the file readable by Tesseract as well as writable by OpenCV, which a
# \\?\ extended-length prefix would not: that prefix fixes the write and
# leaves the external Tesseract process still unable to read it.
_MAX_USABLE_PATH = 250


def _preprocessed_target(out_dir: str, base: str) -> str:
    natural = os.path.join(out_dir, base + "__preprocessed.png")
    if os.name != "nt" or len(os.path.abspath(natural)) < _MAX_USABLE_PATH:
        return natural
    tag = hashlib.sha1(os.path.abspath(out_dir).encode("utf-8")).hexdigest()[:10]
    return os.path.join(tempfile.gettempdir(), f"lrdv_{tag}_{base[:40]}__pre.png")


# Gates on the Hough skew estimate. Both exist because the estimator's
# failure mode is not "no answer" but "a confident wrong answer": on the
# `faded` sample - which make_scans does NOT rotate at all - it reported
# -2.33 degrees, and acting on that would tilt a perfectly straight page.
#
# What separates the two cases is not the angle, it is the EVIDENCE. A
# genuinely skewed page gave 346 near-horizontal segments clustered within a
# couple of degrees; the straight page gave 74, scattered from -5 to 0 with
# no mode. So a minimum count and a minimum concentration are both required,
# and failing either means "no estimate" rather than a weak one.
_MIN_HOUGH_SEGMENTS = 60
_HOUGH_CLUSTER_WINDOW = 1.0      # degrees either side of the median
_HOUGH_MIN_CLUSTERED = 0.5       # fraction that must fall inside it

# Below this, a page is left alone even when the skew estimate is trusted.
# Measured on 32 real pages against a synthetic corpus skewed 2.4-3.6
# degrees by construction; see the long comment at the rotation site in
# assess_and_preprocess() for the numbers and why raising it costs nothing.
MIN_DESKEW_DEGREES = 2.0


def _should_deskew(angle: Optional[float]) -> bool:
    """
    Whether a page is crooked enough that rotating it is worth the resampling.

    A predicate rather than an inline comparison so the threshold can be
    tested against the measured angle bands directly - real pages cluster at
    1.22-1.88 degrees and must NOT be rotated, the synthetic corpus reads
    2.291 and must be.
    """
    return angle is not None and abs(angle) > MIN_DESKEW_DEGREES


def _skew_minarearect(otsu_inv) -> float:
    """
    Angle of the minimum-area rectangle around every ink pixel on the page.

    Cheap, dependency-free, and not very good: the ink cloud of a printed form
    is dominated by its table borders and block layout, so the fitted
    rectangle tracks the shape of the FORM rather than the direction of the
    TEXT. Kept as the fallback for when scikit-image is unavailable.
    """
    np = _numpy
    coords = np.column_stack(np.where(otsu_inv > 0))
    if coords.shape[0] <= 100:
        return 0.0
    angle = _cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return float(angle)


def _skew_hough(gray) -> Optional[float]:
    """
    Skew from the dominant near-horizontal line direction, via a
    probabilistic Hough transform over Canny edges.

    Text baselines and printed rules are both near-horizontal, so the median
    angle of the long near-horizontal segments IS the page skew. Measured on
    the sample corpus this is worth about six points of field recall over the
    minimum-area-rectangle estimate above (34.5% -> 40.4%), the single largest
    preprocessing win found - which makes sense, because a page rotated by
    even two degrees smears every character box Tesseract tries to cut.

    Returns None when scikit-image is missing or no usable line is found, so
    the caller falls back rather than trusting a zero.
    """
    if not SKIMAGE_AVAILABLE:
        return None
    np = _numpy
    try:
        edges = _skimage_feature.canny(gray, sigma=2.0)
        segments = _skimage_transform.probabilistic_hough_line(
            edges, threshold=10,
            # line_length decides whether this works AT ALL. It was
            # width//12 (117px on a 1406px scan) and found ZERO segments on
            # every page, because a Canny edge map of TEXT holds no straight
            # run that long - a baseline is a dotted line of character
            # bottoms, not a ruled line. At width//40 the same page yields
            # ~260 near-horizontal segments and a median within 0.2 degrees
            # of the true skew.
            line_length=max(30, gray.shape[1] // 40), line_gap=4)
    except Exception:
        return None
    if not segments:
        return None
    angles = []
    for (x0, y0), (x1, y1) in segments:
        if x1 == x0:
            continue
        angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        if abs(angle) <= 20.0:
            angles.append(angle)
    if len(angles) < _MIN_HOUGH_SEGMENTS:
        return None
    median = float(np.median(angles))
    clustered = sum(1 for a in angles if abs(a - median) <= _HOUGH_CLUSTER_WINDOW)
    if clustered / len(angles) < _HOUGH_MIN_CLUSTERED:
        return None
    return median


def assess_and_preprocess(image_path: str, out_dir: str) -> Tuple[Optional[str], dict, List[str]]:
    """
    Real, measurable quality assessment and restoration. These numbers drive
    the 'poor image quality' problem named in the problem statement: instead
    of failing silently on a faded, skewed, shadowed or bled-through page,
    the system scores it, restores what can be restored, and warns about
    what it could not.

    Restoration order matters. Illumination is corrected first, because an
    uneven light source otherwise reads as "faded ink" or "low contrast" to
    every metric measured afterwards. Bleed-through suppression and rule-line
    removal happen last, on the binarised ink mask, because both are really
    the same question - which dark pixels are genuine character strokes and
    which are structural or ghost noise - and that question is only
    meaningful once illumination has stopped distorting what "dark" means.

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

    # --- Illumination correction: estimate the page's background lighting
    # with a wide morphological closing (large enough to erase text strokes,
    # small enough to keep the slow lighting gradient), then divide it out.
    bg_ksize = max(15, (min(h, w) // 20) | 1)   # odd, scales with page size
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_ksize, bg_ksize))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel)
    illumination_variation = float(background.std())

    # --- Sharpness: variance of Laplacian, measured on the raw capture
    # *before* illumination correction. Blur is a property of the capture
    # (focus, hand shake), not of lighting, and dividing by the background
    # estimate amplifies pixel-level noise - measuring sharpness after that
    # division would let noise masquerade as sharpness on a genuinely blurred
    # page. A 3x3 median pre-filter is applied first for the same reason: raw
    # sensor/scan noise is itself high-frequency, so an untouched Laplacian
    # reads a noisy-but-blurred page as "sharp" - the exact failure mode this
    # metric exists to catch. The median filter knocks down that noise while
    # leaving genuine edges intact, so remaining Laplacian energy reflects
    # actual focus rather than grain.
    blur_score = float(cv2.Laplacian(cv2.medianBlur(gray, 3), cv2.CV_64F).var())

    gray = cv2.divide(gray, background, scale=255.0)

    # --- Contrast: std-dev of intensities, measured *after* illumination
    # correction, since a shadow legitimately depresses this and correcting
    # for it before judging "is the ink actually faded" is the point of the
    # correction. Low value => faded ink.
    contrast = float(gray.std())

    # --- Ink coverage + skew, from a first-pass global threshold.
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float((otsu_inv > 0).sum()) / float(h * w)

    # MEASURING the skew and CORRECTING it are separate decisions.
    #
    # The minimum-area-rectangle estimator always returns an answer, so it
    # stays as the fallback for the reported metric. It must NOT drive the
    # rotation: measured over the 33-scan corpus, deskewing on its output
    # cost about six points of field recall (40.4% -> 34.5%), because on a
    # printed form it tracks the shape of the TABLE rather than the direction
    # of the TEXT and so cheerfully rotates a straight page. Gated Hough
    # deskew measured exactly zero difference either way (41.1% / 51.0%,
    # identical correct/wrong/missed counts) - the corpus skew is only 2.4 to
    # 3.6 degrees, which Tesseract's own layout analysis absorbs unaided.
    #
    # So: rotate only on a Hough estimate that passed its evidence gate, and
    # otherwise leave the pixels alone while still reporting the skew we
    # believe is present. Correcting on a guess is how the old code lost
    # recall; declining to report is how a quality metric becomes useless.
    hough = _skew_hough(gray)
    skew = hough if hough is not None else _skew_minarearect(otsu_inv)
    skew_correction = hough if hough is not None else 0.0
    skew_method = "hough" if hough is not None else "minarearect"

    # --- Deskew + denoise (helps Tesseract materially).
    #
    # The threshold is 2.0 degrees and NOT the 0.4 it used to be, because
    # rotating is not free. warpAffine with INTER_CUBIC resamples every
    # pixel on the page, which softens strokes that were sharp; below the
    # angle at which Tesseract stops coping on its own, that blur is pure
    # cost for no layout gain.
    #
    # Measured, on real documents rather than generated ones:
    #
    #   * 32 real pages (15 Bhu-Naksha reports + a 13-page notarised GPA):
    #     only 7 produce a Hough estimate at all, and EVERY one of them
    #     falls between 1.22 and 1.88 degrees - median 1.38. At the old 0.4
    #     threshold all 7 were rotated. At 2.0 none are.
    #   * The synthetic corpus is skewed 2.4 and 3.6 degrees deliberately
    #     (tools/make_samples.py), and the estimator reads -2.291 on the
    #     2.4-degree scan. So 2.0 sits inside a real gap: 1.878 below it,
    #     2.291 above it.
    #   * Nothing is given up by raising it. Gated Hough deskew measured
    #     EXACTLY zero difference on the synthetic corpus (see the comment
    #     above: 41.1% / 51.0%, identical correct/wrong/missed counts). The
    #     "six points" often quoted for the Hough estimator is the cost of
    #     minarearect deskewing that it AVOIDS, not a gain from rotating.
    #   * What it recovers is 6.6 points of field accuracy on the real
    #     corpus, which is why 84.4% was only ever reproducible with
    #     scikit-image pinned off. Visible on one field of the real GPA:
    #     rotated, its area read "MEAS. 57 5..0 . ..25." and parsed to
    #     nothing; unrotated, "MEAS. 57 SQ.YDS." parsing to 47.659 m2.
    #
    # Rotation is kept above 2.0 degrees rather than removed: no measurement
    # here covers a genuinely crooked scan (10 degrees, say), and Tesseract
    # does eventually stop absorbing skew. This declines to correct only in
    # the band where correcting was measured to hurt.
    work = gray
    if _should_deskew(skew_correction):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew_correction, 1.0)
        work = cv2.warpAffine(work, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    # --- Denoise. The trained CNN (backend/cnn_denoiser.py) replaces
    # NL-means when its weights are installed; otherwise NL-means stands.
    #
    # Measured over the 33-scan corpus, swapping the two is a wash on field
    # accuracy (40.2% -> 40.4% recall, 50.4% -> 50.3% precision - four fields
    # out of 437, which is noise). What it does move is Tesseract's own mean
    # word confidence, 72.4 -> 74.5, and that is not cosmetic here: confidence
    # decides whether a field is auto-accepted or routed to a human, so a
    # better-calibrated engine changes what reaches the review queue.
    denoised = None
    if _cnn_denoiser is not None and _cnn_denoiser.available():
        denoised = _cnn_denoiser.denoise(work)
    if denoised is not None:
        work = denoised
        denoise_method = "cnn"
    else:
        work = cv2.fastNlMeansDenoising(work, None, 9, 7, 21)
        denoise_method = "nlmeans"

    # --- Bleed-through suppression. Ghost text from the reverse side of the
    # page is systematically fainter than genuine foreground ink, so a
    # *local* adaptive threshold (which only compares a pixel to its own
    # neighbourhood) can still classify it as ink even though a *global*
    # threshold correctly reads it as background. A pixel survives only if
    # both methods agree - exactly the case a local-only or global-only
    # threshold gets wrong.
    adaptive = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 11)
    _, global_mask = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_only = int(((adaptive > 0) & (global_mask == 0)).sum())
    combined = cv2.bitwise_and(adaptive, global_mask)
    bleed_through_ratio = adaptive_only / float(h * w)

    # --- Rule-line removal. Printed-form ruling lines are long, thin and
    # perfectly straight - exactly what a long, thin morphological opening
    # isolates, and nothing a genuine character stroke matches - so they are
    # erased before OCR sees them instead of being read as stray characters
    # or fused into the words that sit against them.
    h_len = max(15, w // 25)
    v_len = max(15, h // 25)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(combined, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(combined, cv2.MORPH_OPEN, v_kernel)
    rule_lines = cv2.bitwise_or(h_lines, v_lines)
    rule_line_ratio = float((rule_lines > 0).sum()) / float(h * w)

    # --- What Tesseract actually receives: the RESTORED GREYSCALE, not the
    # ink mask built above.
    #
    # The masks are still computed, because bleed_through_ratio and
    # rule_line_ratio are genuine quality measurements that feed the
    # legibility score. But handing the binarised result to Tesseract was
    # measurably harmful. Scored over the 33-scan corpus with
    # tools/measure_accuracy.py, feeding the mask reached 28.1% field recall
    # and 38.6% precision - the WORST of seven recipes tried, and worse than
    # passing the untouched original (31.1% / 52.7%). The same chain stopping
    # at greyscale reached 40.4% / 50.4%.
    #
    # The reason is that this module's binarisation and Tesseract's are not
    # cumulative, they compete. Tesseract already runs its own adaptive
    # thresholder tuned for text, on data that still has the grey levels to
    # threshold; a mask arrives with that information spent, and every stroke
    # the AND of two thresholds dropped - thin Devanagari matras above all -
    # is gone before it can be recognised. Rule-line removal compounded it,
    # erasing the parts of characters that sat against a table border
    # (31.7% -> 30.6% in the same measurement).
    #
    # So restoration here stops at the things Tesseract cannot do for
    # itself - correcting illumination, straightening the page, removing
    # sensor noise - and the thresholding is left to the engine that is
    # better at it.

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = _preprocessed_target(out_dir, base)
    if not cv2.imwrite(out_path, work) or not os.path.exists(out_path):
        # cv2.imwrite reports failure by RETURN VALUE, not by raising. Leaving
        # that unchecked is how a write failure turned into "OCR found no
        # text": the path was handed to Tesseract regardless, Tesseract could
        # not open a file that was never created, and the document was
        # recorded as a successful OCR run that happened to find nothing -
        # precisely the silent failure README S3 says this system refuses to
        # produce. Fall back to the untouched original instead, and say so.
        warnings.append("Preprocessed image could not be written - OCR ran on "
                        "the unprocessed original, so quality restoration did "
                        "not apply to this page.")
        return None, {}, warnings

    # --- Composite 0-100 legibility score. Sharpness/contrast/ink carry most
    # of the weight because they answer the question that matters most - is
    # there legible content here at all - whereas skew, illumination and
    # bleed-through are comparatively rare defects. Weighting them equally
    # would let three "nothing wrong here" structural signals mask a page
    # that is blurred, faded or blank to the point of having no real content,
    # which defeats the point of a quality gate.
    sharp_c = min(1.0, blur_score / 350.0)
    contrast_c = min(1.0, contrast / 70.0)
    skew_c = max(0.0, 1.0 - abs(skew) / 12.0)
    ink_c = 1.0 - min(1.0, abs(ink_ratio - 0.10) / 0.25)
    illum_c = max(0.0, 1.0 - illumination_variation / 45.0)
    bleed_c = max(0.0, 1.0 - bleed_through_ratio / 0.05)
    legibility = 100.0 * (
        0.32 * sharp_c + 0.28 * contrast_c + 0.16 * ink_c
        + 0.10 * skew_c + 0.08 * illum_c + 0.06 * bleed_c
    )
    gate = quality_gate_label(legibility)

    if blur_score < 80:
        warnings.append("Low sharpness - page appears blurred or softly scanned.")
    if contrast < 30:
        warnings.append("Low contrast - ink may be faded.")
    if abs(skew) > 3:
        warnings.append(f"Page skewed by {skew:.1f} deg - auto-deskew applied.")
    if ink_ratio < 0.01:
        warnings.append("Very little ink detected - page may be blank or washed out.")
    if illumination_variation > 45:
        warnings.append("Uneven illumination/shadow detected - lighting correction applied.")
    if bleed_through_ratio > 0.01:
        warnings.append(
            f"Possible bleed-through from the reverse side suppressed "
            f"({bleed_through_ratio * 100:.1f}% of page area).")
    if rule_line_ratio > 0.005:
        warnings.append(
            f"Table/form ruling lines detected and removed before OCR "
            f"({rule_line_ratio * 100:.1f}% of page area).")
    if gate == "poor":
        warnings.append(
            f"Quality gate: legibility {legibility:.0f}/100 is below the reliability "
            "threshold - OCR confidence on this page has been discounted accordingly.")

    metrics = {
        "width": w,
        "height": h,
        "sharpness": round(blur_score, 2),
        "contrast": round(contrast, 2),
        "skew_deg": round(skew, 2),
        # Which estimator produced skew_deg. Recorded because the two differ
        # materially in accuracy, so a reviewer comparing two documents needs
        # to know whether scikit-image was present for both.
        "skew_method": skew_method,
        # The rotation actually APPLIED, which is 0.0 whenever the estimate
        # did not earn enough confidence to act on. Reported separately from
        # skew_deg so "we think the page is tilted" and "we straightened it"
        # can never be confused for one another.
        "skew_corrected_deg": round(skew_correction, 2),
        # Which denoiser ran. Two documents processed with different denoisers
        # are not strictly comparable, so this is recorded rather than assumed.
        "denoise_method": denoise_method,
        "ink_ratio": round(ink_ratio, 4),
        "illumination_variation": round(illumination_variation, 2),
        "bleed_through_ratio": round(bleed_through_ratio, 4),
        "rule_line_ratio": round(rule_line_ratio, 4),
        "legibility_score": round(legibility, 1),
        "quality_gate": gate,
        "confidence_gate_multiplier": round(quality_gate_multiplier(legibility), 3),
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


# --------------------------------------------------------------------------
# Script-aware language selection
# --------------------------------------------------------------------------
#
# An Indian land record can be in any of a dozen scripts, and Tesseract does
# NOT pick a model for itself - it reads with whatever pack it is handed. A
# Tamil page read with a Devanagari model does not fail loudly; it maps Tamil
# glyphs onto the nearest Devanagari shapes and returns confident nonsense,
# which then poisons every downstream stage. Hardcoding "hin+eng" therefore
# silently capped this system at one Indic script.
#
# The obvious fix - Tesseract's own OSD script detector - was tried first and
# MEASURED TO NOT WORK on these documents: on four real pages it reported
# "Latin" every time (and "Japanese, rotated 180 degrees" once), because
# Indian land records are bilingual and OSD picks a single dominant script
# for the whole page, which the English half wins.
#
# What does work, measured on four scripts: run the candidate packs and keep
# the one whose words Tesseract is most confident about. The correct pack won
# every case by a clear margin - Tamil 91.0 vs 63.9 next best, Telugu 88.6 vs
# 74.3, Devanagari 79.3 vs 62.6 and 78.9 vs 70.9 - and the winner's output is
# in the expected Unicode block while every loser emits Latin noise.
#
# A "try the default first and only trial if it scores badly" shortcut was
# implemented and then removed, because measuring it showed it cannot work:
# on the preprocessed images selection actually sees, right-script pages
# scored 61.3-71.1 and wrong-script pages 58.6-65.7. Those ranges OVERLAP,
# so no confidence threshold separates them. Rather than tune a number that
# the data says does not exist, every candidate is tried, every time.
#
# The cost is controlled instead by (a) trialling on a HALF-SIZE copy -
# verified to pick the same winner as full resolution, while 0.35x was too
# far and broke a case - and (b) deciding once per document rather than per
# page, since one record is overwhelmingly one script. Documents with a PDF
# text layer never reach this path at all.
DEFAULT_LANGUAGES = "hin+eng"

# ONE ENTRY PER SCRIPT, deliberately - not one per language. Confidence is a
# strong signal for "is this even the right script" (Tamil beat the next pack
# by 27 points, Telugu by 14) but a weak one for choosing between two models
# of the SAME script: mar+eng outscored hin+eng on all three Devanagari
# samples here, including two that are Hindi documents, and a more confident
# model is not necessarily a more accurate one. Claiming to pick the language
# would be reading more into these numbers than they support, so Devanagari
# is represented once, by hin+eng, and Marathi records continue to be read by
# it - correctly, because the script is what matters to the recogniser.
# How far an Indic pack must beat plain `eng` before it is trusted. Set from
# measurement, not taste - see the table in select_languages().
SCRIPT_MARGIN_OVER_ENGLISH = 10.0

SCRIPT_PACK_CANDIDATES = [
    "hin+eng",   # Devanagari - Hindi, Marathi, Sanskrit, Nepali, Konkani
    "ben+eng",   # Bengali - Bengali, Assamese
    "tam+eng",   # Tamil
    "tel+eng",   # Telugu
    "kan+eng",   # Kannada
    "guj+eng",   # Gujarati
    "pan+eng",   # Gurmukhi
    "mal+eng",   # Malayalam
    "ori+eng",   # Odia
    "urd+eng",   # Perso-Arabic
]

# Trial images are halved before selection; see above for why not smaller.
SCRIPT_TRIAL_SCALE = 0.5


def _tsv_mean_confidence(tsv: dict) -> float:
    """Mean word confidence, 0-100, ignoring empty boxes."""
    confs = [float(tsv["conf"][i]) for i in range(len(tsv.get("text", [])))
             if (tsv["text"][i] or "").strip() and float(tsv["conf"][i]) >= 0]
    return sum(confs) / len(confs) if confs else 0.0


# Page-segmentation mode is chosen per page, not fixed.
#
# psm 6 ("assume a single uniform block of text") suits the common case here,
# a one-block register page, and measured 1.1 points of field recall better
# than psm 3 across the scan corpus (91.3% vs 90.2%). But it performs no
# layout analysis, and a property paper carrying a printed form, the
# patwari's handwritten entries and a sketch map on one sheet is emphatically
# not one block. On exactly such a page psm 6 read 1 of 8 printed fields
# where psm 3 read 8 of 8 - because assess_and_preprocess strips the table
# ruling lines, and once that cue is gone psm 6 has nothing left telling it
# the cells are separate, so it welds the whole form into a single block and
# loses it.
#
# So run both and keep psm 6 unless psm 3 finds substantially more text. The
# margin is the load-bearing part. Measured over 13 pages, the multi-region
# page gave psm 3 a 60% lead in confident characters while every single-block
# scan stayed within 6% either way; a 25% bar therefore separates "this page
# has a layout psm 6 cannot see" from ordinary run-to-run variation. Without
# the bar, noise alone would flip the mode on roughly half the corpus and
# hand back the recall this is trying to protect.
_PSM_SINGLE_BLOCK = 6
_PSM_AUTO_LAYOUT = 3
_PSM_SWITCH_MARGIN = 1.25
_PSM_CONFIDENT = 50.0


def _ocr_page(path: str, lang: str, psm: int = _PSM_SINGLE_BLOCK) -> Optional[dict]:
    import pytesseract
    try:
        return pytesseract.image_to_data(
            path, lang=lang, config=f"--oem 1 --psm {psm}",
            output_type=pytesseract.Output.DICT)
    except Exception:
        return None


def _confident_chars(tsv: Optional[dict]) -> int:
    """
    Characters Tesseract was reasonably sure of.

    Length rather than word count, so a page that shattered into many
    one-character fragments cannot outscore one that read whole words - which
    is the exact failure mode being compared against.
    """
    if not tsv:
        return 0
    total = 0
    for i in range(len(tsv.get("text", []))):
        word = (tsv["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(tsv["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf >= _PSM_CONFIDENT:
            total += len(word)
    return total


def _ocr_page_best_layout(path: str, lang: str) -> Tuple[Optional[dict], int]:
    """
    OCR one page, choosing the page-segmentation mode on evidence.

    Returns the chosen TSV and the psm that produced it, so the caller can
    say which one ran rather than leaving it to be guessed.
    """
    block = _ocr_page(path, lang, psm=_PSM_SINGLE_BLOCK)
    auto = _ocr_page(path, lang, psm=_PSM_AUTO_LAYOUT)
    if block is None:
        return auto, _PSM_AUTO_LAYOUT
    if auto is None:
        return block, _PSM_SINGLE_BLOCK
    if _confident_chars(auto) > _confident_chars(block) * _PSM_SWITCH_MARGIN:
        return auto, _PSM_AUTO_LAYOUT
    return block, _PSM_SINGLE_BLOCK


def _usable_pack(pack: str, available: List[str]) -> Optional[str]:
    """Drop languages this machine does not have; None if nothing is left."""
    langs = [l for l in pack.split("+") if l in available]
    return "+".join(langs) if langs else None


def _trial_image(image_path: str, out_dir: str) -> str:
    """Half-size copy used only for script selection. Falls back to the
    original if OpenCV is unavailable or the write fails - selection is then
    slower, never wrong."""
    if _cv2 is None:
        return image_path
    try:
        img = _cv2.imread(image_path, _cv2.IMREAD_GRAYSCALE)
        if img is None:
            return image_path
        small = _cv2.resize(img, None, fx=SCRIPT_TRIAL_SCALE, fy=SCRIPT_TRIAL_SCALE,
                            interpolation=_cv2.INTER_AREA)
        base = os.path.splitext(os.path.basename(image_path))[0]
        target = _preprocessed_target(out_dir, base + "__trial")
        if _cv2.imwrite(target, small) and os.path.exists(target):
            return target
    except Exception:
        pass
    return image_path


def select_languages(image_path: str, available: List[str],
                     out_dir: Optional[str] = None,
                     default: str = DEFAULT_LANGUAGES):
    """
    Choose Tesseract language packs for a document by trialling one pack per
    script and keeping the one whose words Tesseract is most confident about.

    Selection deliberately runs on the RAW page, not the restored one: the
    restoration pipeline is tuned to make text readable, and measurably
    compresses the very confidence differences this relies on (right- and
    wrong-script scores overlap after preprocessing, but separate cleanly
    before it).

    Returns (lang_arg, diagnostics). The winning pass's text is NOT reused -
    it came from a half-size trial image, and the real OCR runs on the
    full-size restored page.
    """
    default_arg = _usable_pack(default, available) or ("eng" if "eng" in available else None)
    if default_arg is None:
        return None, {"reason": "no usable language pack installed"}

    probe = _trial_image(image_path, out_dir) if out_dir else image_path

    scores: Dict[str, float] = {}
    # `eng` is trialled as the BASELINE, not as a competitor. Without it
    # there is nothing to measure an Indic pack's contribution against, which
    # is how an Odia pack came to win a Delhi e-Stamp written in English.
    for pack in ["eng"] + list(SCRIPT_PACK_CANDIDATES):
        arg = _usable_pack(pack, available)
        if not arg or arg in scores:
            continue
        tsv = _ocr_page(probe, arg)
        if tsv is not None:
            scores[arg] = _tsv_mean_confidence(tsv)

    if not scores:
        return default_arg, {"reason": "no candidate pack produced output",
                             "trials": 0}

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_arg, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # An Indic pack must EARN its place against plain English.
    #
    # Comparing the best pack against the second-best cannot answer the
    # question that matters, because every candidate is "X+eng" and they all
    # share the English half - on a Latin page their scores cluster within a
    # point or two of each other and the winner is noise. Measured on a real
    # Delhi e-Stamp (entirely Latin, no Indic text at all), 'ori+eng' won by
    # a 1.56 margin over the next pack and proceeded to inject Odia glyphs
    # through clean English: "INDIA NON JUDICIAL" came back as
    # "ଓ ଆସନ INDIA NON JUDICIAL", "NOTARY REGISTERED" as "¦ NINE ୩୪୭୩".
    # Mean confidence for the document was 0.505 and half its lines were then
    # misflagged as handwriting because mangled text has irregular stroke
    # statistics.
    #
    # The discriminating comparison is against `eng` ALONE, which is why it
    # is now trialled as the baseline. Measured margins over eng:
    #
    #     Latin       (4 GPA pages)      min + 0.00   max + 9.49
    #     Devanagari  (all 15 reports)   min +10.89   max +37.33
    #
    # 10.0 sits in that gap. Below it the page has no meaningful Indic content
    # and English is the safer read; above it an Indic pack is genuinely
    # adding something.
    #
    # The gap is 1.4 points wide, and that thinness is the honest caveat on
    # this constant. A first attempt set it to 12.0 from only THREE Devanagari
    # samples (+15.7, +27.5, +37.3) and it cost two khasra fields on the real
    # corpus - 82.2% down to 77.8% - because four of the fifteen documents sit
    # between 10.9 and 12.9 and were wrongly gated to English, which then
    # could not read their Devanagari khasra label at all. Re-deriving it
    # across all fifteen is what produced 10.0. Anything that changes
    # preprocessing, trial-image scale or the pack list can move these scores,
    # so this number must be re-measured rather than trusted.
    #
    # NOT FIXED by this, and worth stating plainly: WHICH Indic pack wins is
    # still unreliable. On 6aa420f1 - a Devanagari document - 'pan+eng'
    # (Gurmukhi) beat 'hin+eng' by 6 points. Self-consistency cannot catch it
    # either, since a pack can only ever emit its own script plus Latin, so
    # every candidate reports 100% agreement with itself by construction.
    # Separating Devanagari from Gurmukhi needs script identification on the
    # IMAGE, and Tesseract's own OSD was already measured useless here (it
    # reported "Latin" on every page, see S5). This gate stops Indic garbage
    # reaching Latin documents; it does not stop the wrong Indic pack winning
    # on an Indic one.
    baseline = scores.get("eng")
    gated = False
    if baseline is not None and best_arg != "eng":
        if best_score - baseline < SCRIPT_MARGIN_OVER_ENGLISH:
            best_arg, gated = "eng", True

    return best_arg, {
        "reason": "english baseline retained" if gated else "selected by trial",
        "confidence": round(best_score, 2),
        "margin": round(best_score - runner_up, 2),
        "english_baseline": None if baseline is None else round(baseline, 2),
        "margin_over_english": (None if baseline is None
                                else round(ranked[0][1] - baseline, 2)),
        "required_margin": SCRIPT_MARGIN_OVER_ENGLISH,
        "gated_to_english": gated,
        "trials": len(scores),
    }


# Latin digits read by an Indic pack are the weak point of picking one
# language per page.
#
# Measured on a real Bhu-Naksha plot report (Devanagari prose, Latin numerals):
# hin+eng won script selection by a 13.54 confidence margin - correctly, it is
# the only pack that reads the prose - and then misread EVERY number on the
# page. Khata 00100 became 0000, plot 184 became 84, area 1.6350 became .6350,
# scale 1:1914 became 7:94. The same crops under plain eng came back exactly
# right. The digit '1' was the main casualty, variously read as 4, 7, W or
# dropped outright.
#
# That matters more than a typical OCR slip because khasra, khata and area are
# the mandatory identifier fields: a wrong khata does not look wrong, it looks
# like a different person's land.
#
# Neither pack alone is sufficient - eng renders 'खसरा नंबर' as 'GERI AG' - so
# the Indic pass keeps the prose and a second eng pass is consulted ONLY for
# the digit runs. The guards below exist so that a second opinion can never
# turn a right answer into a wrong one: the two readings must describe the
# same line (vertical overlap plus a matching non-digit skeleton) and must
# agree on how many separate numbers that line contains.
# Matching the two readings is done on GEOMETRY, not on text similarity. An
# earlier version also required the non-digit remainder of the two lines to
# look alike, which sounds prudent and is exactly wrong here: on the line
# 'खसरा नंबर : 184' the Devanagari pass reads the label as 'खसरा नंबर' and the
# English pass reads the same pixels as 'GERI AG', so the skeletons never
# match and the guard blocked precisely the cross-script repairs it exists to
# enable. The two passes read the SAME IMAGE, so box agreement on both axes
# already establishes they are looking at the same line.
_DIGIT_RUN = re.compile(r"\d+")
_DIGIT_MIN_OVERLAP = 0.6
_DIGIT_MIN_X_IOU = 0.5


def _digit_runs(text: str) -> List[str]:
    return _DIGIT_RUN.findall(text or "")


def _digit_skeleton(text: str) -> str:
    """The line with every digit run collapsed, so two readings can be
    compared on their non-numeric content alone."""
    return _DIGIT_RUN.sub("#", (text or "")).casefold()


def _vertical_overlap(a, b) -> float:
    """Fraction of the shorter box's height that the two boxes share."""
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    if bottom <= top:
        return 0.0
    shorter = min(a[3] - a[1], b[3] - b[1])
    return (bottom - top) / shorter if shorter > 0 else 0.0


def _horizontal_iou(a, b) -> float:
    """Intersection over union of the two boxes' horizontal extents."""
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    inter = max(0.0, right - left)
    union = max(a[2], b[2]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _repair_digits(primary: List[Line], digits: List[Line]) -> int:
    """
    Replace each line's digit runs with a plain-English pass's reading.

    Returns the number of lines changed. Anything that does not clear every
    guard is left exactly as the Indic pass read it - declining to repair is
    always safe here, while a mis-aligned splice would fabricate an
    identifier that was never on the page.
    """
    if not primary or not digits:
        return 0
    changed = 0
    for line in primary:
        runs = _digit_runs(line.text)
        if not runs:
            continue
        best = None
        best_overlap = _DIGIT_MIN_OVERLAP
        for cand in digits:
            if cand.page != line.page:
                continue
            overlap = _vertical_overlap(line.bbox, cand.bbox)
            if overlap < best_overlap:
                continue
            if len(_digit_runs(cand.text)) != len(runs):
                continue
            # Same physical line, or a different one that happens to sit at
            # the same height? Horizontal extent decides.
            if _horizontal_iou(line.bbox, cand.bbox) < _DIGIT_MIN_X_IOU:
                continue
            best, best_overlap = cand, overlap
        if best is None:
            continue
        replacement = _digit_runs(best.text)
        if replacement == runs:
            continue
        it = iter(replacement)
        line.text = _DIGIT_RUN.sub(lambda _m: next(it), line.text)
        changed += 1
    return changed


def _lines_from_tsv(tsv: dict, page: int, gate_mult: float = 1.0) -> List[Line]:
    """Group Tesseract word boxes into lines using its own line indices."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for i in range(len(tsv.get("text", []))):
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

    lines: List[Line] = []
    for key in sorted(grouped.keys()):
        words = grouped[key]
        text = " ".join(w["text"] for w in words)
        confs = [w["conf"] for w in words]
        # Geometric-leaning mean: one bad word should drag the line down.
        conf = math.exp(sum(math.log(max(c, 0.01)) for c in confs) / len(confs))
        # Quality gate: a word Tesseract reports as confident is only as
        # trustworthy as the page it was read from.
        conf = min(1.0, conf * gate_mult)
        x0 = min(w["left"] for w in words)
        y0 = min(w["top"] for w in words)
        x1 = max(w["left"] + w["width"] for w in words)
        y1 = max(w["top"] + w["height"] for w in words)
        lines.append(Line(text=text, confidence=round(conf, 4),
                          page=page, bbox=(x0, y0, x1, y1), source="tesseract"))
    return lines


def _extract_tesseract(image_paths: List[str], out_dir: str,
                       languages: Optional[str] = None) -> ExtractionResult:
    """
    Path 2: real Tesseract OCR with word-level confidence aggregation.

    `languages=None` selects packs per document by trial (see
    select_languages). Passing an explicit value forces it and skips
    selection entirely, which is what the tests and any caller that already
    knows the script should do.
    """
    from collections import defaultdict

    avail = tesseract_languages()
    auto = languages is None
    requested = [l for l in (languages or DEFAULT_LANGUAGES).split("+") if l]
    usable = [l for l in requested if l in avail] or (["eng"] if "eng" in avail else [])
    lang_arg = "+".join(usable) if usable else "eng"

    result = ExtractionResult(engine=f"tesseract:{lang_arg}", page_count=len(image_paths))
    missing = [l for l in requested if l not in avail]
    if missing and not auto:
        result.warnings.append(
            "Tesseract language pack(s) not installed: " + ", ".join(missing)
            + ". Install tesseract-langpack-hin for Devanagari records."
        )

    # Script is decided once per document, from the first page's RAW image -
    # one record is overwhelmingly one script, so paying the trial cost per
    # page would multiply it for no additional information.
    if auto and image_paths:
        selected, diag = select_languages(image_paths[0], avail, out_dir)
        if selected is None:
            result.warnings.append("No usable Tesseract language pack installed.")
        else:
            lang_arg = selected
            result.engine = f"tesseract:{lang_arg}"
            result.warnings.append(
                f"Script selection: '{lang_arg}' chosen from {diag.get('trials')} "
                f"candidate pack(s) (mean confidence {diag.get('confidence')}, "
                f"{diag.get('margin')} ahead of the next).")

    agg_quality: List[dict] = []
    for idx, img_path in enumerate(image_paths, start=1):
        pre_path, metrics, warns = assess_and_preprocess(img_path, out_dir)
        if metrics:
            agg_quality.append(metrics)
        result.warnings.extend(f"p{idx}: {w}" for w in warns)
        target = pre_path or img_path
        if idx == 1:
            result.render_path = img_path
        gate_mult = metrics.get("confidence_gate_multiplier", 1.0) if metrics else 1.0

        tsv, psm_used = _ocr_page_best_layout(target, lang_arg)
        if tsv is None:
            result.warnings.append(f"p{idx}: OCR failed.")
            continue
        if psm_used == _PSM_AUTO_LAYOUT:
            result.warnings.append(
                f"p{idx}: page read with full layout analysis (psm "
                f"{_PSM_AUTO_LAYOUT}) rather than the single-block default - "
                f"it found substantially more text, which is what a sheet "
                f"carrying several separate regions (form, entries, map) "
                f"looks like.")

        page_lines = _lines_from_tsv(tsv, idx, gate_mult)

        # Second opinion on the numbers only - see _repair_digits.
        if [l for l in lang_arg.split("+") if l != "eng"] and "eng" in avail:
            digit_tsv = _ocr_page(target, "eng", psm=psm_used)
            if digit_tsv is not None:
                repaired = _repair_digits(
                    page_lines, _lines_from_tsv(digit_tsv, idx, gate_mult))
                if repaired:
                    result.warnings.append(
                        f"p{idx}: {repaired} line(s) had their digits re-read "
                        f"with the English pack. An Indic pack reads the prose "
                        f"but misreads Latin numerals, and khasra/khata/area "
                        f"are exactly the fields that cannot absorb that.")

        result.lines.extend(page_lines)
        _assess_handwriting(result, target, idx)

    if agg_quality:
        result.quality = {
            "legibility_score": round(sum(q["legibility_score"] for q in agg_quality) / len(agg_quality), 1),
            "pages": agg_quality,
        }
    return result


def _assess_handwriting(result: "ExtractionResult", page_path: str, page: int) -> None:
    """
    Judge each line of one page as printed or not, in place.

    Runs on the SAME image Tesseract read, because the line boxes are in that
    image's coordinates - cropping the raw scan instead would sample the wrong
    pixels once deskew has moved anything.

    A line that cannot be judged keeps handwriting=None. That is deliberately
    distinct from a verdict of "printed": the caller must not be able to
    mistake "we did not look" for "we looked and it was fine".
    """
    if _handwriting is None or not _handwriting.available():
        return
    if _cv2 is None:
        return
    page_gray = _cv2.imread(page_path, _cv2.IMREAD_GRAYSCALE)
    if page_gray is None:
        return

    flagged = 0
    transcribed = 0
    for line in result.lines:
        if line.page != page or line.handwriting is not None:
            continue
        crop = _handwriting.crop_line(page_gray, line.bbox)
        if crop is None:
            continue
        verdict = _handwriting.inspect_line(crop)
        if verdict is None:
            continue
        line.handwriting = verdict
        if verdict.get("is_handwriting_suspected"):
            flagged += 1
            # Tesseract is trained on print. On a handwritten line it does not
            # fail loudly, it returns plausible text at an ordinary
            # confidence - so the confidence is the thing that has to be
            # corrected, or extraction will trust it downstream.
            line.confidence = round(min(line.confidence, 0.35), 4)

        # Recognition is gated on the SEGMENT verdict, not the line verdict.
        #
        # Gating on the line was the obvious wiring and it was wrong: measured
        # on a mixed form, the line-level detector flagged four lines and all
        # four were printed - the Devanagari title, "Land Classification",
        # "ULPIN" - while missing every one of the five handwritten rows.
        # TrOCR then dutifully re-read the word "ULPIN" and nothing that
        # mattered.
        #
        # The line verdict averages a printed label together with a
        # handwritten value and describes neither, which is the same reason
        # split_segments exists. So every line is offered to the segment
        # detector, and only a run that scores as handwriting ON ITS OWN is
        # sent to the recogniser.
        if _transcribe_handwriting(line, crop):
            transcribed += 1

    if flagged:
        note = (f"p{page}: {flagged} line(s) look handwritten rather than "
                f"printed. Tesseract cannot read handwriting reliably, so "
                f"their confidence was suppressed")
        if transcribed:
            note += (f". The handwriting recogniser (TrOCR) offered an "
                     f"alternative reading for {transcribed} line(s), kept "
                     f"alongside the original rather than replacing it")
        result.warnings.append(note + "; they must be confirmed by hand.")


# A TrOCR reading replaces Tesseract's only when it is not obviously worse.
# An empty or single-character result usually means the crop was wrong rather
# than the writing illegible, and swapping a flawed reading for nothing is a
# loss - the verifier has less to correct from, not more.
_HTR_MIN_CHARS = 2


def _transcribe_handwriting(line: "Line", crop) -> bool:
    """
    Re-read the handwritten part of one line with TrOCR, in place.

    Two things make this work, and both were measured. The recogniser is
    given the handwritten SEGMENT rather than the whole row: on five
    handwritten fields, the full row returned "owner 1887 1888" and
    "share # and #" while tight crops of the same values returned
    "Ramesh Kumar Yadav" and "Shri Mohan Lal Yadav" exactly - 0 of 5 against
    3 of 5, same model, same pixels. And the printed label is put back in
    front of the new reading, because field extraction is label-anchored and
    a value with its label removed can no longer be found.

    Tesseract's reading is never discarded; it is kept on the line as
    `htr.tesseract` so a verifier can see both. Returns True when the line
    text was replaced.
    """
    if _handwriting is None or _trocr is None or not _trocr.available():
        return False
    if not hasattr(_handwriting, "handwritten_segments"):
        return False
    try:
        segments = _handwriting.handwritten_segments(crop)
    except Exception:
        return False
    if not segments:
        return False

    # The widest handwritten run is the value; narrower ones are usually a
    # stray mark or part of the label that scored oddly.
    seg = max(segments, key=lambda s: s.get("width", 0))
    try:
        out = _trocr.transcribe_line(seg["crop"], line.text)
    except Exception:
        return False
    if not out or not out.get("script_supported", True):
        line.htr = {"attempted": True, "applied": False,
                    "reason": out.get("reason") if out else "no result"}
        return False

    text = (out.get("text") or "").strip()
    if hasattr(_trocr, "trim_overrun"):
        text = _trocr.trim_overrun(text, seg["crop"]).strip()
    if len(text) < _HTR_MIN_CHARS:
        line.htr = {"attempted": True, "applied": False,
                    "reason": "reading too short to trust"}
        return False

    # The TrOCR reading is recorded as a SECOND OPINION. line.text is never
    # overwritten.
    #
    # Splicing it in was tried and measured, and it made things worse: the
    # handwritten run has to be located inside the line and the printed label
    # put back in front of it, and getting either slightly wrong corrupts a
    # reading that was nearly right. On the mixed test page it turned
    # "Owner Ramesh Kumar Yaday" into "Ramesn 41017 Yaday" - Tesseract had
    # been one character from correct.
    #
    # The asymmetry decides it. A better reading offered alongside the
    # original costs nothing if ignored; a worse reading substituted for the
    # original destroys information the verifier needed. Every handwritten
    # field is already routed to needs_review, so a human is looking at these
    # lines regardless - what they need is both candidates, not a silent
    # swap. Promoting the TrOCR reading automatically is a decision for
    # whoever can measure it against real handwritten records; the reading
    # itself is available on the line from now on.
    line.htr = {
        "attempted": True, "applied": False, "suggestion": True,
        "tesseract": line.text,
        "trocr": text,
        "confidence": out.get("confidence"),
        "segment": {k: seg[k] for k in ("x_offset", "y_offset", "width", "height")},
    }
    return True


def extract(path: str, work_dir: Optional[str] = None,
            languages: Optional[str] = None) -> ExtractionResult:
    """
    Main entry point. Chooses the best available extraction path for `path`,
    and - unless `languages` is given explicitly - the right script's
    language packs for it (see select_languages).
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
