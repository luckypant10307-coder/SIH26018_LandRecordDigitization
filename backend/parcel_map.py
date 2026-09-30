"""
Reading the parcel map that a Bhu-Naksha plot report carries inside itself.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHY THIS IS NOT cadastral.py

cadastral.py vectorises SCANNED cadastral sheets - dark ink on paper - and
does it well: 6 of 6 parcels in 0.09s on the sheets it was built for. Handed
the map embedded in a plot report it returns ZERO, and the reason is not a
tuning problem:

    Otsu chose a threshold of 29 on a 600x600 portal map. The near-black
    text labels sat below it and the light blue boundary lines sat above,
    so the lines were classified as BACKGROUND and the entire map came back
    as one connected region of 356,762 pixels.

A portal map is a vector render - thin coloured lines on white - not ink on
paper. Luminance is the wrong axis: the lines are LIGHTER than the text they
sit beside, and darker than nothing. Colour is the right axis, because the
lines are saturated and the paper is not. Detecting ink as "coloured OR
dark" took ink detection from 0.15% of pixels to 6.9%.

WHAT THIS DELIBERATELY DOES NOT PROMISE

Even with the lines seen, the parcels do not separate into closed regions:
the rendered boundaries have gaps, so the interiors leak into one another
and into the page. Closing those reliably is a harder problem and this
module does not pretend to have solved it - find_parcels reports what it
finds and is honest when that is nothing.

What it DOES deliver is the part that needs no closed boundary at all. The
subject parcel of a plot report is drawn FILLED, so it is a solid connected
region and can be lifted out exactly. Measured on a real report: 21,692 px,
eight vertices after simplification - the same order of complexity
cadastral.py produces on maps it handles well.

WHAT THE MAP IS WORTH, WITHOUT GEOREFERENCING

These maps carry no world file and no control points, so nothing here yields
latitude and longitude. What it yields is a SHAPE and a set of NEIGHBOURS,
and the neighbours are the valuable part: a khasra's surroundings are hard
to fake and cheap to compare, and the plot report prints them right there on
the map. That is an independent check on the one field everything else keys
off - and the text pipeline and the map pipeline read it from different
places, so agreement between them means something.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple


def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


_cv2 = _try_import("cv2")
_numpy = _try_import("numpy")

CV_AVAILABLE = _cv2 is not None and _numpy is not None

# A boundary line is COLOURED; paper is not. 40 of 255 is comfortably above
# the JPEG/PNG colour noise on a white background and far below the
# saturation of any line a portal actually draws.
INK_MIN_SATURATION = 40
# Text labels are near-black and barely saturated, so they need the second
# test. 160 keeps anti-aliased grey glyph edges without swallowing paper.
INK_MAX_VALUE = 160

# A highlight is a large, strongly coloured FILL. The floor is in pixels
# rather than a fraction so a bigger export does not change what counts.
HIGHLIGHT_MIN_AREA_PX = 2000
HIGHLIGHT_MIN_SATURATION = 80
HIGHLIGHT_MIN_VALUE = 150

# Simplification, as a fraction of the contour's own perimeter. 0.01 is what
# cadastral.py uses and it produced 8 vertices here - a parcel outline, not a
# traced-pixel staircase.
SIMPLIFY_EPSILON_FRAC = 0.01

# A khasra label on a map: digits, optionally a sub-division after a slash.
_LABEL_RE = re.compile(r"^\d{1,5}(?:\s*/\s*\d{1,3})?$")


@dataclass
class MapReading:
    """What could be read off one embedded parcel map."""
    subject_polygon: List[Tuple[int, int]] = dc_field(default_factory=list)
    subject_area_px: float = 0.0
    subject_centroid: Optional[Tuple[float, float]] = None
    labels: List[str] = dc_field(default_factory=list)      # every khasra on the map
    subject_label: Optional[str] = None                      # the one inside the fill
    neighbour_labels: List[str] = dc_field(default_factory=list)
    notes: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subject_polygon": [list(p) for p in self.subject_polygon],
            "subject_area_px": round(self.subject_area_px, 1),
            "subject_centroid": (list(self.subject_centroid)
                                 if self.subject_centroid else None),
            "subject_label": self.subject_label,
            "neighbour_labels": self.neighbour_labels,
            "labels": self.labels,
            "notes": self.notes,
        }


def ink_mask(bgr):
    """
    Where the drawing is: anything COLOURED or DARK.

    Two tests rather than one because a portal map has two kinds of mark and
    a single threshold cannot hold both - the boundary lines are light but
    saturated, the labels are dark but grey. Thresholding on luminance alone
    is what made Otsu discard every line on a real map.
    """
    cv2, np = _cv2, _numpy
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    coloured = hsv[..., 1] > INK_MIN_SATURATION
    dark = gray < INK_MAX_VALUE
    return (coloured | dark).astype(np.uint8)


def find_highlight(bgr):
    """
    The filled parcel, as (contour, area, centroid), or None.

    Hue is NOT hardcoded. These reports fill in yellow, but the fill colour
    is a portal's styling choice and a green or pink one would defeat a hue
    test for no good reason. What identifies a highlight is that it is a
    LARGE, STRONGLY COLOURED, SOLID region - boundary lines are saturated too
    but are thin, and paper is large but unsaturated.
    """
    cv2, np = _cv2, _numpy
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    filled = ((hsv[..., 1] > HIGHLIGHT_MIN_SATURATION)
              & (hsv[..., 2] > HIGHLIGHT_MIN_VALUE)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, 8)
    best = None
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < HIGHLIGHT_MIN_AREA_PX:
            continue
        if best is None or area > best[1]:
            best = (i, area)
    if best is None:
        return None
    index, area = best
    mask = (labels == index).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    cx, cy = centroids[index]
    return contour, float(area), (float(cx), float(cy))


TEXT_MAX_VALUE = 150          # a label is near-black
TEXT_MAX_SATURATION = 60      # ...and, unlike every line on the map, grey


def isolate_labels(bgr):
    """
    Black text on white, with the map itself removed.

    Handed the map as-is, Tesseract read seven plot numbers as `['a,', 'Sr)']`
    - the coloured boundaries and the highlight fill dominate its binarisation
    and the digits are lost inside it. The labels are separable on colour
    though: they are the only marks that are DARK AND UNSATURATED, where
    every line and the fill are saturated. Isolating them first took the same
    image from two junk tokens to six of seven plot numbers read correctly.

    Returned at NATIVE resolution on purpose. Upscaling before OCR is the
    usual advice and it is wrong here: these glyphs are single-pixel strokes,
    and interpolating them 3x blurred the result back down to two tokens.
    """
    cv2, np = _cv2, _numpy
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    is_text = (gray < TEXT_MAX_VALUE) & (hsv[..., 1] < TEXT_MAX_SATURATION)
    clean = np.full(gray.shape, 255, np.uint8)
    clean[is_text] = 0
    return clean


def _tesseract_labels(image_path: str) -> List[Tuple[str, int, int]]:
    """
    Khasra numbers printed on the map, as (text, x, y) in pixels.

    Sparse-text mode with a digit whitelist: the map has no sentences, only
    scattered plot numbers, and letting the engine expect prose turns "89"
    into "B9" often enough to matter on a field that identifies the land.
    """
    if not shutil.which("tesseract") or not CV_AVAILABLE:
        return []
    cv2 = _cv2
    out_dir = tempfile.mkdtemp(prefix="parcelmap_")
    stem = os.path.join(out_dir, "labels")
    try:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return []
        prepared = os.path.join(out_dir, "text.png")
        cv2.imwrite(prepared, isolate_labels(bgr))
        subprocess.run(
            ["tesseract", prepared, stem, "--psm", "11",
             "-c", "tessedit_char_whitelist=0123456789/", "tsv"],
            capture_output=True, timeout=60, check=False)
        tsv = stem + ".tsv"
        if not os.path.exists(tsv):
            return []
        found = []
        with open(tsv, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            try:
                ix, iy = header.index("left"), header.index("top")
                iw, ih = header.index("width"), header.index("height")
                it = header.index("text")
            except ValueError:
                return []
            for row in fh:
                cells = row.rstrip("\n").split("\t")
                if len(cells) <= it:
                    continue
                text = cells[it].strip()
                if not text or not _LABEL_RE.match(text):
                    continue
                try:
                    x = int(cells[ix]) + int(cells[iw]) // 2
                    y = int(cells[iy]) + int(cells[ih]) // 2
                except ValueError:
                    continue
                found.append((text.replace(" ", ""), x, y))
        return found
    except Exception:
        return []
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def read_map(image_path: str) -> MapReading:
    """
    Read one embedded parcel map. Never raises; reports what it could not do.
    """
    reading = MapReading()
    if not CV_AVAILABLE:
        reading.notes.append("OpenCV/numpy not installed - the map was not read.")
        return reading
    cv2 = _cv2
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        reading.notes.append("The map image could not be opened.")
        return reading

    highlight = find_highlight(bgr)
    if highlight is None:
        reading.notes.append(
            "No highlighted parcel on this map - the report does not mark its "
            "subject plot, so the shape could not be attributed to a khasra.")
    else:
        contour, area, centroid = highlight
        perimeter = cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, SIMPLIFY_EPSILON_FRAC * perimeter, True)
        reading.subject_polygon = [(int(p[0][0]), int(p[0][1])) for p in poly]
        reading.subject_area_px = area
        reading.subject_centroid = centroid

    labels = _tesseract_labels(image_path)
    if not labels:
        reading.notes.append("No khasra labels could be read from the map.")
        return reading

    reading.labels = sorted({t for t, _, _ in labels}, key=_label_sort)

    # Which label sits INSIDE the highlighted parcel: that is the map's own
    # statement of which plot this report is about, arrived at independently
    # of the text layer.
    if highlight is not None:
        contour, _, centre = highlight
        inside = [(t, x, y) for t, x, y in labels
                  if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0]
        if inside:
            # CLOSEST TO THE CENTROID, not the first one found.
            #
            # More than one label can fall inside the filled region: a
            # neighbour's number printed near the shared edge spills across
            # it. Measured on a real report, taking the first match returned
            # 90 - the thin plot to the left - for a parcel the text and the
            # map both call 89, and reported a MISMATCH on a document where
            # the two agreed. A false mismatch is worse than no check,
            # because it teaches a reviewer to distrust the check.
            #
            # A parcel's own label is drawn near its middle; a neighbour's
            # that overlaps the edge is not. So distance to the centroid is
            # what separates them.
            cx, cy = centre
            inside.sort(key=lambda t: (t[1] - cx) ** 2 + (t[2] - cy) ** 2)
            reading.subject_label = inside[0][0]
        reading.neighbour_labels = sorted(
            {t for t in reading.labels if t != reading.subject_label},
            key=_label_sort)
    return reading


def _label_sort(value: str):
    head = value.split("/")[0]
    return (int(head) if head.isdigit() else 0, value)


# Below this many labels read, the map was not read well enough to contradict
# anything. Measured: the two documents where the cross-check accused a
# correct record had exactly ONE label recognised between them.
MIN_LABELS_TO_CONTRADICT = 3


def cross_check(reading: MapReading, khasra_from_text: Optional[str]) -> Optional[dict]:
    """
    Compare what the MAP says the plot is against what the TEXT said.

    This is the point of reading the map at all: the two pipelines take the
    number from different places - one from a printed field, one from a label
    drawn on a polygon - so they fail independently, and agreement between
    them is evidence neither could give alone.

    THREE ANSWERS, NOT TWO, because the first version had only two and was
    wrong on half its warnings. Across 20 real documents it reported six
    mismatches, and in three of them the text's khasra was printed ON THE MAP
    - 60 among ['47','59','60','61','62','70'], 183 among the labels of its
    own report - and the centroid heuristic had simply picked a different
    one. Identifying WHICH parcel is highlighted is a much stronger claim
    than noticing whether a number appears at all, and it fails much more
    often, so the two claims are now separated:

      confirmed    - the label at the highlight's centre matches the text
      corroborated - the text's khasra is drawn somewhere on this map
      mismatch     - it is drawn NOWHERE on the parcel's own map

    Only the third is a warning, and only when enough labels were read for
    absence to mean something. A check that cries wolf on a correct record
    teaches a reviewer to ignore it, which is worse than not checking.
    """
    if not khasra_from_text:
        return None
    in_text = str(khasra_from_text).strip().replace(" ", "")
    if not in_text:
        return None
    on_map = (reading.subject_label or "").replace(" ", "")
    all_labels = {t.replace(" ", "") for t in reading.labels}

    if on_map and on_map == in_text:
        return {
            "rule": "MAP_CONFIRMS_KHASRA", "severity": "info", "field": "khasra_number",
            "message": (f"The parcel highlighted on the document's own map is "
                        f"labelled {on_map}, matching the khasra number read "
                        f"from the text."),
            "suggestion": None,
        }

    if in_text in all_labels:
        return {
            "rule": "MAP_SHOWS_KHASRA", "severity": "info", "field": "khasra_number",
            "message": (f"Khasra {in_text} is drawn on the document's own map"
                        + (f", though the highlighted parcel was read as "
                           f"{on_map}" if on_map else "") + "."),
            "suggestion": None,
        }

    if len(all_labels) < MIN_LABELS_TO_CONTRADICT:
        return None                      # too little of the map was read

    return {
        "rule": "MAP_KHASRA_MISMATCH", "severity": "warning", "field": "khasra_number",
        "message": (f"Khasra {in_text} does not appear anywhere on the "
                    f"document's own map, which is labelled "
                    f"{', '.join(sorted(all_labels)[:8])}."),
        "suggestion": ("Check which plot this report is about - the text and "
                       "the map it carries do not name the same parcel."),
    }
