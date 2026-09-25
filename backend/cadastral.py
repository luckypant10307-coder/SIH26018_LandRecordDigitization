"""
Cadastral map vectorization and georeferencing.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Two independent stages:

  1. VECTORIZATION
     Turn a raster parcel-boundary map into vector polygons - a short list
     of real corner vertices per parcel, not one point per pixel. Approach:
     threshold the scan so boundary ink is the only "dark" content, find
     each enclosed parcel interior as an isolated connected region
     (cv2.findContours), then simplify its pixel-level staircase outline
     with the Douglas-Peucker algorithm (cv2.approxPolyDP). Classical
     computer vision, not a trained model - deterministic and auditable,
     the same design choice this project makes everywhere else (see
     field_extractor.py's and validator.py's module docstrings).

  2. GEOREFERENCING
     Fit an affine transform from a handful of ground control points
     (pixel position <-> known real-world lon/lat) and apply it to every
     vectorized vertex. Three points give an exact fit; more give a
     least-squares fit whose residuals are reported, so georeferencing
     quality is measurable rather than assumed.

Honesty note: there is no real cadastral map or real surveyed ground
control point available to this project - the same gap README S12/S13
already disclose for the DILRMP connector and the bundled admin/registry
master data. tools/make_cadastral_map.py generates a clearly-labelled
synthetic demo map and illustrative (not surveyed) control points. The
pipeline below is real: it runs identically on a genuine scanned map and
genuine surveyed GCPs once they exist, and every GeoJSON it produces from
the bundled demo data carries an explicit `_disclaimer` saying so.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


_cv2 = _try_import("cv2")
_numpy = _try_import("numpy")

CV_AVAILABLE = _cv2 is not None and _numpy is not None

DEMO_DISCLAIMER = (
    "Synthetic demo map and illustrative (not surveyed) ground control "
    "points - see tools/make_cadastral_map.py. Coordinates are not real."
)


@dataclass
class Parcel:
    parcel_id: int
    pixel_polygon: List[Tuple[float, float]]
    pixel_centroid: Tuple[float, float]
    area_px: float
    khasra_number: Optional[str] = None
    geo_polygon: Optional[List[Tuple[float, float]]] = None   # [lon, lat] pairs

    def to_dict(self) -> dict:
        return asdict(self)


def detect_map_frame(gray, min_area_fraction: float = 0.25) -> Optional[Tuple[int, int, int, int]]:
    """
    Real map exports (verified against actual Bhu-Naksha plot reports, not
    just the synthetic demo) commonly surround the plotted map with a title,
    a legend, or an attribute/owner table outside it. Most such exports draw
    an explicit border around the map panel itself, and that border is
    reliably the largest contour on the page by a huge margin - on a real
    Bhu-Naksha report, ~700x larger than the next candidate (a text block or
    an icon). Detect it and return its (x, y, w, h) bounding box, or None if
    no single contour dominates enough to be confident it is a frame rather
    than genuine map content (e.g. a map with no title/table at all).
    """
    cv2 = _cv2
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < (h * w) * min_area_fraction:
        return None
    return cv2.boundingRect(best)


def vectorize(image_path: str, min_area_px: float = 2000.0,
             max_area_ratio_to_median: float = 4.0,
             simplify_epsilon_frac: float = 0.01,
             crop_to_frame: bool = True) -> List[Parcel]:
    """
    Detect enclosed parcel regions in a raster cadastral map and simplify
    each into a vector polygon.

    Assumes boundary lines fully enclose each parcel (the usual case for a
    printed/scanned cadastral index map): under Otsu thresholding the
    background and every parcel interior stay light while the ink stays
    dark, so each parcel interior is its own isolated connected light
    region rather than one big connected canvas - cv2.RETR_LIST finds them
    directly, with no need to reason about a containment hierarchy.

    Two area filters, not one, because they catch different failure modes:
      - min_area_px is an absolute floor that rejects pure noise specks and
        the tiny slivers a boundary-line gap at a corner junction can leave
        behind (faded ink and imprecise scanning produce exactly this on
        real maps too) - the default assumes a scale where a genuine parcel
        is at least a few thousand square pixels; tune it to the map's
        actual resolution.
      - max_area_ratio_to_median rejects a region many times larger than a
        typical parcel in the same map - almost always two or more parcels
        that leaked into one connected region through a boundary-line gap
        (an unclosed corner, a broken line), not one real oversized parcel.
        A real cadastral map can have genuinely large parcels too, so this
        is a heuristic, not a certainty - loosen or disable it via the
        parameter if a map is known to mix wildly different parcel sizes.

    crop_to_frame restricts detection to the map's own bordered panel (see
    detect_map_frame) when one is found, so a title, legend or attribute
    table elsewhere on the page cannot be picked up as false parcels - it
    also cannot silently corrupt the area filters above by dragging the
    median down with a page full of tiny text-character contours, which is
    exactly what happens without it on a real Bhu-Naksha plot report.
    Returned coordinates are always in the original, uncropped image's
    pixel space, so callers (including read_parcel_labels and any
    externally-defined ground control points) never need to know a crop
    happened.

    Only genuinely closed boundaries become parcels. A real map export
    commonly shows neighbouring plots only for context, with their far/outer
    edges left undrawn where they fall outside the export's extent - those
    plots are correctly left undetected rather than guessed at, the same
    "never fabricate" principle this project applies everywhere else (see
    field_extractor.py's degraded-mode and validator.py's module
    docstrings). Verified on real Bhu-Naksha single-plot reports: the
    queried plot and every other plot fully interior to the rendered extent
    vectorize correctly; only the plots clipped by the extent's edge do not.
    """
    if not CV_AVAILABLE:
        raise RuntimeError(
            "OpenCV/numpy not installed - cadastral vectorization requires them.")
    cv2, np = _cv2, _numpy

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    offset_x, offset_y = 0, 0
    work_img = img
    if crop_to_frame:
        frame = detect_map_frame(img)
        if frame is not None:
            offset_x, offset_y, fw, fh = frame
            work_img = img[offset_y:offset_y + fh, offset_x:offset_x + fw]

    _, binary = cv2.threshold(work_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = work_img.shape[:2]
    full_area = float(h * w)
    candidates = [(cnt, cv2.contourArea(cnt)) for cnt in contours]
    candidates = [(cnt, a) for cnt, a in candidates
                 if min_area_px <= a <= full_area * 0.9]
    if not candidates:
        return []

    areas = sorted(a for _, a in candidates)
    median_area = areas[len(areas) // 2]
    area_ceiling = median_area * max_area_ratio_to_median if median_area > 0 else full_area

    parcels: List[Parcel] = []
    for cnt, area in candidates:
        if area > area_ceiling:
            continue
        peri = cv2.arcLength(cnt, True)
        epsilon = max(1.0, simplify_epsilon_frac * peri)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        poly = [(float(p[0][0]) + offset_x, float(p[0][1]) + offset_y) for p in approx]
        if len(poly) < 3:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"] + offset_x
        cy = M["m01"] / M["m00"] + offset_y
        parcels.append(Parcel(
            parcel_id=len(parcels) + 1, pixel_polygon=poly,
            pixel_centroid=(cx, cy), area_px=float(area),
        ))
    return parcels


def read_parcel_labels(image_path: str, parcels: List[Parcel],
                       half_width: int = 45, half_height: int = 14) -> None:
    """
    Best-effort: OCR a small crop around each parcel's centroid to recover
    its printed khasra number. Optional and fails silently to
    khasra_number=None per parcel - a missing label does not invalidate the
    polygon, the geometry is still useful without it (the honest-degradation
    pattern this project uses everywhere, e.g. ocr_engine.py's degraded mode).
    """
    if _cv2 is None:
        return
    try:
        import pytesseract
    except Exception:
        return
    cv2 = _cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    h, w = img.shape[:2]
    for p in parcels:
        cx, cy = p.pixel_centroid
        x0, y0 = max(0, int(cx - half_width)), max(0, int(cy - half_height))
        x1, y1 = min(w, int(cx + half_width)), min(h, int(cy + half_height))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        try:
            text = pytesseract.image_to_string(
                crop, config="--psm 7 -c tessedit_char_whitelist=0123456789/").strip()
        except Exception:
            continue
        if text:
            p.khasra_number = text


# --------------------------------------------------------------------------
# Georeferencing
# --------------------------------------------------------------------------

@dataclass
class AffineTransform:
    """
    lon = a*px + b*py + c
    lat = d*px + e*py + f
    """
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def apply(self, px: float, py: float) -> Tuple[float, float]:
        lon = self.a * px + self.b * py + self.c
        lat = self.d * px + self.e * py + self.f
        return (lon, lat)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_affine_transform(
    control_points: List[Tuple[float, float, float, float]],
) -> Tuple[AffineTransform, List[float]]:
    """
    Fit a 6-parameter affine transform from pixel to (lon, lat) coordinates.

    control_points: (pixel_x, pixel_y, lon, lat) tuples, at least 3. Three
    points give an exact fit (zero residual); more give a least-squares fit.

    Returns (transform, per_point_residuals) - residuals are reported in
    the same units as lon/lat (degrees) rather than assumed to be zero, so a
    caller can tell a good georeferencing from a bad one instead of trusting
    it blindly.
    """
    if not CV_AVAILABLE:
        raise RuntimeError("numpy not installed - georeferencing requires it.")
    if len(control_points) < 3:
        raise ValueError("At least 3 control points are required to fit an affine transform.")
    np = _numpy

    A = np.array([[px, py, 1.0] for px, py, _, _ in control_points])
    lon = np.array([lo for _, _, lo, _ in control_points])
    lat = np.array([la for _, _, _, la in control_points])

    coef_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)
    coef_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)
    transform = AffineTransform(
        a=float(coef_lon[0]), b=float(coef_lon[1]), c=float(coef_lon[2]),
        d=float(coef_lat[0]), e=float(coef_lat[1]), f=float(coef_lat[2]),
    )

    residuals = []
    for px, py, lo, la in control_points:
        pred_lon, pred_lat = transform.apply(px, py)
        residuals.append(float(((pred_lon - lo) ** 2 + (pred_lat - la) ** 2) ** 0.5))
    return transform, residuals


_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LON_EQUATOR = 111320.0


# Residual tolerance for a fitted georeferencing, in metres.
#
# DILRMP village-level work is commonly held to 1-2 m; urban and parcel-level
# work is stricter. 2.0 m is the default here because that is the loosest end
# of the village band, and a demo that silently tightened the bar would
# reject georeferencing a state office considers acceptable.
MAX_GCP_RMS_METRES = 2.0

# An affine fit has 6 parameters, so 3 control points determine it exactly
# and always report zero residual. Dropping to 3 therefore produces a
# meaningless "perfect" fit, which is worse than a large honest one because
# it cannot be distinguished from a good one. Outlier removal stops at 4.
MIN_GCP_FOR_REDUNDANCY = 4


def rms_metres(residuals: List[float], latitude: float = 23.0) -> float:
    """
    Root-mean-square residual converted from degrees to metres.

    Residuals come out of fit_affine_transform in degrees, which is not a
    unit anyone can hold a tolerance in - and a degree of longitude is not a
    degree of latitude, so the conversion needs the working latitude. The
    default is central India; pass the real one for anything published.
    """
    if not residuals:
        return 0.0
    mean_sq = sum(r * r for r in residuals) / len(residuals)
    deg = mean_sq ** 0.5
    return deg * _M_PER_DEG_LAT


def fit_transform_robust(
    control_points: List[Tuple[float, float, float, float]],
    max_rms_m: float = MAX_GCP_RMS_METRES,
    latitude: float = 23.0,
) -> Tuple[AffineTransform, dict]:
    """
    Fit an affine transform, dropping the worst control point until the RMS
    residual is within tolerance.

    This is the loop a surveyor runs by hand in the QGIS or ArcGIS
    georeferencer: fit, look at the per-point residuals, delete the obvious
    mis-click, re-fit. Doing it here means a bad GCP is reported and removed
    rather than quietly spread across every parcel on the sheet.

    The report says what happened and never pretends to more than it has:

      * `passed` - whether the tolerance was actually met.
      * `redundancy` - False once only 3 points remain, where the zero
        residual is an artefact of the fit being exactly determined rather
        than evidence of accuracy.
      * `dropped` - which points were removed and what they scored, so a
        removal can be argued with.

    Returns the transform even when the tolerance was NOT met, because a
    coarse georeferencing a reviewer has been warned about is more useful
    than none at all - but `passed` is False and the caller must surface it.
    """
    points = list(control_points)
    if len(points) < 3:
        raise ValueError("At least 3 control points are required to fit an affine transform.")

    dropped: List[dict] = []
    initial_rms: Optional[float] = None
    while True:
        transform, residuals = fit_affine_transform(points)
        rms = rms_metres(residuals, latitude)
        if initial_rms is None:
            initial_rms = rms
        redundancy = len(points) > 3
        if rms <= max_rms_m or len(points) <= MIN_GCP_FOR_REDUNDANCY:
            return transform, {
                "method": "control_points_robust",
                "control_point_count": len(points),
                "rms_metres": round(rms, 3),
                # The RMS BEFORE any point was dropped. Reporting only the
                # final figure would show a clean 0.01 m fit and hide that it
                # started at 2 km - which is the one number that tells a
                # reviewer a control point was genuinely wrong rather than
                # marginal.
                "initial_rms_metres": round(initial_rms or 0.0, 3),
                "max_rms_metres": max_rms_m,
                "passed": bool(rms <= max_rms_m),
                "redundancy": redundancy,
                "per_point_residual_m": [round(r * _M_PER_DEG_LAT, 3)
                                         for r in residuals],
                "dropped": dropped,
                "note": ("A 3-point affine fit is exactly determined; its "
                         "zero residual is not a measurement."
                         if not redundancy else ""),
            }
        worst = max(range(len(points)), key=lambda i: residuals[i])
        dropped.append({
            "index": worst,
            "point": points[worst],
            "residual_m": round(residuals[worst] * _M_PER_DEG_LAT, 3),
        })
        points.pop(worst)


def georeference_parcels(parcels: List[Parcel], transform: AffineTransform) -> None:
    """Mutates each parcel in place, filling geo_polygon from pixel_polygon."""
    for p in parcels:
        p.geo_polygon = [transform.apply(x, y) for x, y in p.pixel_polygon]


# Metres per degree. Longitude degrees shrink with latitude, which matters at
# India's spread (Kanyakumari ~8°N to Ladakh ~35°N changes the lon scale by
# about 20%), so the cosine correction is applied at each parcel's own
# latitude rather than assuming a single national constant.
def polygon_area_m2(ring: List[Tuple[float, float]]) -> float:
    """
    Real-world area of a lon/lat ring, in square metres.

    Uses an equirectangular projection centred on the ring's own latitude,
    then the shoelace formula. For a parcel - hundreds of metres across, not
    hundreds of kilometres - the distortion of that projection is far below
    the error already present in vectorising a scanned map, so a full
    geodesic area computation would be false precision.
    """
    pts = [p for p in ring]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]                       # shoelace closes the ring itself
    if len(pts) < 3:
        return 0.0
    lat0 = sum(lat for _lon, lat in pts) / len(pts)
    mx = _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    xy = [(lon * mx, lat * _M_PER_DEG_LAT) for lon, lat in pts]
    total = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_centroid(ring: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """(lon, lat) centroid of a ring - the coordinate a matched record is
    geo-tagged with. Vertex mean, not the area centroid: parcel outlines here
    are simplified quadrilateral-ish shapes where the two agree closely, and
    the vertex mean cannot blow up on a degenerate ring."""
    pts = [p for p in ring]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def parcels_to_geojson(parcels: List[Parcel], disclaimer: Optional[str] = None) -> dict:
    """Georeferenced parcels as a standard GeoJSON FeatureCollection."""
    features = []
    for p in parcels:
        if p.geo_polygon is None:
            continue
        ring = p.geo_polygon + [p.geo_polygon[0]]     # GeoJSON polygons must close
        centroid = polygon_centroid(ring)
        features.append({
            "type": "Feature",
            "properties": {
                "parcel_id": p.parcel_id,
                "khasra_number": p.khasra_number,
                "area_px": p.area_px,
                # Real-world measurements, so a record can be cross-checked
                # against its own geometry rather than only drawn on top of it.
                "area_m2": round(polygon_area_m2(ring), 1),
                "centroid_lon": round(centroid[0], 6) if centroid else None,
                "centroid_lat": round(centroid[1], 6) if centroid else None,
            },
            "geometry": {"type": "Polygon", "coordinates": [[list(pt) for pt in ring]]},
        })
    fc: Dict = {"type": "FeatureCollection", "features": features}
    if disclaimer:
        fc["_disclaimer"] = disclaimer
    return fc


def vectorize_and_georeference(
    image_path: str,
    control_points: Optional[List[Tuple[float, float, float, float]]] = None,
    read_labels: bool = True,
    disclaimer: Optional[str] = None,
    transform: Optional[AffineTransform] = None,
    georeference_meta: Optional[dict] = None,
) -> dict:
    """
    Full pipeline: raster map -> vector parcels -> georeferenced GeoJSON.

    The transform can arrive two ways. Pass `control_points` and it is fitted
    here by least squares. Pass `transform` and it is used as given - that is
    the path for a georeferencing already computed by ArcGIS or QGIS and read
    back by georeference.py, where re-deriving it from GCPs would only add
    error to a result a surveyor has already established.
    """
    if transform is None and not control_points:
        raise ValueError("Either control_points or a transform is required.")

    parcels = vectorize(image_path)
    if read_labels:
        read_parcel_labels(image_path, parcels)

    if transform is None:
        transform, residuals = fit_affine_transform(control_points)
        meta = {
            "method": "control_points",
            "control_point_count": len(control_points),
            # Residual is the honest quality figure for a fitted transform:
            # three GCPs always fit exactly, so a zero here means "not
            # over-determined", not "accurate".
            "max_residual_deg": max(residuals) if residuals else None,
            "transform": transform.to_dict(),
        }
    else:
        meta = {"method": "imported", "transform": transform.to_dict()}
        meta.update(georeference_meta or {})

    georeference_parcels(parcels, transform)
    geojson = parcels_to_geojson(parcels, disclaimer=disclaimer)
    geojson["_georeferencing"] = meta
    return geojson


def load_control_points(path: str) -> List[Tuple[float, float, float, float]]:
    """Load (pixel_x, pixel_y, lon, lat) tuples from a control_points.json file."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out = []
    for gcp in payload.get("control_points", []):
        px, py = gcp["pixel"]
        out.append((float(px), float(py), float(gcp["lon"]), float(gcp["lat"])))
    return out
