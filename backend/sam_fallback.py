"""
Segment Anything as a LAST-RESORT parcel finder.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHERE THIS SITS, AND WHY IT IS NOT THE PRIMARY PATH

cadastral.vectorize() follows the drawn boundary. On a cadastral sheet that is
exact by construction: the parcel edge IS the surveyed line, so tracing it
recovers a 4-to-11 vertex polygon with no inference at all. Measured over two
real Bhu-Naksha reports it found every plot in 0.09 s.

SAM 2 was measured against it on the same sheets, given every advantage - the
official package, a 1024-point grid, the model the user had already installed:

    metric                     vectorize()        SAM 2 (hiera-small)
    plots found                6 of 6             4 of 6, two merged
    boundary vertices          4 - 11             304 - 861
    time per map               0.09 s             64.5 s
    IoU vs traced polygons     -                  0.09 - 0.98

The vertex column is the disqualifying one. A cadastral parcel is a polygon of
straight survey lines; SAM returns the same shape as a jagged raster blob.
Areas computed from a wobbling outline are wrong, adjacent parcels stop
sharing exact edges so topology.py's gap and overlap checks break, and in one
pass it merged plots 184, 185 and 258 into a single region. Merging two
neighbours is not a cosmetic error in a land record - it is the shape of a
property dispute.

So SAM is NOT an upgrade to the vectoriser and must never run in front of it.

WHAT IT IS ACTUALLY FOR

The vectoriser needs a closed, drawn boundary. On a mudded, torn or heavily
faded cloth revenue sheet the lines are broken and contour following returns
NOTHING - not a worse answer, no answer. That is the only situation where an
appearance-based segmenter helps: approximate blobs beat zero parcels, because
a reviewer can correct a rough boundary and cannot correct an empty map.

Hence the contract enforced here:

  * fires ONLY when the primary vectoriser returned zero parcels,
  * every parcel it produces is marked approximate=True and carries the
    method that made it, so nothing downstream can mistake one for a traced
    boundary, and
  * it never runs unless explicitly enabled, because it needs torch (534 MB)
    plus weights, which the deployment image deliberately excludes.

Enable with SAM_FALLBACK=1. Off by default, and run.py --check says so.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import cadastral

SAM_ENABLED = os.environ.get("SAM_FALLBACK") == "1"
SAM_MODEL = os.environ.get("SAM_MODEL", "facebook/sam2.1-hiera-small")
# A mask covering most of the sheet is the page, not a parcel. Measured: the
# background blob SAM returns on a real report covered 78% of the frame.
MAX_FRAME_FRACTION = float(os.environ.get("SAM_MAX_FRAME_FRACTION", "0.55"))
# Two masks overlapping this much are the same region proposed twice; SAM's
# grid returns many nested variants of one parcel.
NESTED_OVERLAP = 0.80
# Below this a "parcel" is a label smudge or a speck of grid.
MIN_AREA_PX = float(os.environ.get("SAM_MIN_AREA_PX", "2000"))
POINTS_PER_SIDE = int(os.environ.get("SAM_POINTS_PER_SIDE", "32"))

_generator = None
_load_attempted = False
_load_error: Optional[str] = None


def _load():
    """
    Import torch and build the mask generator on demand.

    Deferred hard. Importing torch costs hundreds of MB resident, and an
    installation that never meets a ruined map must not pay for it - nor may
    it happen at module import, which would make `import server` pay on every
    start.
    """
    global _generator, _load_attempted, _load_error
    if _load_attempted:
        return _generator
    _load_attempted = True
    if not SAM_ENABLED:
        _load_error = "SAM_FALLBACK is not 1"
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            # device is pinned to CPU: from_pretrained defaults to CUDA and
            # raises "Torch not compiled with CUDA enabled" on a CPU build,
            # which is what a demo laptop has.
            _generator = SAM2AutomaticMaskGenerator.from_pretrained(
                SAM_MODEL,
                device="cpu",
                points_per_side=POINTS_PER_SIDE,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                min_mask_region_area=int(MIN_AREA_PX),
            )
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        _generator = None
    return _generator


def available() -> bool:
    return _load() is not None


def status() -> dict:
    """Honest status for --check, without forcing a multi-hundred-MB load."""
    return {
        "enabled": SAM_ENABLED,
        "model": SAM_MODEL,
        "loaded": _generator is not None,
        "attempted": _load_attempted,
        "error": _load_error,
    }


def _dedupe(masks, frame_area: float):
    """
    Reduce SAM's overlapping proposals to one per region.

    Two filters, both derived from what the model actually returned on a real
    sheet rather than from taste: a mask covering most of the frame is the page
    itself, and a mask almost entirely inside a larger kept mask is that same
    parcel proposed again at a different grid point.
    """
    import numpy as np

    kept = []
    candidates = [m for m in masks
                  if MIN_AREA_PX <= int(m["segmentation"].sum()) <= frame_area * MAX_FRAME_FRACTION]
    for mask in sorted(candidates, key=lambda m: -int(m["segmentation"].sum())):
        seg = mask["segmentation"]
        area = int(seg.sum())
        nested = False
        for other in kept:
            overlap = int(np.logical_and(seg, other["segmentation"]).sum())
            if overlap / max(area, 1) > NESTED_OVERLAP:
                nested = True
                break
        if not nested:
            kept.append(mask)
    return kept


def segment(image_path: str, simplify_epsilon_frac: float = 0.012) -> List["cadastral.Parcel"]:
    """
    Approximate parcels for a map the vectoriser could not read.

    Returns cadastral.Parcel objects so the rest of the pipeline - the affine
    transform, area computation, the khasra linker - needs no special case.
    Each carries approximate=True.

    The mask outlines are simplified with the same Douglas-Peucker step the
    vectoriser uses, at a slightly looser tolerance. That is damage control,
    not a fix: it turns an 800-vertex blob into something a reviewer can drag,
    while leaving the boundary where SAM put it rather than where a surveyor
    did.

    Never raises. A failure here must leave the caller with the empty result
    it already had.
    """
    generator = _load()
    if generator is None:
        return []
    try:
        import cv2
        import numpy as np
    except Exception:
        return []

    try:
        image = cv2.imread(image_path)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        offset_x = offset_y = 0
        frame = cadastral.detect_map_frame(gray)
        if frame:
            x, y, w, h = frame
            image = image[y:y + h, x:x + w]
            offset_x, offset_y = x, y

        masks = generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frame_area = float(image.shape[0] * image.shape[1])
        parcels: List[cadastral.Parcel] = []

        for index, mask in enumerate(_dedupe(masks, frame_area), start=1):
            binary = mask["segmentation"].astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = simplify_epsilon_frac * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) < 3:
                continue
            ring = [(float(p[0][0] + offset_x), float(p[0][1] + offset_y))
                    for p in simplified]
            area = float(cv2.contourArea(contour))
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid = (float(moments["m10"] / moments["m00"] + offset_x),
                        float(moments["m01"] / moments["m00"] + offset_y))
            parcel = cadastral.Parcel(
                parcel_id=index,
                pixel_polygon=ring,
                pixel_centroid=centroid,
                area_px=area,
            )
            # Tagged rather than silently returned: every consumer can see that
            # this boundary was inferred from appearance, not traced.
            setattr(parcel, "approximate", True)
            setattr(parcel, "method", "sam2")
            parcels.append(parcel)
        return parcels
    except Exception:
        return []


def segment_if_empty(image_path: str, parcels: List["cadastral.Parcel"]) -> tuple:
    """
    The entry point callers should use.

    Returns (parcels, note). When the primary vectoriser found anything at all
    its result is returned untouched - SAM does not get to second-guess a
    traced boundary. Only a genuinely empty result opens this path.
    """
    if parcels:
        return parcels, None
    if not available():
        return parcels, None
    started = time.time()
    approximate = segment(image_path)
    if not approximate:
        return parcels, None
    return approximate, {
        "rule": "PARCELS_APPROXIMATE",
        "severity": "warning",
        "method": "sam2",
        "model": SAM_MODEL,
        "count": len(approximate),
        "elapsed_ms": int((time.time() - started) * 1000),
        "message": (
            f"No drawn parcel boundary could be traced on this map, so "
            f"{len(approximate)} parcel outline(s) were estimated by image "
            f"segmentation instead. These are APPROXIMATE: the boundaries "
            f"follow what the image looks like, not a surveyed line, and "
            f"their areas must not be treated as measurements."),
        "suggestion": ("Confirm every boundary against the source sheet before "
                       "approving, or re-scan the map at higher quality."),
    }
