"""
Generate a synthetic cadastral (parcel-boundary) map for demonstrating
vectorization and georeferencing.

There is no real cadastral map, and no real ground-control-point survey,
available to this project - the same honesty gap README S12/S13 already
disclose for the DILRMP connector and the bundled admin/registry master
data. This generates a plausible-looking but entirely synthetic village
parcel map instead: a set of irregular quadrilateral parcels, each labelled
with a khasra number, plus four corner ground control points with
illustrative (NOT surveyed) coordinates loosely centred on Lucknow -
continuing the Narharpur/Lucknow village already used in the text-record
demo samples (sample_01/03), so the two demos tell one consistent story.

The pipeline this feeds (backend/cadastral.py) is real: contour detection,
polygon simplification and affine georeferencing work identically on a real
scanned map and real surveyed control points once they exist.

Run:  python3 tools/make_cadastral_map.py
"""

from __future__ import annotations

import json
import os
import random
import sys

try:
    import cv2
    import numpy as np
except Exception:
    print("OpenCV/numpy are required to generate the cadastral demo map "
          "(pip install opencv-python numpy).")
    sys.exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(ROOT, "samples", "cadastral")

W, H = 900, 900
N_PARCELS = 14
MARGIN = 40


def _split_rects(rects, n, rnd, min_size=90):
    """Recursively guillotine-split the largest rectangle until `n` remain."""
    while len(rects) < n:
        idx = max(range(len(rects)),
                  key=lambda i: (rects[i][2] - rects[i][0]) * (rects[i][3] - rects[i][1]))
        x0, y0, x1, y1 = rects.pop(idx)
        w, h = x1 - x0, y1 - y0
        if w < min_size * 2 and h < min_size * 2:
            rects.insert(idx, (x0, y0, x1, y1))
            break
        if w >= h:
            cut = x0 + rnd.uniform(0.35, 0.65) * w
            rects.append((x0, y0, cut, y1))
            rects.append((cut, y0, x1, y1))
        else:
            cut = y0 + rnd.uniform(0.35, 0.65) * h
            rects.append((x0, y0, x1, cut))
            rects.append((x0, cut, x1, y1))
    return rects


def _jitter_quad(rect, rnd, jitter=6):
    """Nudge a rectangle's corners so parcels look hand-surveyed, not CAD-perfect."""
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(x + rnd.uniform(-jitter, jitter), y + rnd.uniform(-jitter, jitter))
            for x, y in corners]


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    rnd = random.Random(42)

    rects = _split_rects([(MARGIN, MARGIN, W - MARGIN, H - MARGIN)], N_PARCELS, rnd)
    canvas = np.full((H, W), 255, dtype=np.uint8)

    ground_truth = []
    for i, rect in enumerate(rects):
        quad = _jitter_quad(rect, rnd)
        pts = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=0, thickness=3)
        cx = sum(p[0] for p in quad) / 4.0
        cy = sum(p[1] for p in quad) / 4.0
        khasra = f"{200 + i}/{rnd.randint(1, 3)}"
        cv2.putText(canvas, khasra, (int(cx) - 26, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        ground_truth.append({
            "khasra_number": khasra,
            "pixel_quad": [list(p) for p in quad],
            "pixel_centroid": [cx, cy],
        })

    # Explicit border frame around the plotted parcels - real map exports
    # (verified against actual Bhu-Naksha plot reports) draw exactly this
    # kind of panel border, distinct and much larger than any single parcel,
    # which is what backend/cadastral.py's detect_map_frame() looks for to
    # separate the map panel from surrounding title/legend/table content.
    # Without it, the largest contour on this synthetic map would just be
    # the parcels' own outer hull - not a real frame - which is the exact
    # false-positive detect_map_frame is meant to avoid.
    border_margin = 12
    cv2.rectangle(canvas, (border_margin, border_margin),
                  (W - border_margin, H - border_margin), 0, 2)

    # Corner ground control points: crosshair marks on the image, illustrative
    # (not surveyed) coordinates in the sidecar file below.
    gcps = [
        {"pixel": [0, 0], "lon": 80.9490, "lat": 26.8550},
        {"pixel": [W, 0], "lon": 80.9530, "lat": 26.8550},
        {"pixel": [0, H], "lon": 80.9490, "lat": 26.8510},
        {"pixel": [W, H], "lon": 80.9530, "lat": 26.8510},
    ]
    for gcp in gcps:
        x, y = gcp["pixel"]
        x, y = max(8, min(W - 8, x)), max(8, min(H - 8, y))
        cv2.drawMarker(canvas, (x, y), 0, cv2.MARKER_CROSS, 18, 2)

    cv2.putText(canvas, "SYNTHETIC MAP - FOR DEMONSTRATION ONLY, NOT A REAL SURVEY",
                (MARGIN, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)

    map_path = os.path.join(OUT_DIR, "village_map_narharpur.png")
    cv2.imwrite(map_path, canvas)

    with open(os.path.join(OUT_DIR, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump({"image": os.path.basename(map_path), "parcels": ground_truth},
                  fh, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "control_points.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "_comment": "Illustrative placeholder ground control points for the "
                        "synthetic demo map only - not a real survey. Loosely "
                        "centred on Lucknow for continuity with sample_01/03's "
                        "Narharpur village, not a claim about its real location.",
            # The village this map covers. Record-to-parcel matching is scoped
            # to it (server._parcel_for_record) so that holding several
            # villages' maps at once cannot cross-match their khasra numbers,
            # which are only unique within a village. Written here rather than
            # added by hand, so regenerating the map does not silently drop
            # the scoping and re-open that hole.
            "village": "नरहरपुर",
            "village_aliases": ["नरहरपुर", "Narharpur"],
            "district": "Lucknow",
            "control_points": gcps,
        }, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {map_path}")
    print(f"Wrote {len(ground_truth)} parcels of ground truth")
    print(f"Wrote {len(gcps)} ground control points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
