#!/usr/bin/env python3
"""
Training data for learned parcel-boundary extraction.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Why this exists
---------------
The classical vectoriser in backend/cadastral.py is excellent on a clean
sheet and collapses on a damaged one. Measured on the bundled Bhu-Naksha
plot report:

    undegraded              7 of 7 parcels
    5% line breakage        0 of 7
    20% line breakage       1 of 7
    85% fade                7 of 7
    noise sigma 25          2 of 7

Fading is survivable because thresholding handles contrast. Broken linework
is fatal, because contour extraction needs a topologically CLOSED boundary
and a single gap lets the region leak away. That is the failure a learned
model is for: it can complete a boundary because it learns what a boundary
looks like, rather than requiring closure.

Why synthetic
-------------
A U-Net needs thousands of annotated sheets and this project has one real
one. Annotating by hand is not a route. But a drawn map knows its own
geometry, so generating the sheet yields pixel-exact ground truth for free -
the same trick tools/train_denoiser.py already uses, including its
held-out-degradation-family discipline, so an honest out-of-family number
can be reported rather than an in-family one.

What the mask contains
----------------------
Boundary lines ONLY. Khasra labels, the north arrow, the scale bar and the
sheet frame are all drawn into the IMAGE and deliberately left out of the
MASK. That difference is the entire learning signal: a model that merely
finds dark pixels will score badly, because it must learn that a numeral
inside a parcel is not a boundary and the frame around the sheet is not a
parcel edge. Without those distractors the task would be thresholding with
extra steps.

Usage
-----
    python3 tools/make_boundary_dataset.py --count 2000
    python3 tools/make_boundary_dataset.py --count 400 --holdout breaks
    python3 tools/make_boundary_dataset.py --count 8 --preview
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

try:
    import cv2
    import numpy as np
except Exception:
    print("OpenCV and numpy are required (pip install opencv-python numpy).")
    sys.exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(ROOT, "storage", "boundary_dataset")

TILE = 512
BOUNDARY_WIDTH = 3          # mask stroke width; the image varies around this

# The degradation families. Named so one can be held out and reported
# separately - an in-family score measures memorisation of the corruption,
# not generalisation to a real sheet.
FAMILIES = ("breaks", "fade", "noise", "blur", "bleed", "speckle",
            "stain", "crease", "dashes")

# Not every damage family is equally hard, and picking a holdout without
# measuring that is how an experiment produces a number meaning nothing.
# Classical parcel recall on this generator's own sheets, one family at a
# time, 20 sheets each:
#
#     breaks    0.0%   (20 of 20 sheets yielded ZERO parcels)
#     fade     98.3%
#     blur     99.4%
#     speckle  98.3%
#     stain    98.3%
#     crease  100.0%
#     bleed    98.3%
#     noise    99.4%
#
# The task IS line breakage. Everything else is survivable, so holding out
# any of the others would compare the model against an unchallenged baseline.
#
# `breaks` must therefore stay in training - it is the capability being
# bought. The honest out-of-family test is a DIFFERENT BREAKAGE MECHANISM:
# train on `breaks` (random pixel-level gaps) and evaluate on `dashes`
# (contiguous segments of boundary removed outright). That asks the question
# worth asking - did the model learn to complete a boundary, or just to fill
# small holes?


# --------------------------------------------------------------- geometry

def _subdivide(rect, depth, rnd, min_side=52):
    """
    Recursively split a rectangle into parcel-shaped pieces.

    Real cadastral subdivision is successive partition of a holding, not a
    grid, so recursive splitting reproduces the characteristic mix: a few
    large fields beside clusters of small ones, and the occasional long thin
    strip where a plot was divided along its length. A uniform grid looks
    nothing like a shajra and would teach the model the wrong prior.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if depth <= 0 or (w < min_side * 2 and h < min_side * 2):
        return [rect]
    vertical = w > h if abs(w - h) > min_side else rnd.random() < 0.5
    # Off-centre splits; occasionally very lopsided, which is what produces
    # the long narrow strips that appear on every real sheet.
    ratio = rnd.uniform(0.18, 0.82) if rnd.random() > 0.22 else rnd.uniform(0.08, 0.2)
    if vertical:
        cut = int(x0 + w * ratio)
        if cut - x0 < min_side or x1 - cut < min_side:
            return [rect]
        left = _subdivide((x0, y0, cut, y1), depth - 1, rnd, min_side)
        right = _subdivide((cut, y0, x1, y1), depth - 1, rnd, min_side)
        return left + right
    cut = int(y0 + h * ratio)
    if cut - y0 < min_side or y1 - cut < min_side:
        return [rect]
    top = _subdivide((x0, y0, x1, cut), depth - 1, rnd, min_side)
    bottom = _subdivide((x0, cut, x1, y1), depth - 1, rnd, min_side)
    return top + bottom


def _shared_jitter(rects, rnd, amount=5):
    """
    Perturb corners so boundaries are not axis-aligned, keeping shared
    corners shared.

    Jittering each parcel independently would pull adjacent parcels apart and
    manufacture the exact sliver gaps topology.py exists to detect - the
    dataset would then be teaching the model that correct sheets contain
    gaps. Snapping to a shared lookup keyed on the original corner keeps
    neighbours welded together.
    """
    moved = {}

    def key(pt):
        return (int(round(pt[0])), int(round(pt[1])))

    def shift(pt):
        k = key(pt)
        if k not in moved:
            moved[k] = (pt[0] + rnd.uniform(-amount, amount),
                        pt[1] + rnd.uniform(-amount, amount))
        return moved[k]

    polys = []
    for (x0, y0, x1, y1) in rects:
        quad = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        polys.append([shift(p) for p in quad])
    return polys


def _warp_network(polys, rnd):
    """
    Apply ONE shared warp to every vertex of the whole parcel network.

    Recursive subdivision produces axis-aligned rectangles, and per-vertex
    jitter big enough to look hand-drawn is also big enough to tear adjacent
    parcels apart. A single transform applied to the entire network escapes
    that trade-off: identical shared corners receive identical displacement,
    so neighbours stay welded while the sheet as a whole acquires the slant,
    taper and wobble of a real shajra.

    Three components, composed:
      * a perspective-like taper, because a sheet traced from a field book is
        never square to the page;
      * a shear, which is what makes parcel corners meet at angles other
        than 90 degrees;
      * a low-frequency sinusoid, standing in for the unsteadiness of a hand
        drawing a long boundary with a straight edge.
    """
    cx = cy = TILE / 2.0
    shear_x = rnd.uniform(-0.18, 0.18)
    shear_y = rnd.uniform(-0.12, 0.12)
    taper = rnd.uniform(-0.00035, 0.00035)
    amp_a = rnd.uniform(0.0, 5.0)
    amp_b = rnd.uniform(0.0, 5.0)
    freq_a = rnd.uniform(0.006, 0.02)
    freq_b = rnd.uniform(0.006, 0.02)
    phase_a = rnd.uniform(0, 6.283)
    phase_b = rnd.uniform(0, 6.283)

    def move(pt):
        x, y = pt[0] - cx, pt[1] - cy
        # shear
        nx = x + shear_x * y
        ny = y + shear_y * x
        # taper: scale grows with distance along one axis
        k = 1.0 + taper * ny
        nx, ny = nx * k, ny * k
        # hand wobble
        nx += amp_a * math.sin(freq_a * (ny + cy) + phase_a)
        ny += amp_b * math.sin(freq_b * (nx + cx) + phase_b)
        return (nx + cx, ny + cy)

    return [[move(p) for p in poly] for poly in polys]


# --------------------------------------------------------------- rendering

def _render(polys, rnd):
    """Return (clean_image, boundary_mask). Mask carries boundaries only."""
    paper = rnd.randint(238, 252)
    img = np.full((TILE, TILE), paper, np.uint8)
    mask = np.zeros((TILE, TILE), np.uint8)

    ink = rnd.randint(10, 60)
    width = rnd.choice([2, 2, 3, 3, 4])

    for poly in polys:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], np.int32)
        cv2.polylines(img, [pts], True, int(ink), width, cv2.LINE_AA)
        # The mask stroke is a fixed width regardless of how thick the drawn
        # line happens to be: the target is WHERE the boundary is, not how
        # heavily the draughtsman pressed.
        cv2.polylines(mask, [pts], True, 255, BOUNDARY_WIDTH, cv2.LINE_8)

    _add_distractors(img, polys, rnd, ink)
    return img, mask


def _add_distractors(img, polys, rnd, ink):
    """
    Everything on a real sheet that is dark but is NOT a parcel boundary.

    These are the whole point of the dataset. A model trained without them
    learns "dark pixel = boundary" and then traces every khasra numeral on a
    real sheet.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Khasra numerals inside most parcels - some left blank, as on the real
    # sheet where the outer envelope carried no number.
    for poly in polys:
        if rnd.random() < 0.12:
            continue
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        label = str(rnd.randint(1, 999))
        if rnd.random() < 0.18:
            label += "/" + str(rnd.randint(1, 9))
        scale = rnd.uniform(0.32, 0.52)
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.putText(img, label, (int(cx - tw / 2), int(cy + th / 2)),
                    font, scale, int(ink), 1, cv2.LINE_AA)

    # Sheet frame: a strong rectangle that is not a parcel edge.
    if rnd.random() < 0.75:
        m = rnd.randint(4, 14)
        cv2.rectangle(img, (m, m), (TILE - m, TILE - m), int(ink),
                      rnd.choice([1, 2]), cv2.LINE_AA)

    # North arrow.
    if rnd.random() < 0.6:
        nx, ny = rnd.randint(40, TILE - 40), rnd.randint(34, 70)
        cv2.arrowedLine(img, (nx, ny + 26), (nx, ny - 18), int(ink), 2,
                        cv2.LINE_AA, tipLength=0.45)
        cv2.putText(img, "N", (nx - 5, ny + 44), font, 0.4, int(ink), 1, cv2.LINE_AA)

    # Scale bar with tick marks - a row of short parallel strokes, which is
    # the distractor most easily mistaken for boundary fragments.
    if rnd.random() < 0.55:
        sx = rnd.randint(30, TILE - 200)
        sy = rnd.randint(TILE - 60, TILE - 24)
        cv2.line(img, (sx, sy), (sx + 150, sy), int(ink), 2, cv2.LINE_AA)
        for t in range(0, 151, 50):
            cv2.line(img, (sx + t, sy - 6), (sx + t, sy + 6), int(ink), 2, cv2.LINE_AA)

    # Hatching inside the occasional parcel - roads, tanks, built-up land.
    if rnd.random() < 0.3 and polys:
        poly = polys[rnd.randrange(len(polys))]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, x1 = int(min(xs)) + 4, int(max(xs)) - 4
        y0, y1 = int(min(ys)) + 4, int(max(ys)) - 4
        if x1 - x0 > 14 and y1 - y0 > 14:
            for yy in range(y0, y1, 7):
                cv2.line(img, (x0, yy), (x1, yy), int(ink), 1, cv2.LINE_AA)


# --------------------------------------------------------------- degradation

def _degrade(img, family, rnd, np_rng):
    """Apply one damage family. Returns (image, strength_used)."""
    out = img.copy()
    if family == "breaks":
        # The fatal one: punch gaps in the ink.
        strength = rnd.uniform(0.03, 0.30)
        dark = out < 140
        ys, xs = np.nonzero(dark)
        if len(ys):
            n = int(len(ys) * strength)
            if n:
                pick = np_rng.choice(len(ys), size=n, replace=False)
                for i in pick:
                    y, x = int(ys[i]), int(xs[i])
                    r = rnd.choice([1, 1, 2])
                    out[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 250
        return out, strength

    if family == "fade":
        strength = rnd.uniform(0.25, 0.85)
        return np.clip(255 - (255 - out.astype(np.float32)) * (1 - strength),
                       0, 255).astype(np.uint8), strength

    if family == "noise":
        strength = rnd.uniform(6, 30)
        return np.clip(out.astype(np.float32) + np_rng.normal(0, strength, out.shape),
                       0, 255).astype(np.uint8), strength

    if family == "blur":
        strength = rnd.uniform(0.6, 2.4)
        return cv2.GaussianBlur(out, (0, 0), strength), strength

    if family == "bleed":
        # Show-through from the reverse of the sheet: a faint mirrored ghost.
        strength = rnd.uniform(0.08, 0.3)
        ghost = cv2.flip(out, 1).astype(np.float32)
        mixed = out.astype(np.float32) * (1 - strength) + ghost * strength
        return np.clip(mixed, 0, 255).astype(np.uint8), strength

    if family == "dashes":
        # Every boundary turned into a dashed line, by erasing ink wherever a
        # periodic stripe pattern falls. Two reasons this is a stripe rather
        # than scattered patches: cadastral sheets really do draw some
        # boundaries dashed, and - measured - scattered patches only reached
        # 94% classical recall because they miss most parcels entirely,
        # whereas a stripe crosses EVERY boundary on the sheet and so breaks
        # closure everywhere, the way `breaks` does.
        #
        # Mechanically different from `breaks` all the same: the gaps here are
        # long, clean and regular instead of scattered single pixels, so a
        # model that only learned to bridge tiny holes cannot pass this.
        strength = rnd.uniform(0.3, 0.6)
        period = rnd.randint(14, 34)
        angle = rnd.uniform(0, math.pi)
        yy, xx = np.mgrid[0:TILE, 0:TILE]
        phase = (xx * math.cos(angle) + yy * math.sin(angle)) % period
        erase = phase < (period * strength)
        out = np.where(erase & (out < 140), 250, out).astype(np.uint8)
        return out, strength

    if family == "stain":
        # Damp blotches and foxing: large irregular dark regions that swallow
        # boundaries underneath them. Hard for contour extraction because the
        # blotch itself becomes a closed region competing with the parcels.
        strength = rnd.uniform(0.04, 0.16)
        n = rnd.randint(2, 6)
        overlay = np.zeros((TILE, TILE), np.float32)
        for _ in range(n):
            cx, cy = rnd.randint(0, TILE), rnd.randint(0, TILE)
            rx, ry = rnd.randint(30, 130), rnd.randint(30, 130)
            blob = np.zeros((TILE, TILE), np.uint8)
            cv2.ellipse(blob, (cx, cy), (rx, ry),
                        rnd.uniform(0, 180), 0, 360, 255, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), rnd.uniform(12, 34))
            overlay = np.maximum(overlay, blob.astype(np.float32) / 255.0)
        depth = strength * 255.0 * rnd.uniform(1.2, 3.0)
        return np.clip(out.astype(np.float32) - overlay * depth,
                       0, 255).astype(np.uint8), strength

    if family == "crease":
        # A fold or tear: one or two dark lines right across the sheet, at an
        # angle unrelated to any parcel edge. The failure mode is a false
        # boundary that cuts real parcels in two, which is different from
        # anything the other families do.
        strength = rnd.uniform(0.3, 1.0)
        for _ in range(rnd.randint(1, 2)):
            if rnd.random() < 0.5:
                p0 = (rnd.randint(0, TILE), 0)
                p1 = (rnd.randint(0, TILE), TILE)
            else:
                p0 = (0, rnd.randint(0, TILE))
                p1 = (TILE, rnd.randint(0, TILE))
            layer = np.zeros((TILE, TILE), np.uint8)
            cv2.line(layer, p0, p1, 255, rnd.choice([1, 2, 3]), cv2.LINE_AA)
            layer = cv2.GaussianBlur(layer, (0, 0), rnd.uniform(0.6, 2.0))
            out = np.clip(out.astype(np.float32)
                          - layer.astype(np.float32) * strength * 0.75,
                          0, 255).astype(np.uint8)
        return out, strength

    if family == "speckle":
        # Foxing and dust: isolated dark specks that are not boundary.
        strength = rnd.uniform(0.001, 0.02)
        n = int(TILE * TILE * strength)
        ys = np_rng.integers(0, TILE, n)
        xs = np_rng.integers(0, TILE, n)
        for y, x in zip(ys, xs):
            r = rnd.choice([0, 0, 1])
            out[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = rnd.randint(20, 90)
        return out, strength

    return out, 0.0


def _skew(img, mask, rnd):
    """A small rotation, applied to image and mask together."""
    angle = rnd.uniform(-3.5, 3.5)
    if abs(angle) < 0.3:
        return img, mask, 0.0
    m = cv2.getRotationMatrix2D((TILE / 2, TILE / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (TILE, TILE), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    # Nearest-neighbour for the mask: interpolating a binary target invents
    # half-boundary pixels that do not correspond to anything drawn.
    mask = cv2.warpAffine(mask, m, (TILE, TILE), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask, angle


# --------------------------------------------------------------- sample

def make_sample(seed, families):
    rnd = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    margin = rnd.randint(18, 46)
    depth = rnd.randint(2, 5)
    rects = _subdivide((margin, margin, TILE - margin, TILE - margin), depth, rnd)
    polys = _shared_jitter(rects, rnd, amount=rnd.uniform(2, 6))
    polys = _warp_network(polys, rnd)
    # Clamp back inside the tile: the warp can push a corner off the sheet,
    # and a boundary clipped at the edge is a boundary the mask claims exists
    # where no line was drawn.
    polys = [[(min(max(x, 1.0), TILE - 2.0), min(max(y, 1.0), TILE - 2.0))
              for x, y in poly] for poly in polys]

    img, mask = _render(polys, rnd)
    img, mask, angle = _skew(img, mask, rnd)

    applied = []
    # One to three damage families, in a random order - or exactly the one
    # family when the caller is building an out-of-family evaluation set.
    chosen = (list(families) if len(families) == 1
              else rnd.sample(list(families), rnd.randint(1, min(3, len(families)))))
    for family in chosen:
        img, strength = _degrade(img, family, rnd, np_rng)
        applied.append({"family": family, "strength": round(float(strength), 4)})

    return img, mask, {
        "seed": seed,
        "parcels": len(polys),
        "skew_deg": round(angle, 2),
        "degradations": applied,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--holdout", choices=FAMILIES, default=None,
                    help="Exclude this damage family, so it can be scored "
                         "out-of-family later.")
    ap.add_argument("--only", choices=FAMILIES, default=None,
                    help="Apply ONLY this family. Used to build the "
                         "out-of-family evaluation set matching a --holdout "
                         "training run, which is the only number that says "
                         "anything about a damage type never seen in training.")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--preview", action="store_true",
                    help="Also write side-by-side image|mask previews.")
    args = ap.parse_args()

    if args.only and args.holdout:
        print("--only and --holdout together are contradictory.")
        return 1
    families = (args.only,) if args.only else tuple(
        f for f in FAMILIES if f != args.holdout)
    if not families:
        print("Nothing left to apply.")
        return 1

    img_dir = os.path.join(args.out, "images")
    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    prev_dir = os.path.join(args.out, "preview")
    if args.preview:
        os.makedirs(prev_dir, exist_ok=True)

    manifest = []
    written = 0
    for i in range(args.count):
        seed = args.seed + i
        img, mask, meta = make_sample(seed, families)
        name = f"{i:05d}.png"
        ok_i = cv2.imwrite(os.path.join(img_dir, name), img)
        ok_m = cv2.imwrite(os.path.join(mask_dir, name), mask)
        # Unchecked imwrite has silently dropped files in this project before
        # (a path at exactly MAX_PATH); a dataset short of a few pairs would
        # be far harder to notice than a crash.
        if not (ok_i and ok_m):
            print(f"  ! failed to write {name} - stopping")
            break
        if args.preview:
            cv2.imwrite(os.path.join(prev_dir, name),
                        np.hstack([img, np.where(mask > 0, 0, 255).astype(np.uint8)]))
        meta["file"] = name
        manifest.append(meta)
        written += 1
        if written % 250 == 0:
            print(f"  {written}/{args.count}")

    payload = {
        "only": args.only,
        "tile": TILE,
        "boundary_width": BOUNDARY_WIDTH,
        "count": written,
        "families_used": list(families),
        "holdout": args.holdout,
        "samples": manifest,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    counts = {}
    for m in manifest:
        for d in m["degradations"]:
            counts[d["family"]] = counts.get(d["family"], 0) + 1
    print(f"\nwrote {written} pairs to {args.out}")
    print(f"  tile {TILE}x{TILE}, boundary width {BOUNDARY_WIDTH}px")
    print(f"  parcels per sheet: "
          f"{min(m['parcels'] for m in manifest)}-{max(m['parcels'] for m in manifest)}"
          if manifest else "")
    print(f"  held out: {args.holdout or '(none)'}")
    for f in FAMILIES:
        print(f"    {f:9} {counts.get(f, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
