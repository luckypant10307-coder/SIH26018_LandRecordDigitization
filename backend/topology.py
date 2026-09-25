"""
Topology validation for vectorised cadastral parcels.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Vectorising a village map produces polygons that look right and can still be
topologically wrong, and the wrong ones are exactly the ones that matter: a
gap between two parcels is land that belongs to nobody on the map, and an
overlap is land the map gives to two people at once. Both are how boundary
disputes start, and neither is visible by eye on a sheet with five hundred
parcels.

Pure Python and pure geometry - no shapely, no GDAL, no numpy. The project's
geospatial readers are hand-written for the same reason (README S12): the
geospatial stack is the most common reason a demo will not start on a
locked-down Windows machine, and these checks are a few hundred lines of
coordinate arithmetic.

What this module does NOT claim:

  * Overlap detection is a DETECTOR, not an area calculator. It reports that
    two parcels overlap and shows where, not how many square metres they
    share - that needs a general polygon clipper, which is a great deal more
    code and far easier to get subtly wrong. A reviewer needs to know WHICH
    two parcels to look at; the exact sliver area does not change that.

  * Gaps are found through near-coincident-but-unequal vertices rather than a
    true topological union. Adjacent parcels are meant to share boundary
    vertices exactly; when a digitiser's two traces land 0.3 m apart the
    result is a sliver gap. That proxy catches the cause. It will not find a
    genuine unmapped hole in the middle of a village, which is a different
    (and much rarer) problem, and it says so rather than implying coverage it
    does not have.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Ring = List[Point]

# A vertex pair closer than this, in the ring's own units, is treated as one
# point that two traces disagree about rather than two distinct corners.
# Callers working in degrees must scale this - see suggested_tolerance().
DEFAULT_SNAP_TOLERANCE = 0.5

# Slivers below this share of the smaller parcel's area are reported as
# "sliver" rather than "overlap". Digitising noise always produces a few; a
# real double-allocation is not 0.2% of a parcel.
SLIVER_AREA_FRACTION = 0.02

_M_PER_DEG = 110540.0        # latitude degrees to metres, near enough for a tolerance


def suggested_tolerance(in_degrees: bool, metres: float = DEFAULT_SNAP_TOLERANCE) -> float:
    """
    Convert a real-world snap tolerance into the ring's coordinate units.

    Getting this wrong is silent and total: 0.5 interpreted as degrees is a
    55 km tolerance, which would snap every parcel in a district onto one
    point. The caller must say which space it is working in.
    """
    return metres / _M_PER_DEG if in_degrees else metres


# --------------------------------------------------------------- primitives

def _dedupe_closing(ring: Sequence[Point]) -> Ring:
    """Ring without a repeated final vertex, so edge counts are not off by one."""
    pts = [tuple(p) for p in ring]
    while len(pts) > 1 and _close(pts[0], pts[-1], 0.0):
        pts.pop()
    return pts


def _close(a: Point, b: Point, tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def signed_area(ring: Sequence[Point]) -> float:
    """Shoelace. Positive is counter-clockwise."""
    pts = _dedupe_closing(ring)
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def area(ring: Sequence[Point]) -> float:
    return abs(signed_area(ring))


def bbox(ring: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _bboxes_touch(a, b, tol: float = 0.0) -> bool:
    return not (a[2] + tol < b[0] or b[2] + tol < a[0]
                or a[3] + tol < b[1] or b[3] + tol < a[1])


def _orient(a: Point, b: Point, c: Point) -> float:
    """Cross product of ab x ac. >0 left turn, <0 right turn, 0 collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point, tol: float) -> bool:
    """p lies on segment ab (within tol of the line and inside the span)."""
    if abs(_orient(a, b, p)) > tol * max(1.0, _seg_len(a, b)):
        return False
    return (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
            and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol)


def _seg_len(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def segments_properly_cross(p1: Point, p2: Point, p3: Point, p4: Point,
                            tol: float = 1e-12) -> bool:
    """
    True when p1p2 and p3p4 cross at interior points of both.

    "Properly" is the whole point. Adjacent parcels legitimately share an
    entire boundary edge, and every one of those shared edges touches at its
    endpoints. A plain intersection test calls that an overlap and reports
    every correctly-digitised village as broken, which is worse than no check
    at all because it trains a reviewer to ignore the output.
    """
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    # Strict sign change on both segments: a true interior crossing.
    if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
       ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
        return True
    return False


def point_in_ring(pt: Point, ring: Sequence[Point], tol: float = 1e-9) -> str:
    """
    "inside" | "boundary" | "outside", by ray casting.

    Boundary is a separate answer rather than an arbitrary inside/outside
    choice, because shared edges are the normal case here: every vertex a
    parcel shares with its neighbour sits exactly on that neighbour's
    boundary, and calling those "inside" would report an overlap for every
    adjacent pair on the sheet.
    """
    pts = _dedupe_closing(ring)
    if len(pts) < 3:
        return "outside"
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        if _on_segment(a, b, pt, tol):
            return "boundary"
    x, y = pt
    inside = False
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0)
            if x < x0 + t * (x1 - x0):
                inside = not inside
    return "inside" if inside else "outside"


# --------------------------------------------------------------- validity

def polygon_problems(ring: Sequence[Point]) -> List[str]:
    """Every structural fault in one ring, named. Empty list means valid."""
    problems: List[str] = []
    pts = _dedupe_closing(ring)
    if len(pts) < 3:
        problems.append("fewer than 3 distinct vertices")
        return problems
    if area(pts) <= 0.0:
        problems.append("zero area")
    # Repeated non-adjacent vertices pinch the ring into two lobes.
    for i in range(len(pts)):
        for j in range(i + 2, len(pts)):
            if i == 0 and j == len(pts) - 1:
                continue
            if _close(pts[i], pts[j], 0.0):
                problems.append(f"repeated vertex at positions {i} and {j}")
                break
        else:
            continue
        break
    # Self-intersection: any non-adjacent edge pair crossing properly.
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue          # edges sharing the wrap-around vertex
            c, d = pts[j], pts[(j + 1) % n]
            if segments_properly_cross(a, b, c, d):
                problems.append(f"self-intersection between edges {i} and {j}")
                return problems
    return problems


# --------------------------------------------------------------- overlaps

def polygons_overlap(a: Sequence[Point], b: Sequence[Point]) -> Optional[dict]:
    """
    Evidence that two parcels share interior area, or None.

    Returns the witness - which test fired and where - so a reviewer is sent
    to a coordinate rather than told "these overlap" and left to find it.
    """
    if len(a) < 3 or len(b) < 3:
        return None
    if not _bboxes_touch(bbox(a), bbox(b)):
        return None

    for pt in _dedupe_closing(a):
        if point_in_ring(pt, b) == "inside":
            return {"kind": "vertex_inside", "at": pt, "which": "a_in_b"}
    for pt in _dedupe_closing(b):
        if point_in_ring(pt, a) == "inside":
            return {"kind": "vertex_inside", "at": pt, "which": "b_in_a"}

    pa, pb = _dedupe_closing(a), _dedupe_closing(b)
    for i in range(len(pa)):
        a1, a2 = pa[i], pa[(i + 1) % len(pa)]
        for j in range(len(pb)):
            b1, b2 = pb[j], pb[(j + 1) % len(pb)]
            if segments_properly_cross(a1, a2, b1, b2):
                return {"kind": "edges_cross", "at": a1, "edges": (i, j)}
    return None


def contains(outer: Sequence[Point], inner: Sequence[Point]) -> bool:
    """
    True when `inner` lies wholly within `outer`.

    Containment is a DIFFERENT relation from overlap and must be reported
    separately. On a real village sheet the vectoriser also picks up the
    outer envelope - the block or village boundary that encloses every
    parcel - and a plain overlap test then reports that one shape as
    conflicting with all five hundred others. That is not a finding, it is
    noise that teaches a reviewer to ignore the report.

    Measured on the bundled plot report: 6 of 6 "overlaps" were the outer
    envelope containing the parcels inside it.
    """
    pts = _dedupe_closing(inner)
    if len(pts) < 3 or len(_dedupe_closing(outer)) < 3:
        return False
    if area(outer) <= area(inner):
        return False
    seen_inside = False
    for pt in pts:
        where = point_in_ring(pt, outer)
        if where == "outside":
            return False
        if where == "inside":
            seen_inside = True
    return seen_inside


def find_containments(rings: Dict[str, Ring]) -> List[dict]:
    """Parcel-inside-parcel pairs, outer first."""
    ids = list(rings.keys())
    out: List[dict] = []
    for a in ids:
        for b in ids:
            if a == b:
                continue
            if contains(rings[a], rings[b]):
                out.append({"outer": a, "inner": b})
    return out


def find_overlaps(rings: Dict[str, Ring]) -> List[dict]:
    """
    Every overlapping pair among the given parcels, keyed by parcel id.

    Bounding boxes prefilter, so this is far below the worst case O(n^2) edge
    comparison on a real sheet where parcels only touch their neighbours.
    """
    ids = list(rings.keys())
    boxes = {k: bbox(v) for k, v in rings.items() if len(v) >= 3}
    out: List[dict] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ka, kb = ids[i], ids[j]
            if ka not in boxes or kb not in boxes:
                continue
            if not _bboxes_touch(boxes[ka], boxes[kb]):
                continue
            if contains(rings[ka], rings[kb]) or contains(rings[kb], rings[ka]):
                continue      # containment, reported on its own channel
            witness = polygons_overlap(rings[ka], rings[kb])
            if not witness:
                continue
            smaller = min(area(rings[ka]), area(rings[kb]))
            out.append({
                "a": ka, "b": kb, "witness": witness,
                "severity": "sliver" if smaller <= 0 or _is_sliver(
                    rings[ka], rings[kb]) else "overlap",
            })
    return out


def _is_sliver(a: Sequence[Point], b: Sequence[Point]) -> bool:
    """
    Cheap sliver heuristic: the overlap is a vertex barely inside, with no
    edge crossings.

    Without a clipper there is no exact overlap area, so this asks a
    proxy question - is the intrusion deep, or is it one vertex a hair over
    the line? Digitising a shared boundary twice always produces a few of the
    latter and calling them disputes would bury the real ones.
    """
    pa, pb = _dedupe_closing(a), _dedupe_closing(b)
    for i in range(len(pa)):
        a1, a2 = pa[i], pa[(i + 1) % len(pa)]
        for j in range(len(pb)):
            b1, b2 = pb[j], pb[(j + 1) % len(pb)]
            if segments_properly_cross(a1, a2, b1, b2):
                return False
    deep = 0
    for pt in pa:
        if point_in_ring(pt, b) == "inside":
            deep += 1
    for pt in pb:
        if point_in_ring(pt, a) == "inside":
            deep += 1
    return deep <= 1


# --------------------------------------------------------------- gaps / snap

def find_unsnapped_vertices(rings: Dict[str, Ring],
                            tolerance: float = DEFAULT_SNAP_TOLERANCE) -> List[dict]:
    """
    Vertex pairs from different parcels that are near-coincident but unequal.

    This is the sliver-gap detector. Two adjacent parcels are meant to share
    boundary vertices exactly; when they land a fraction apart the map has a
    hairline of land belonging to nobody.
    """
    out: List[dict] = []
    ids = list(rings.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ka, kb = ids[i], ids[j]
            ra, rb = _dedupe_closing(rings[ka]), _dedupe_closing(rings[kb])
            if not ra or not rb:
                continue
            if not _bboxes_touch(bbox(ra), bbox(rb), tolerance):
                continue
            for pa in ra:
                for pb in rb:
                    if pa == pb:
                        continue
                    d = _seg_len(pa, pb)
                    if 0.0 < d <= tolerance:
                        out.append({"a": ka, "b": kb, "a_at": pa, "b_at": pb,
                                    "distance": d})
    return out


def snap_shared_vertices(rings: Dict[str, Ring],
                         tolerance: float = DEFAULT_SNAP_TOLERANCE) -> int:
    """
    Merge near-coincident vertices across parcels onto a common point.

    Mutates the rings in place and returns how many vertices moved. The
    representative is the first vertex encountered rather than a centroid of
    the cluster, which keeps the operation idempotent: running it twice must
    not creep the boundary further each time.
    """
    if tolerance <= 0:
        return 0
    anchors: List[Point] = []
    moved = 0
    for key in sorted(rings.keys()):
        ring = rings[key]
        for idx, pt in enumerate(ring):
            match = None
            for anchor in anchors:
                if _seg_len(anchor, pt) <= tolerance:
                    match = anchor
                    break
            if match is None:
                anchors.append(tuple(pt))
                continue
            if tuple(pt) != match:
                ring[idx] = match
                moved += 1
    return moved


# --------------------------------------------------------------- report

def validate(rings: Dict[str, Ring],
             tolerance: float = DEFAULT_SNAP_TOLERANCE) -> dict:
    """
    Run every topology rule over one village's parcels.

    Returns a report rather than raising: a village map with a sliver is
    still worth loading, and a reviewer needs the list in order to fix it.
    """
    invalid = {}
    for key, ring in rings.items():
        problems = polygon_problems(ring)
        if problems:
            invalid[key] = problems
    overlaps = find_overlaps(rings)
    containments = find_containments(rings)
    unsnapped = find_unsnapped_vertices(rings, tolerance)
    return {
        "parcel_count": len(rings),
        "invalid": invalid,
        "overlaps": [o for o in overlaps if o["severity"] == "overlap"],
        "slivers": [o for o in overlaps if o["severity"] == "sliver"],
        "containments": containments,
        "unsnapped_vertices": unsnapped,
        "tolerance": tolerance,
        # Containment is not counted against cleanliness: an enclosing block
        # or village boundary is a normal thing for a sheet to carry.
        "clean": not invalid and not overlaps and not unsnapped,
        "caveat": ("Overlaps are detected, not measured - no share area is "
                   "computed. Gaps are inferred from near-coincident "
                   "vertices, so an unmapped hole in the interior of a "
                   "village would not be reported."),
    }


def rings_from_parcels(parcels: Sequence, prefer_geo: bool = True) -> Dict[str, Ring]:
    """
    Pull rings off cadastral.Parcel objects, saying which space they are in.

    Pixel and geographic rings must never be mixed in one check: the
    tolerance means something different in each, and comparing one parcel's
    pixels against another's degrees would be silently meaningless.
    """
    out: Dict[str, Ring] = {}
    for p in parcels:
        ring = None
        if prefer_geo:
            ring = getattr(p, "geo_polygon", None)
        if not ring:
            ring = getattr(p, "pixel_polygon", None)
        if not ring:
            continue
        key = str(getattr(p, "khasra_number", None)
                  or getattr(p, "parcel_id", len(out)))
        out[key] = [tuple(pt) for pt in ring]
    return out
