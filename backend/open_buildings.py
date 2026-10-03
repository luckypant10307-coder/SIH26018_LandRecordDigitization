"""
Building footprints from Google Open Buildings v3.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

WHAT THIS FIXES

Every vertical volume in this system currently carries the WHOLE PARCEL as its
footprint, because nothing in a land record says where on the plot the
building actually stands. That single gap is what produces both of the
module's honest non-answers:

    LEVEL_NOT_PARTITIONED    several units on a level, extents unknown
    BUILDINGS_NOT_LOCATED    several buildings on a parcel, positions unknown

Open Buildings v3 supplies the missing outline - 1.8 billion building polygons
across South Asia, each with its own `area_in_meters` and a confidence score.
With a real building footprint, `footprint_is_parcel` becomes False, the
building is located on its plot, and an area in SQUARE METRES replaces an area
in pixels of a scanned map.

IT IS A REGIONAL IMPORT, NOT A PER-PARCEL LOOKUP

This is the opposite of `building_height`, and the difference is worth
understanding before using it. Heights are a tiled raster, so one parcel costs
two HTTP range reads - about 80 KB against a 270 MB tile. Footprints are
gzipped CSV, which cannot be seeked into: reaching the last row means
decompressing every row before it.

The files are one per S2 level-6 cell. Measured across the bucket: 3,330
cells, median 9 MB - but the Gangetic plain is densely built and the cell
covering Jaunpur, where this project's documents come from, is 988 MB.

So a region is imported ONCE, filtered to a bounding box, and kept locally;
after that every parcel inside it is answered offline with no network at all.
`tools/import_buildings.py` does that one-time pass. This module reads the
result, and will stream a cell itself only when explicitly allowed to.

WHAT IT IS NOT

A building outline is a ROOF seen from above. It does not know storeys, it
does not know where one flat ends and the next begins, and it is not a survey.
It locates a building on a parcel and measures its ground area - which is
exactly the gap above, and nothing more.

The data is CC-BY 4.0 / ODbL 1.0, and `ATTRIBUTION` must be shown wherever a
footprint from here is displayed.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# -- configuration ---------------------------------------------------------

ENABLED = os.environ.get("OPEN_BUILDINGS") == "1"

BUCKET = os.environ.get(
    "OPEN_BUILDINGS_BUCKET",
    "https://storage.googleapis.com/open-buildings-data/v3")

# Level 6 rather than level 4: a level-4 cell holds sixteen of these and the
# largest is 8.4 GB, which is not an import anyone will run twice.
CELL_LEVEL = 6
CELL_PREFIX = "polygons_s2_level_6_gzip_no_header"

CACHE_DIR = os.environ.get(
    "OPEN_BUILDINGS_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "storage", "open_buildings"))

REQUEST_TIMEOUT_S = int(os.environ.get("OPEN_BUILDINGS_TIMEOUT", "120"))

ATTRIBUTION = ("Building footprints: Google Open Buildings v3, "
               "CC-BY 4.0 / ODbL 1.0.")

DATASET_NOTE = (
    "A building outline inferred from satellite imagery - the roof seen from "
    "above. It locates a building and measures its ground area; it does not "
    "know storeys or unit boundaries, and it is not a survey.")

# Below this the model is guessing. Open Buildings publishes confidence per
# building and it is kept rather than averaged away, because a 0.6 outline
# used to settle a boundary is a different object from a 0.9 one.
MIN_CONFIDENCE = float(os.environ.get("OPEN_BUILDINGS_MIN_CONFIDENCE", "0.70"))

# A cell this large is a deliberate decision, not something to do by accident
# inside a web request.
STREAM_WARN_BYTES = 200 * 1024 * 1024


class OpenBuildingsError(Exception):
    """A lookup could not be completed."""


def available() -> bool:
    return ENABLED or os.path.isdir(CACHE_DIR)


def unavailable_reason() -> str:
    if available():
        return ""
    return ("OPEN_BUILDINGS != 1 and no local import - vertical volumes keep "
            "the whole parcel as their footprint")


def describe() -> dict:
    return {
        "enabled": ENABLED,
        "dataset": "Google Open Buildings v3 polygons",
        "cells_cached": len(cached_cells()),
        "licence": "CC-BY 4.0 / ODbL 1.0",
        "attribution": ATTRIBUTION,
        "caveat": DATASET_NOTE,
        "access": ("regional import, not a per-parcel fetch: gzipped CSV "
                   "cannot be seeked, and the cell covering Jaunpur is 988 MB"),
    }


# -- S2 cell addressing ----------------------------------------------------
#
# The files are named by S2 cell token, so a coordinate has to be converted
# into one. Written out rather than taking a dependency, in the same spirit as
# the UTM transform in building_height: it is arithmetic, and the whole point
# of these modules is to add nothing to the install.
#
# Verified against the live bucket: Jaunpur, Amari, Lucknow, Hinjewadi and
# Vansar all resolve to tokens that exist as files.

_SWAP, _INVERT = 1, 2
_POS_TO_IJ = [[0, 1, 3, 2], [0, 2, 3, 1], [3, 2, 0, 1], [3, 1, 0, 2]]
_POS_TO_ORIENT = [_SWAP, 0, 0, _INVERT | _SWAP]
_IJ_TO_POS = [[row.index(ij) for ij in range(4)] for row in _POS_TO_IJ]


def _xyz(lat: float, lon: float) -> Tuple[float, float, float]:
    p, l = math.radians(lat), math.radians(lon)
    return (math.cos(p) * math.cos(l), math.cos(p) * math.sin(l), math.sin(p))


def _face_uv(x: float, y: float, z: float) -> Tuple[int, float, float]:
    magnitudes = (abs(x), abs(y), abs(z))
    face = magnitudes.index(max(magnitudes))
    if (x, y, z)[face] < 0:
        face += 3
    return [(face, y / x, z / x),
            (face, -x / y, z / y),
            (face, -y / z, -x / z),
            (face, z / x, y / x),
            (face, z / y, -x / y),
            (face, -y / z, -x / z)][face]


def _uv_to_st(u: float) -> float:
    """S2's quadratic projection, which keeps cell areas closer to equal."""
    return 0.5 * math.sqrt(1 + 3 * u) if u >= 0 else 1 - 0.5 * math.sqrt(1 - 3 * u)


def _st_to_ij(s: float) -> int:
    return max(0, min(2 ** 30 - 1, int(math.floor(s * 2 ** 30))))


def cell_token(lat: float, lon: float, level: int = CELL_LEVEL) -> str:
    """The S2 cell token containing a point, which is also the file name."""
    face, u, v = _face_uv(*_xyz(lat, lon))
    i, j = _st_to_ij(_uv_to_st(u)), _st_to_ij(_uv_to_st(v))

    # Walk the Hilbert curve from the most significant bit down, carrying the
    # orientation forward - this is what makes neighbouring cells adjacent in
    # the ordering, and why rows for nearby buildings sit near each other in
    # the file.
    orientation = face & _SWAP
    position = 0
    for k in range(29, 29 - level, -1):
        bit_i, bit_j = (i >> k) & 1, (j >> k) & 1
        quadrant = _IJ_TO_POS[orientation][(bit_i << 1) | bit_j]
        position = (position << 2) | quadrant
        orientation ^= _POS_TO_ORIENT[quadrant]

    cell_id = ((face << 61) | (position << (61 - 2 * level))
               | (1 << (60 - 2 * level)))
    return format(cell_id, "016x").rstrip("0")


def cell_url(token: str) -> str:
    return f"{BUCKET}/{CELL_PREFIX}/{token}_buildings.csv.gz"


def cell_size(token: str) -> Optional[int]:
    """Bytes of a cell file, so a caller can refuse before committing."""
    try:
        request = urllib.request.Request(cell_url(token), method="HEAD")
        request.add_header("User-Agent", "LandRecordDigitization/1.0 (SIH 2026)")
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return int(response.headers.get("Content-Length") or 0)
    except (urllib.error.URLError, OSError, ValueError):
        return None


# -- geometry --------------------------------------------------------------

_POLYGON = re.compile(r"POLYGON\s*\(\((.*?)\)\)", re.I | re.S)


def parse_wkt_polygon(wkt: str) -> List[Tuple[float, float]]:
    """
    The exterior ring of a WKT POLYGON, as lon/lat pairs.

    Only the exterior ring: Open Buildings outlines are simple, and a hole in
    a building footprint would be a courtyard this dataset does not model.
    """
    match = _POLYGON.search(wkt or "")
    if not match:
        raise OpenBuildingsError("Not a WKT POLYGON.")
    ring: List[Tuple[float, float]] = []
    for pair in match.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            continue
        ring.append((float(parts[0]), float(parts[1])))
    # WKT closes its rings; the rest of this system does not.
    if len(ring) > 3 and ring[0] == ring[-1]:
        ring.pop()
    if len(ring) < 3:
        raise OpenBuildingsError("Polygon ring has fewer than three points.")
    return ring


def ring_bounds(ring: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (min(lons), min(lats), max(lons), max(lats))


def point_in_ring(lon: float, lat: float,
                  ring: Sequence[Tuple[float, float]]) -> bool:
    """Ray casting. Used to ask whether a building sits inside a parcel."""
    inside = False
    count = len(ring)
    for index in range(count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % count]
        if (y1 > lat) != (y2 > lat):
            x_at = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if x_at > lon:
                inside = not inside
    return inside


def centroid(ring: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    return (sum(p[0] for p in ring) / len(ring),
            sum(p[1] for p in ring) / len(ring))


# -- local store -----------------------------------------------------------

def cached_cells() -> List[str]:
    if not os.path.isdir(CACHE_DIR):
        return []
    return sorted(name[:-7] for name in os.listdir(CACHE_DIR)
                  if name.endswith(".ndjson"))


def _subset_path(token: str) -> str:
    return os.path.join(CACHE_DIR, token + ".ndjson")


def store_subset(token: str, buildings: Iterable[dict]) -> int:
    """Write an imported subset. One JSON object per line, so it streams back."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    written = 0
    with open(_subset_path(token), "w", encoding="utf-8") as handle:
        for building in buildings:
            handle.write(json.dumps(building, ensure_ascii=False) + "\n")
            written += 1
    return written


def read_subset(token: str) -> List[dict]:
    path = _subset_path(token)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# -- the import pass -------------------------------------------------------

def stream_cell(token: str, bounds: Tuple[float, float, float, float],
                min_confidence: float = MIN_CONFIDENCE,
                progress=None) -> List[dict]:
    """
    Stream one S2 cell and keep only the buildings inside `bounds`.

    Decompresses as it downloads rather than buffering the file, because these
    run to hundreds of megabytes and the whole point is to end up with a few
    thousand rows. Nothing is written here; the caller decides what to store.
    """
    if not ENABLED:
        raise OpenBuildingsError(unavailable_reason())

    west, south, east, north = bounds
    kept: List[dict] = []
    seen = 0

    request = urllib.request.Request(
        cell_url(token),
        headers={"User-Agent": "LandRecordDigitization/1.0 (SIH 2026)"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            with gzip.GzipFile(fileobj=response) as stream:
                text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
                for row in csv.reader(text):
                    if len(row) < 5:
                        continue
                    seen += 1
                    if progress and seen % 250000 == 0:
                        progress(seen, len(kept))
                    try:
                        lat, lon = float(row[0]), float(row[1])
                    except ValueError:
                        continue          # the header row, if one is present
                    if not (west <= lon <= east and south <= lat <= north):
                        continue
                    try:
                        confidence = float(row[3])
                    except ValueError:
                        continue
                    if confidence < min_confidence:
                        continue
                    try:
                        ring = parse_wkt_polygon(row[4])
                    except (OpenBuildingsError, ValueError):
                        continue
                    kept.append({
                        "lat": lat, "lon": lon,
                        "area_m2": float(row[2]) if row[2] else None,
                        "confidence": round(confidence, 3),
                        "footprint": [list(p) for p in ring],
                    })
    except (urllib.error.URLError, OSError, EOFError) as exc:
        raise OpenBuildingsError(f"Could not stream cell {token}: {exc}")

    return kept


# -- the public call -------------------------------------------------------

def buildings_in_parcel(parcel_ring: Sequence[Tuple[float, float]]) -> Optional[dict]:
    """
    The buildings standing on a parcel, or None.

    Returns None rather than raising on every failure - no import for this
    region, no coordinates, nothing found - so a caller keeps the parcel
    footprint it already had. This can add a building outline; it cannot take
    one away.

    THE RING MUST BE LON/LAT DEGREES, for the same reason as everywhere else
    in this system: most footprints here are in pixels of a scanned map, and
    (246, 20) is a real coordinate in the Atlantic.
    """
    if not parcel_ring or len(parcel_ring) < 3:
        return None

    lons = [float(p[0]) for p in parcel_ring]
    lats = [float(p[1]) for p in parcel_ring]
    if not (all(-180 <= x <= 180 for x in lons)
            and all(-90 <= y <= 90 for y in lats)):
        return None
    if max(lons) - min(lons) > 0.5 or max(lats) - min(lats) > 0.5:
        return None               # a pixel ring that happens to look valid

    ring = [(float(p[0]), float(p[1])) for p in parcel_ring]
    token = cell_token(sum(lats) / len(lats), sum(lons) / len(lons))
    candidates = read_subset(token)
    if not candidates:
        return None

    inside = []
    for building in candidates:
        lon, lat = building["lon"], building["lat"]
        if point_in_ring(lon, lat, ring):
            inside.append(building)

    if not inside:
        return None

    inside.sort(key=lambda b: b.get("area_m2") or 0, reverse=True)
    return {
        "source": "open_buildings_v3",
        "cell": token,
        "count": len(inside),
        "buildings": inside,
        "largest": inside[0],
        "total_area_m2": round(sum(b.get("area_m2") or 0 for b in inside), 1),
        "attribution": ATTRIBUTION,
        "caveat": DATASET_NOTE,
    }
