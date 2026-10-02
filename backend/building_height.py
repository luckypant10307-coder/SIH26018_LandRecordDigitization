"""
Measured building heights from Google Open Buildings 2.5D Temporal.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

WHY THIS EXISTS

Until now every vertical volume in this system was DECLARED: an operator typed
a floor count and the heights came from a nominal 3 m storey. That is honest
but weak - nothing in the record contradicts a claim of twelve floors on a
single-storey shop.

This module supplies the missing evidence. Open Buildings 2.5D Temporal gives
building height in metres, from Sentinel-2, annually 2016-2023, covering South
Asia including all of India. Critically, it is height RELATIVE TO THE TERRAIN,
which is the same datum `vertical.py` already uses for base_m and top_m - so
there is no datum conversion to get wrong.

WHAT THIS IS AND IS NOT

    REMOTE-SENSED, NOT SURVEYED. It measures a building ENVELOPE from
    satellite imagery. It does not know storeys, it does not know unit
    boundaries, and it is not a legal survey. It can say "the building on
    this ground is about 9 m tall", which is enough to contradict a claim of
    twelve floors and NOT enough to register anyone's flat.

So a height from here produces a third provenance state, between a declared
guess and a survey, and the code never lets it be called the latter.

THE RESOLUTION MATTERS

4 m effective resolution (stored in 0.5 m rasters). A village parcel is a
handful of effective pixels, so a single pixel reading is noise. Every answer
here is a DISTRIBUTION over the footprint - max, 95th percentile, mean, and
the fraction of it that is built at all - because the 95th percentile of a
footprint is a defensible "how tall is the building here" and one pixel is
not.

HOW IT READS 270 MB TILES WITHOUT DOWNLOADING THEM

The source tiles are 25000x25000 at 0.5 m - 270 MB each - which is far too
much to ship or fetch for one parcel. Three properties of the format make a
cheap read possible, and the module depends on all three:

  * the bucket serves HTTP range requests
  * the GeoTIFFs are internally tiled at 512x512 and Deflate-compressed,
    so zlib from the standard library is the only decoder needed
  * PlanarConfiguration is 2 (band-separated), so the height band's tiles
    are contiguous and the other two bands are never fetched

One parcel lookup costs the IFD, the tile-offset table and a single
compressed 512x512 tile: on the order of 80 KB against 270 MB. There is no
GDAL, no rasterio and no Earth Engine dependency - stdlib struct and zlib.

OFF BY DEFAULT, AND NEVER WORSE THAN BEFORE

This reaches a Google-hosted bucket, so it is gated on a flag like
`geocode_online`. What leaves the machine is a pair of coordinates and a byte
range - no owner name, no khasra number, no document text. Every failure path
returns None and the caller keeps its declared geometry, so enabling this can
add evidence and cannot remove any.

ATTRIBUTION IS A LICENCE CONDITION, NOT A COURTESY

The data is CC-BY 4.0 / ODbL 1.0. `ATTRIBUTION` below must be displayed
wherever a height from here is shown, and the API returns it with every
reading so a caller cannot forget.
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

# -- configuration ---------------------------------------------------------

ENABLED = os.environ.get("BUILDING_HEIGHT") == "1"

BUCKET = os.environ.get(
    "BUILDING_HEIGHT_BUCKET",
    "https://storage.googleapis.com/open-buildings-temporal-data")

# 2023 is the last year in v1. A building finished after that is simply not
# here, which is a real limitation and the reason the year is reported with
# every reading rather than assumed to be current.
YEAR = os.environ.get("BUILDING_HEIGHT_YEAR", "2023")
DATASET_VERSION = "v1"

REQUEST_TIMEOUT_S = int(os.environ.get("BUILDING_HEIGHT_TIMEOUT", "30"))

CACHE_DIR = os.environ.get(
    "BUILDING_HEIGHT_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "storage", "building_height"))

ATTRIBUTION = ("Building heights: Google Open Buildings 2.5D Temporal "
               f"({DATASET_VERSION}, {YEAR}), CC-BY 4.0 / ODbL 1.0.")

DATASET_NOTE = (
    "Remotely sensed from Sentinel-2 at 4 m effective resolution; the height "
    "of a building envelope relative to the terrain, not a survey of any "
    "unit's boundaries.")

# The height band. Band 0 is building_fractional_count and band 2 is
# building_presence; neither is read, and PlanarConfig=2 means neither is
# even fetched.
HEIGHT_BAND = 1
NODATA = -99.0

# Values outside this range are not plausible building heights relative to
# terrain. The dataset documents [0, 100]; anything else means a decode fault
# or a corrupt tile, and is dropped rather than averaged in.
MIN_PLAUSIBLE_M = 0.0
MAX_PLAUSIBLE_M = 100.0

# A footprint needs enough valid pixels for a percentile to mean anything.
# Below this the reading is refused rather than reported with false
# confidence: at 4 m effective resolution a tiny parcel is a couple of
# samples, and the 95th percentile of three numbers is theatre.
MIN_SAMPLES = 8

_LOCK = threading.Lock()
_MANIFEST_CACHE: Dict[str, dict] = {}
_TILE_CACHE: Dict[Tuple[str, int], List[Tuple[float, ...]]] = {}
_MAX_CACHED_TILES = 8


class HeightError(Exception):
    """A lookup could not be completed. Never raised at callers of sample()."""


def available() -> bool:
    return ENABLED


def unavailable_reason() -> str:
    if ENABLED:
        return ""
    return ("BUILDING_HEIGHT != 1 - measured building heights are off, so "
            "vertical volumes stay declared from a nominal storey height")


def describe() -> dict:
    """Capability line for run.py --check."""
    return {
        "enabled": ENABLED,
        "dataset": "Google Open Buildings 2.5D Temporal",
        "version": DATASET_VERSION,
        "year": YEAR,
        "resolution_m": 4.0,
        "datum": "height relative to terrain (same datum as vertical.py)",
        "licence": "CC-BY 4.0 / ODbL 1.0",
        "attribution": ATTRIBUTION,
        "caveat": DATASET_NOTE,
    }


# -- projection ------------------------------------------------------------
#
# The tiles are in UTM, one zone per manifest, so a lon/lat footprint has to
# be projected before it can be addressed. Written out rather than imported
# because the whole point of this module is to add no dependency, and a
# forward UTM transform is thirty lines of arithmetic.

_A = 6378137.0                      # WGS84 semi-major axis
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996


def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) + 1


def utm_forward(lon: float, lat: float) -> Tuple[int, float, float]:
    """
    (zone, easting, northing) in WGS84 UTM for a northern-hemisphere point.

    Southern latitudes would need the 10,000,000 m false northing. They are
    refused rather than quietly mis-projected: this dataset covers Latin
    America too, so a southern footprint is a real possibility and silently
    returning a northern-hemisphere northing would place it on another
    continent.
    """
    if lat < 0:
        raise HeightError(
            f"Latitude {lat} is in the southern hemisphere, which this "
            f"transform does not handle. Indian land records are northern; "
            f"a southern coordinate means the footprint is wrong.")
    if not (-180 <= lon <= 180) or lat > 84:
        raise HeightError(f"Coordinates {lon},{lat} are outside the UTM domain.")

    zone = utm_zone(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    p = math.radians(lat)
    l = math.radians(lon)

    sin_p, cos_p, tan_p = math.sin(p), math.cos(p), math.tan(p)
    N = _A / math.sqrt(1 - _E2 * sin_p ** 2)
    T = tan_p ** 2
    C = _EP2 * cos_p ** 2
    A = (l - lon0) * cos_p

    M = _A * ((1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256) * p
              - (3 * _E2 / 8 + 3 * _E2 ** 2 / 32 + 45 * _E2 ** 3 / 1024) * math.sin(2 * p)
              + (15 * _E2 ** 2 / 256 + 45 * _E2 ** 3 / 1024) * math.sin(4 * p)
              - (35 * _E2 ** 3 / 3072) * math.sin(6 * p))

    easting = _K0 * N * (A + (1 - T + C) * A ** 3 / 6
                         + (5 - 18 * T + T * T + 72 * C - 58 * _EP2) * A ** 5 / 120) + 500000.0
    northing = _K0 * (M + N * tan_p * (A * A / 2
                      + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                      + (61 - 58 * T + T * T + 600 * C - 330 * _EP2) * A ** 6 / 720))
    return zone, easting, northing


# -- HTTP ------------------------------------------------------------------

def _fetch(url: str, start: Optional[int] = None,
           end: Optional[int] = None) -> bytes:
    headers = {"User-Agent": "LandRecordDigitization/1.0 (SIH 2026)"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise HeightError(f"Could not read {url}: {exc}")


def _manifest(zone: int) -> dict:
    """
    The tile index for one UTM zone, cached on disk after the first fetch.

    A manifest is a couple of megabytes and lists every tile's affine
    transform, so caching it is what makes the second lookup in a district
    nearly free - and what lets the whole module work offline once a region
    has been touched.
    """
    key = f"{zone}_{YEAR}"
    with _LOCK:
        if key in _MANIFEST_CACHE:
            return _MANIFEST_CACHE[key]

    local = os.path.join(CACHE_DIR, f"manifest_{key}.json")
    if os.path.exists(local):
        try:
            with open(local, encoding="utf-8") as fh:
                merged = json.load(fh)
            with _LOCK:
                _MANIFEST_CACHE[key] = merged
            return merged
        except (OSError, ValueError):
            pass          # a corrupt cache is re-fetched, not fatal

    if not ENABLED:
        raise HeightError(unavailable_reason())

    # Manifests are named <s2token>_EPSG_<code>_<date>.json and several S2
    # tokens can cover one zone, so every manifest for the zone is merged.
    # The token is not derivable from a coordinate, so the bucket is listed.
    epsg = 32600 + zone
    listing_url = (f"https://storage.googleapis.com/storage/v1/b/"
                   f"open-buildings-temporal-data/o"
                   f"?prefix={DATASET_VERSION}/manifests/&maxResults=1000"
                   f"&fields=items/name,nextPageToken")
    names: List[str] = []
    token = None
    while True:
        url = listing_url + (f"&pageToken={token}" if token else "")
        page = json.loads(_fetch(url).decode("utf-8"))
        names += [i["name"] for i in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            break

    want = f"_EPSG_{epsg}_{YEAR}_"
    mine = sorted(n for n in names if want in n)
    if not mine:
        raise HeightError(
            f"Open Buildings has no {YEAR} tiles for UTM zone {zone} "
            f"(EPSG:{epsg}). The dataset covers Africa, South and South-East "
            f"Asia, Latin America and the Caribbean - not the whole world.")

    sources: List[dict] = []
    for name in mine:
        doc = json.loads(_fetch(f"{BUCKET}/{name}").decode("utf-8"))
        prefix = doc["uriPrefix"].replace(
            "gs://open-buildings-temporal-data/", "")
        for src in doc["tilesets"][0]["sources"]:
            sources.append({
                "path": prefix + src["uris"][0],
                "t": src["affineTransform"],
                "w": src["dimensions"]["width"],
                "h": src["dimensions"]["height"],
            })
    merged = {"zone": zone, "year": YEAR, "sources": sources}

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(local, "w", encoding="utf-8") as fh:
            json.dump(merged, fh)
    except OSError:
        pass              # an unwritable cache slows things down, nothing more

    with _LOCK:
        _MANIFEST_CACHE[key] = merged
    return merged


def _covering_tile(zone: int, easting: float, northing: float) -> dict:
    for src in _manifest(zone)["sources"]:
        t = src["t"]
        x0, y0 = t["translateX"], t["translateY"]
        x1 = x0 + src["w"] * t["scaleX"]
        y1 = y0 + src["h"] * t["scaleY"]          # scaleY is negative
        if x0 <= easting < x1 and y1 < northing <= y0:
            return src
    raise HeightError(
        f"No Open Buildings tile covers UTM {zone} E={easting:.0f} "
        f"N={northing:.0f}. The point may be offshore or outside coverage.")


# -- GeoTIFF -------------------------------------------------------------

_TAG_WIDTH, _TAG_LENGTH = 256, 257
_TAG_TILEWIDTH, _TAG_TILELENGTH = 322, 323
_TAG_TILEOFFSETS, _TAG_TILEBYTES = 324, 325
_TAG_COMPRESSION, _TAG_PREDICTOR, _TAG_PLANAR = 259, 317, 284


def _read_ifd(url: str) -> dict:
    head = _fetch(url, 0, 65535)
    if head[:2] != b"II":
        raise HeightError("Expected a little-endian TIFF.")
    if struct.unpack("<H", head[2:4])[0] != 42:
        raise HeightError("Expected a classic TIFF, not BigTIFF.")

    ifd = struct.unpack("<I", head[4:8])[0]
    count = struct.unpack("<H", head[ifd:ifd + 2])[0]
    tags: Dict[int, int] = {}
    for i in range(count):
        e = ifd + 2 + i * 12
        tag, _typ, _cnt = struct.unpack("<HHI", head[e:e + 8])
        tags[tag] = struct.unpack("<I", head[e + 8:e + 12])[0]

    # Only the combination this dataset actually uses is supported. Guessing
    # at LZW or JPEG would mean a decoder with no test data behind it.
    if tags.get(_TAG_COMPRESSION) != 8:
        raise HeightError(
            f"Tile compression {tags.get(_TAG_COMPRESSION)} is not Deflate; "
            f"this reader deliberately supports only the one scheme the "
            f"dataset uses.")
    if tags.get(_TAG_PLANAR) != 2:
        raise HeightError("Expected band-separated (PlanarConfig=2) tiles.")
    return tags


def _tile_values(url: str, tags: dict, band: int,
                 tile_row: int, tile_col: int) -> List[Tuple[float, ...]]:
    """One decoded 512x512 float tile, as rows of values."""
    tw, th = tags[_TAG_TILEWIDTH], tags[_TAG_TILELENGTH]
    across = (tags[_TAG_WIDTH] + tw - 1) // tw
    down = (tags[_TAG_LENGTH] + th - 1) // th
    per_plane = across * down
    index = band * per_plane + tile_row * across + tile_col

    cache_key = (url, index)
    with _LOCK:
        if cache_key in _TILE_CACHE:
            return _TILE_CACHE[cache_key]

    def table(offset: int, n: int) -> Tuple[int, ...]:
        raw = _fetch(url, offset, offset + n * 4 - 1)
        return struct.unpack("<" + "I" * n, raw)

    total = 3 * per_plane
    offsets = table(tags[_TAG_TILEOFFSETS], total)
    counts = table(tags[_TAG_TILEBYTES], total)

    raw = zlib.decompress(
        _fetch(url, offsets[index], offsets[index] + counts[index] - 1))
    expected = tw * th * 4
    if len(raw) != expected:
        raise HeightError(
            f"Tile decompressed to {len(raw)} bytes, expected {expected}.")

    rows = _undo_float_predictor(raw, tw, th) \
        if tags.get(_TAG_PREDICTOR) == 3 else \
        [struct.unpack("<" + "f" * tw, raw[r * tw * 4:(r + 1) * tw * 4])
         for r in range(th)]

    with _LOCK:
        if len(_TILE_CACHE) >= _MAX_CACHED_TILES:
            _TILE_CACHE.pop(next(iter(_TILE_CACHE)))
        _TILE_CACHE[cache_key] = rows
    return rows


def _undo_float_predictor(raw: bytes, tw: int, th: int) -> List[Tuple[float, ...]]:
    """
    Reverse TIFF predictor 3, the floating-point predictor.

    Two steps, in this order, and both are easy to get subtly wrong:

      1. Horizontal differencing is undone over BYTES with stride 1. Stride 1
         is because PlanarConfiguration is 2 - for interleaved bands it would
         be the sample count. Using the row width here instead (an obvious
         misreading, since the shuffle below is row-width based) decodes
         almost correctly: flat areas are unaffected because their deltas are
         zero, so it survives a careless eyeball check and produces absurd
         values only where the terrain actually varies.

      2. The bytes are then de-shuffled from per-significance planes back
         into floats. The encoder writes all the most-significant bytes of a
         row, then all the next, and so on, so for little-endian output the
         plane order reverses.
    """
    row_bytes = tw * 4
    out: List[Tuple[float, ...]] = []
    for r in range(th):
        row = bytearray(raw[r * row_bytes:(r + 1) * row_bytes])
        for i in range(1, row_bytes):
            row[i] = (row[i] + row[i - 1]) & 0xFF
        rebuilt = bytearray(row_bytes)
        for i in range(tw):
            rebuilt[i * 4 + 3] = row[i]
            rebuilt[i * 4 + 2] = row[tw + i]
            rebuilt[i * 4 + 1] = row[2 * tw + i]
            rebuilt[i * 4 + 0] = row[3 * tw + i]
        out.append(struct.unpack("<" + "f" * tw, bytes(rebuilt)))
    return out


# -- the public call -------------------------------------------------------

def sample_footprint(ring: Sequence[Tuple[float, float]],
                     max_samples: int = 4096) -> Optional[dict]:
    """
    Measured building heights over a lon/lat footprint, or None.

    Returns None - never raises - on every failure: disabled, no network, no
    coverage, too few valid pixels. The caller keeps its declared geometry, so
    this can only add evidence.

    THE RING MUST BE IN LON/LAT DEGREES. Most footprints in this system are
    in map pixels, because the sheet carried no control points, and sampling
    a raster with pixel coordinates would read a position in the Gulf of
    Guinea with complete confidence. The guard below is the one thing in this
    function that must not be removed.
    """
    if not ENABLED and not os.path.isdir(CACHE_DIR):
        return None
    if not ring or len(ring) < 3:
        return None

    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    if not (all(-180 <= x <= 180 for x in lons)
            and all(-90 <= y <= 90 for y in lats)):
        return None               # pixel coordinates, not degrees
    # A ring spanning degrees is not a parcel; it is a pixel ring that
    # happens to fall inside the valid range.
    if max(lons) - min(lons) > 0.5 or max(lats) - min(lats) > 0.5:
        return None

    try:
        return _sample(lons, lats, max_samples)
    except HeightError:
        return None
    except Exception:
        return None


def _sample(lons: List[float], lats: List[float], max_samples: int) -> Optional[dict]:
    zone, e0, n0 = utm_forward(min(lons), min(lats))
    _, e1, n1 = utm_forward(max(lons), max(lats))
    east_lo, east_hi = min(e0, e1), max(e0, e1)
    north_lo, north_hi = min(n0, n1), max(n0, n1)

    src = _covering_tile(zone, (east_lo + east_hi) / 2,
                         (north_lo + north_hi) / 2)
    url = f"{BUCKET}/{src['path']}"
    tags = _read_ifd(url)
    t = src["t"]
    sx, sy = t["scaleX"], t["scaleY"]
    tw, th = tags[_TAG_TILEWIDTH], tags[_TAG_TILELENGTH]

    def to_pixel(east: float, north: float) -> Tuple[int, int]:
        return (int((east - t["translateX"]) / sx),
                int((north - t["translateY"]) / sy))

    px_lo, py_hi = to_pixel(east_lo, north_lo)
    px_hi, py_lo = to_pixel(east_hi, north_hi)
    px_lo, px_hi = max(0, min(px_lo, px_hi)), min(src["w"] - 1, max(px_lo, px_hi))
    py_lo, py_hi = max(0, min(py_lo, py_hi)), min(src["h"] - 1, max(py_lo, py_hi))

    # Stride the bounding box so a large parcel costs the same as a small one
    # and never fetches more than a bounded number of internal tiles.
    span_x, span_y = px_hi - px_lo + 1, py_hi - py_lo + 1
    step = max(1, int(math.sqrt(span_x * span_y / max(1, max_samples))))

    values: List[float] = []
    nodata = 0
    for py in range(py_lo, py_hi + 1, step):
        for px in range(px_lo, px_hi + 1, step):
            rows = _tile_values(url, tags, HEIGHT_BAND, py // th, px // tw)
            v = rows[py % th][px % tw]
            if v == NODATA or not (MIN_PLAUSIBLE_M <= v <= MAX_PLAUSIBLE_M):
                nodata += 1
                continue
            values.append(v)

    if len(values) < MIN_SAMPLES:
        return None

    values.sort()
    built = [v for v in values if v > 0.5]

    def pct(q: float) -> float:
        return round(values[min(len(values) - 1, int(len(values) * q))], 2)

    return {
        "source": "open_buildings_2_5d",
        "provenance": "remote_sensed",
        "year": YEAR,
        "max_m": round(values[-1], 2),
        "p95_m": pct(0.95),
        "median_m": pct(0.5),
        "mean_m": round(sum(values) / len(values), 2),
        "built_fraction": round(len(built) / len(values), 3),
        "samples": len(values),
        "dropped": nodata,
        "resolution_m": 4.0,
        "attribution": ATTRIBUTION,
        "caveat": DATASET_NOTE,
    }


def representative_height(reading: Optional[dict]) -> Optional[float]:
    """
    The one number to compare a declared building against.

    The 95th percentile, not the maximum: at 4 m resolution a footprint's
    tallest pixel may be a neighbour's roof bleeding across the boundary, and
    the point of this figure is to contradict a wildly wrong floor count, not
    to win an argument about a metre.

    None when the footprint is mostly unbuilt, because the tallest pixel of a
    field is not a building height.
    """
    if not reading:
        return None
    if reading.get("built_fraction", 0) < 0.1:
        return None
    return reading.get("p95_m")
