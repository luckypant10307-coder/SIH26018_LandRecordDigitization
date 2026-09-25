"""
Import georeferencing produced by ArcGIS or QGIS.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

cadastral.fit_affine_transform builds a 6-parameter pixel -> (lon, lat)
affine from hand-picked control points. That is the same mathematical object
a first-order (affine) georeferencing in ArcGIS Georeferencing or the QGIS
Georeferencer produces - which means a surveyor who has already done the work
properly, against a real basemap with real ground control, should not have to
redo it by hand here. This module reads their output.

Three carriers are supported, in the order they are looked for:

  1. WORLD FILE - the six-line sidecar ArcGIS writes next to the raster
     (.tfw for TIFF, .jgw for JPEG, .pgw for PNG, or a generic .wld).
  2. GeoTIFF TAGS - the transform embedded in the TIFF itself, which is what
     "Export -> georeferenced TIFF" gives you by default.
  3. CONTROL POINTS - the existing control_points.json path, unchanged.

TWO CONVENTIONS THAT SILENTLY CORRUPT A MAP IF GOT WRONG
--------------------------------------------------------
* World file LINE ORDER is A, D, B, E, C, F - not A, B, C, D, E, F. Lines 2
  and 3 are the two skew terms and they are the other way round from the
  order the letters suggest. On a north-up map both are 0.0 and a swap is
  invisible; on a rotated map - which is most scanned cadastral sheets - it
  mirrors the rotation and puts every parcel in the wrong place.
* PIXEL ANCHOR. A world file's C/F give the CENTRE of the upper-left pixel.
  A GeoTIFF tiepoint gives its CORNER. That is half a pixel of disagreement,
  which on a 1:4000 mudded village sheet is a real distance on the ground, so
  the GeoTIFF path shifts by half a pixel to match the world-file convention
  and the pixel coordinates the vectoriser produces.

WHY A PROJECTED CRS IS REFUSED RATHER THAN GUESSED
--------------------------------------------------
The rest of this system works in WGS84 longitude/latitude, because that is
what RFC 7946 GeoJSON requires and what the Leaflet map consumes. ArcGIS
georeferencing is very often done in a projected CRS instead - a UTM zone, or
one of the Everest-datum Lambert grids the old revenue sheets use - and those
coordinates are metres, not degrees. Read as degrees, a UTM easting of 730000
is not a small error; it is off the Earth.

This module therefore checks the numbers and REFUSES anything that cannot be
longitude/latitude, naming the CRS it found so the message is actionable. It
does not attempt the reprojection: doing that correctly needs a datum shift
(Everest 1830 to WGS84 is hundreds of metres in places), which needs pyproj
and a proper grid, not arithmetic invented here. Re-exporting from ArcGIS in
EPSG:4326 takes a surveyor ten seconds and is exact.
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# World-file extension per raster extension, as ArcGIS names them.
WORLD_FILE_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    ".tif": (".tfw", ".tifw", ".wld"),
    ".tiff": (".tfw", ".tiffw", ".wld"),
    ".jpg": (".jgw", ".jpgw", ".wld"),
    ".jpeg": (".jgw", ".jpegw", ".wld"),
    ".png": (".pgw", ".pngw", ".wld"),
    ".bmp": (".bpw", ".bmpw", ".wld"),
    ".gif": (".gfw", ".gifw", ".wld"),
}

# GeoTIFF tags this module understands.
_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_TRANSFORM = 34264
_TAG_GEOKEYS = 34735

# GeoKey ids for the CRS, used only to name it in an error message.
_GEOKEY_PROJECTED_CS = 3072
_GEOKEY_GEOGRAPHIC_CS = 2048

# EPSG codes that ARE WGS84 longitude/latitude.
_LONLAT_EPSG = {4326, 4979, 4327}


# A skew term below this fraction of the pixel scale is treated as zero.
# Least-squares fitting and text round-tripping both leave crumbs - a
# perfectly north-up demo map came back with an x-skew of -2.5e-13, which is
# non-zero, truthy, and utterly meaningless. Reporting that map as "rotated"
# is a lie told by floating point.
_ROTATION_EPSILON = 1e-6


def is_rotated(a: float, b: float, d: float, e: float) -> bool:
    """
    Whether the transform carries a real rotation or skew.

    Judged relative to the pixel scale rather than against zero: what matters
    is whether the off-diagonal terms are big enough to tilt the image, and
    "big enough" only means anything compared with how far a pixel spans.
    """
    scale = max(abs(a), abs(e))
    if scale == 0.0:
        return bool(b or d)
    return max(abs(b), abs(d)) > _ROTATION_EPSILON * scale


class GeoreferenceError(Exception):
    """Raised when a georeferencing exists but cannot be used as-is."""


@dataclass
class ImportedGeoreference:
    """
    A pixel -> (lon, lat) affine read from a GIS product, with provenance.

    The six coefficients match cadastral.AffineTransform exactly:
        lon = a*px + b*py + c
        lat = d*px + e*py + f
    """
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    source: str                      # "world_file" | "geotiff_tags"
    source_path: str
    crs: Optional[str] = None        # what we could determine, for the record
    epsg: Optional[int] = None
    rotated: bool = False            # non-zero skew terms

    def as_transform_kwargs(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c,
                "d": self.d, "e": self.e, "f": self.f}

    def apply(self, px: float, py: float) -> Tuple[float, float]:
        return (self.a * px + self.b * py + self.c,
                self.d * px + self.e * py + self.f)

    def to_dict(self) -> dict:
        return {
            "transform": self.as_transform_kwargs(),
            "source": self.source,
            "source_file": os.path.basename(self.source_path),
            "crs": self.crs,
            "epsg": self.epsg,
            "rotated": self.rotated,
        }


# --------------------------------------------------------------------------
# World file
# --------------------------------------------------------------------------

def find_world_file(image_path: str) -> Optional[str]:
    """The world file sitting beside `image_path`, if there is one."""
    stem, ext = os.path.splitext(image_path)
    for candidate_ext in WORLD_FILE_EXTENSIONS.get(ext.lower(), (".wld",)):
        candidate = stem + candidate_ext
        if os.path.exists(candidate):
            return candidate
    # ArcGIS also accepts the doubled form, e.g. map.png.aux -> map.png.wld
    doubled = image_path + ".wld"
    return doubled if os.path.exists(doubled) else None


def read_world_file(path: str) -> Tuple[float, float, float, float, float, float]:
    """
    Parse a six-line world file into (a, b, c, d, e, f).

    File order is A, D, B, E, C, F. See the module docstring - this is the
    single easiest thing to get wrong about world files, and it is silent on
    a north-up map because both skew terms are then zero.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        numbers: List[float] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                numbers.append(float(line.split()[0]))
            except (ValueError, IndexError):
                raise GeoreferenceError(
                    f"{os.path.basename(path)} line {len(numbers) + 1} is not a "
                    f"number: {line!r}. A world file is six plain numbers.")
    if len(numbers) < 6:
        raise GeoreferenceError(
            f"{os.path.basename(path)} has {len(numbers)} numbers; a world "
            f"file needs six (x-scale, y-skew, x-skew, y-scale, x-origin, "
            f"y-origin).")
    a, d, b, e, c, f = numbers[:6]
    if a == 0.0 and b == 0.0:
        raise GeoreferenceError(
            f"{os.path.basename(path)} has a zero x-scale, so every pixel "
            f"would map to the same longitude. The file is not a usable "
            f"georeferencing.")
    return (a, b, c, d, e, f)


# --------------------------------------------------------------------------
# GeoTIFF tags
# --------------------------------------------------------------------------

def _read_tiff_tags(path: str) -> Dict[int, List[float]]:
    """
    The georeferencing tags of a TIFF, as {tag: [values]}.

    A deliberately small reader: it walks the first IFD and pulls only the
    handful of tags this module needs, rather than pulling in a TIFF library
    for six numbers. Anything it does not recognise it ignores; anything it
    cannot parse safely it declines by raising.
    """
    with open(path, "rb") as fh:
        header = fh.read(8)
        if len(header) < 8:
            raise GeoreferenceError("File is too short to be a TIFF.")
        if header[:2] == b"II":
            endian = "<"
        elif header[:2] == b"MM":
            endian = ">"
        else:
            raise GeoreferenceError("Not a TIFF file (no II/MM byte-order mark).")
        magic = struct.unpack(endian + "H", header[2:4])[0]
        if magic == 43:
            raise GeoreferenceError(
                "This is a BigTIFF. Re-export as a standard TIFF, or supply a "
                "world file (.tfw) alongside it.")
        if magic != 42:
            raise GeoreferenceError(f"Not a TIFF file (magic {magic}, expected 42).")

        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        fh.seek(ifd_offset)
        count_bytes = fh.read(2)
        if len(count_bytes) < 2:
            raise GeoreferenceError("TIFF directory is truncated.")
        entry_count = struct.unpack(endian + "H", count_bytes)[0]

        wanted = {_TAG_PIXEL_SCALE, _TAG_TIEPOINT, _TAG_TRANSFORM, _TAG_GEOKEYS}
        sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
                 11: 4, 12: 8}
        fmts = {3: "H", 4: "I", 11: "f", 12: "d"}
        out: Dict[int, List[float]] = {}

        entries = fh.read(entry_count * 12)
        for i in range(entry_count):
            raw = entries[i * 12:(i + 1) * 12]
            if len(raw) < 12:
                break
            tag, dtype, n = struct.unpack(endian + "HHI", raw[:8])
            if tag not in wanted or dtype not in fmts:
                continue
            total = sizes.get(dtype, 0) * n
            if total <= 4:
                payload = raw[8:8 + total]
            else:
                offset = struct.unpack(endian + "I", raw[8:12])[0]
                here = fh.tell()
                fh.seek(offset)
                payload = fh.read(total)
                fh.seek(here)
            if len(payload) < total:
                continue
            out[tag] = list(struct.unpack(endian + fmts[dtype] * n, payload))
        return out


def _epsg_from_geokeys(geokeys: Optional[List[float]]) -> Optional[int]:
    """
    The CRS's EPSG code from a GeoKeyDirectory, if it is stored inline.

    The directory is a flat list of 4-shorts records after a 4-short header:
    (key_id, tiff_tag_location, count, value_or_offset). A key with
    tiff_tag_location == 0 holds its value directly, which is how an EPSG
    code is normally written - that is the only case read here, because the
    code is wanted for a diagnostic message, not for computation.
    """
    if not geokeys or len(geokeys) < 8:
        return None
    values = [int(v) for v in geokeys]
    number_of_keys = values[3]
    for i in range(number_of_keys):
        base = 4 + i * 4
        if base + 3 >= len(values):
            break
        key_id, location, _count, value = values[base:base + 4]
        if location != 0:
            continue
        if key_id in (_GEOKEY_PROJECTED_CS, _GEOKEY_GEOGRAPHIC_CS):
            if 1024 <= value <= 32767:
                return value
    return None


def read_geotiff_transform(path: str) -> Tuple[Tuple[float, ...], Optional[int]]:
    """
    ((a, b, c, d, e, f), epsg) from a GeoTIFF's own tags.

    Handles both carriers GDAL/ArcGIS use: an explicit 4x4
    ModelTransformation, or the far more common ModelPixelScale plus one
    ModelTiepoint. Both are CORNER-anchored, and both are shifted here by
    half a pixel onto the centre-anchored convention the rest of the
    pipeline uses (see the module docstring).
    """
    tags = _read_tiff_tags(path)
    epsg = _epsg_from_geokeys(tags.get(_TAG_GEOKEYS))

    matrix = tags.get(_TAG_TRANSFORM)
    if matrix and len(matrix) >= 16:
        a, b, _m2, c = matrix[0], matrix[1], matrix[2], matrix[3]
        d, e, _m6, f = matrix[4], matrix[5], matrix[6], matrix[7]
    else:
        scale = tags.get(_TAG_PIXEL_SCALE)
        tie = tags.get(_TAG_TIEPOINT)
        if not scale or len(scale) < 2 or not tie or len(tie) < 6:
            raise GeoreferenceError(
                "The TIFF carries no georeferencing tags (no "
                "ModelPixelScale/ModelTiepoint and no ModelTransformation). "
                "Georeference it in ArcGIS or QGIS first, or supply a .tfw "
                "world file beside it.")
        pixel_i, pixel_j = tie[0], tie[1]
        world_x, world_y = tie[3], tie[4]
        a, b = scale[0], 0.0
        d, e = 0.0, -abs(scale[1])
        c = world_x - pixel_i * a
        f = world_y - pixel_j * e

    # Corner -> centre of the upper-left pixel.
    c = c + 0.5 * a + 0.5 * b
    f = f + 0.5 * d + 0.5 * e
    if a == 0.0 and b == 0.0:
        raise GeoreferenceError("The TIFF's georeferencing has a zero x-scale.")
    return ((a, b, c, d, e, f), epsg)


# --------------------------------------------------------------------------
# CRS sanity
# --------------------------------------------------------------------------

def read_aux_crs(image_path: str) -> Optional[str]:
    """
    The CRS name from an ArcGIS `.aux.xml` sidecar, for diagnostics only.

    Best-effort by design: this exists to make "your map is not in lon/lat"
    into "your map is in WGS_1984_UTM_Zone_44N", which is the difference
    between a message a surveyor can act on and one they cannot.
    """
    for candidate in (image_path + ".aux.xml",
                      os.path.splitext(image_path)[0] + ".aux.xml"):
        if not os.path.exists(candidate):
            continue
        try:
            root = ET.parse(candidate).getroot()
        except Exception:
            continue
        for tag in (".//SRS", ".//SpatialReference", ".//WKT"):
            node = root.find(tag)
            if node is not None and (node.text or "").strip():
                text = node.text.strip()
                name = _crs_name_from_wkt(text)
                return name or text[:120]
    return None


def _crs_name_from_wkt(wkt: str) -> Optional[str]:
    """The quoted name of the outermost WKT object, e.g. PROJCS["...", ..."""
    start = wkt.find('"')
    if start == -1:
        return None
    end = wkt.find('"', start + 1)
    return wkt[start + 1:end] if end > start else None


def check_lonlat(coefficients: Tuple[float, ...], width: int, height: int,
                 epsg: Optional[int], crs_name: Optional[str]) -> None:
    """
    Refuse a transform whose outputs cannot be longitude/latitude.

    Checked at the raster's four corners rather than at its origin, because a
    projected grid whose false easting happens to be small can look
    plausible at one point and be absurd across the sheet.
    """
    a, b, c, d, e, f = coefficients
    if epsg is not None and epsg not in _LONLAT_EPSG:
        raise GeoreferenceError(
            f"This map is georeferenced in EPSG:{epsg}"
            + (f" ({crs_name})" if crs_name else "")
            + ", which is a projected coordinate system measured in metres. "
              "This system stores WGS84 longitude/latitude, as RFC 7946 "
              "GeoJSON requires. Re-export the georeferenced raster in "
              "EPSG:4326 (in ArcGIS: Data -> Export Raster, set the Spatial "
              "Reference to WGS 1984) and load it again.")

    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    for px, py in corners:
        lon = a * px + b * py + c
        lat = d * px + e * py + f
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise GeoreferenceError(
                f"The georeferencing puts a corner of this map at "
                f"({lon:.1f}, {lat:.1f}), which is not a longitude/latitude. "
                + (f"The file reports {crs_name}. " if crs_name else "")
                + "It is almost certainly in a projected CRS measured in "
                  "metres (a UTM zone, or an Everest-datum revenue grid). "
                  "Re-export it in EPSG:4326 and load it again - this system "
                  "does not guess a datum shift, because getting it wrong "
                  "moves parcels by hundreds of metres.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def discover(image_path: str, width: int, height: int) -> Optional[ImportedGeoreference]:
    """
    Read whatever georeferencing accompanies `image_path`.

    Returns None when there is none to find, so the caller can fall back to
    control_points.json. Raises GeoreferenceError when a georeferencing IS
    present but unusable - which must not be silently downgraded to "none",
    because a surveyor who georeferenced this map deserves to be told why
    their work was ignored rather than to watch it disappear.
    """
    crs_name = read_aux_crs(image_path)

    world = find_world_file(image_path)
    if world:
        coefficients = read_world_file(world)
        check_lonlat(coefficients, width, height, None, crs_name)
        a, b, c, d, e, f = coefficients
        return ImportedGeoreference(
            a=a, b=b, c=c, d=d, e=e, f=f,
            source="world_file", source_path=world, crs=crs_name,
            rotated=is_rotated(a, b, d, e))

    if os.path.splitext(image_path)[1].lower() in (".tif", ".tiff"):
        coefficients, epsg = read_geotiff_transform(image_path)
        check_lonlat(coefficients, width, height, epsg, crs_name)
        a, b, c, d, e, f = coefficients
        return ImportedGeoreference(
            a=a, b=b, c=c, d=d, e=e, f=f,
            source="geotiff_tags", source_path=image_path,
            crs=crs_name, epsg=epsg, rotated=is_rotated(a, b, d, e))

    return None
