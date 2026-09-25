"""
Import parcel polygons digitized in ArcGIS or QGIS (ESRI Shapefile).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

cadastral.vectorize() recovers parcel boundaries from a scanned map by
contour-finding. That works, and it is guessing: a boundary drawn with a
2-pixel pen on a mudded, folded village sheet has a real ambiguity of metres
on the ground, and the vectoriser resolves it by whatever the threshold did.

A parcel layer digitized by a person - by hand, or semi-automatically with
ArcScan / the QGIS raster tracer - has no such ambiguity. The operator
decided where the boundary is. Importing that layer replaces the weakest link
in the geometry chain, so this is the highest-accuracy source of parcels the
system can accept, and it needs no georeferencing step at all because
shapefile coordinates are already world coordinates.

WHAT IS READ
------------
  .shp  polygon geometry (types 5 / 15 / 25 - Polygon, PolygonZ, PolygonM)
  .dbf  attributes, to recover each parcel's khasra / survey number
  .prj  the coordinate system, so a projected layer is refused not misplaced
  .cpg  the .dbf's codepage, when present

THE CONVENTION THAT SILENTLY BREAKS THIS
----------------------------------------
Shapefile and GeoJSON wind their rings in OPPOSITE directions. A shapefile
outer ring is CLOCKWISE and its holes counter-clockwise (ESRI Shapefile
Technical Description, page 8); RFC 7946 GeoJSON requires the exterior
counter-clockwise, by the right-hand rule. Copying coordinates across without
reversing produces a polygon that draws correctly in Leaflet - which does not
care - and is rejected or inverted by anything that does, PostGIS and
strict GeoJSON validators included. Rings are therefore re-wound on import,
and there is a test for it.

Coordinates are checked the same way georeference.py checks a raster's: a
layer in UTM metres is refused with the CRS named, never reinterpreted as
degrees. See that module for why no datum shift is attempted here.
"""

from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import cadastral

# Shape types this module accepts. The Z and M variants carry extra arrays
# AFTER the XY coordinates, so the XY parsing is identical for all three and
# the trailing data is simply not read.
POLYGON_TYPES = {5: "Polygon", 15: "PolygonZ", 25: "PolygonM"}

# Named so a refusal can say what the file actually holds.
OTHER_SHAPE_TYPES = {
    0: "Null", 1: "Point", 3: "PolyLine", 8: "MultiPoint",
    11: "PointZ", 13: "PolyLineZ", 18: "MultiPointZ",
    21: "PointM", 23: "PolyLineM", 28: "MultiPointM",
    31: "MultiPatch",
}

# .dbf column names that plausibly hold the plot identifier, in the order they
# are preferred. Matched case-insensitively with separators stripped, because
# every state's export names this differently.
KHASRA_FIELD_CANDIDATES = (
    "khasranumber", "khasrano", "khasra", "khasrasankhya",
    "surveynumber", "surveyno", "survey", "sno",
    "plotnumber", "plotno", "plot",
    "gatnumber", "gatno",              # Maharashtra
    "kitnumber", "kitno",              # Haryana / Punjab
    "fmbno",                           # Tamil Nadu field measurement book
    "parcelid", "parcelno",
)

_SHP_FILE_CODE = 9994


class ShapefileError(Exception):
    """Raised when a shapefile exists but cannot be imported."""


# --------------------------------------------------------------------------
# .prj  (coordinate system)
# --------------------------------------------------------------------------

def read_prj(shp_path: str) -> Tuple[Optional[str], Optional[int], bool]:
    """
    (crs_name, epsg, is_projected) from the .prj beside `shp_path`.

    A .prj is WKT, and the distinction that matters is the outermost keyword:
    PROJCS means the coordinates are a projected grid in linear units,
    GEOGCS means they are angular. That single word is enough to decide
    whether this layer can be used, without a full WKT parser.
    """
    prj = os.path.splitext(shp_path)[0] + ".prj"
    if not os.path.exists(prj):
        return (None, None, False)
    try:
        with open(prj, "r", encoding="utf-8-sig", errors="replace") as fh:
            wkt = fh.read().strip()
    except Exception:
        return (None, None, False)
    if not wkt:
        return (None, None, False)

    is_projected = wkt.upper().lstrip().startswith("PROJCS")
    name = None
    start = wkt.find('"')
    if start != -1:
        end = wkt.find('"', start + 1)
        if end > start:
            name = wkt[start + 1:end]

    epsg = None
    marker = wkt.upper().rfind('AUTHORITY["EPSG"')
    if marker != -1:
        tail = wkt[marker:]
        digits = ""
        for ch in tail[tail.find(",") + 1:]:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        if digits:
            epsg = int(digits)
    return (name, epsg, is_projected)


# --------------------------------------------------------------------------
# .dbf  (attributes)
# --------------------------------------------------------------------------

def _dbf_encoding(shp_path: str) -> str:
    """
    The .dbf's text encoding: the .cpg sidecar if present, else UTF-8.

    Owner and village names in these layers are frequently Devanagari, so
    guessing latin-1 by default (as the dBASE III era would) mangles them
    into unusable mojibake. UTF-8 first, with a lossy fallback only so a bad
    codepage cannot crash an otherwise good import.
    """
    cpg = os.path.splitext(shp_path)[0] + ".cpg"
    if os.path.exists(cpg):
        try:
            with open(cpg, "r", encoding="ascii", errors="replace") as fh:
                declared = fh.read().strip()
            if declared:
                if declared.isdigit():
                    return "cp" + declared
                return declared
        except Exception:
            pass
    return "utf-8"


DELETED_KEY = "_deleted"


def read_dbf(path: str, encoding: Optional[str] = None) -> List[Dict[str, str]]:
    """
    The attribute table as a list of {field_name: value} dicts, one per row
    IN FILE ORDER, with deleted rows marked rather than removed.

    dBASE marks a deletion with '*' in the row's first byte and leaves the
    data in place. Dropping those rows here looked tidy and was a real bug:
    attributes are matched to geometry BY POSITION, so removing row 3 shifts
    every row after it up by one and quietly relabels parcel 4 with parcel
    5's khasra number. Attaching a neighbour's plot number to a parcel is
    exactly the kind of confident, invisible error this system exists to
    prevent, so the row keeps its place and carries a DELETED_KEY flag for
    the caller to act on.
    """
    encoding = encoding or "utf-8"
    with open(path, "rb") as fh:
        header = fh.read(32)
        if len(header) < 32:
            raise ShapefileError("The .dbf file is truncated.")
        record_count, header_length, record_length = struct.unpack(
            "<IHH", header[4:12])

        fields: List[Tuple[str, int]] = []
        consumed = 32
        while consumed < header_length - 1:
            descriptor = fh.read(32)
            if len(descriptor) < 32 or descriptor[0] in (0x0D, 0x00):
                break
            consumed += 32
            name = descriptor[:11].split(b"\x00")[0].decode("ascii", "replace").strip()
            length = descriptor[16]
            fields.append((name, length))

        fh.seek(header_length)
        rows: List[Dict[str, str]] = []
        for _ in range(record_count):
            raw = fh.read(record_length)
            if len(raw) < record_length:
                break
            offset = 1
            row: Dict[str, str] = {}
            for name, length in fields:
                chunk = raw[offset:offset + length]
                offset += length
                row[name] = chunk.decode(encoding, "replace").strip()
            if raw[:1] == b"*":
                row[DELETED_KEY] = "1"
            rows.append(row)
        return rows


def pick_khasra_field(field_names: Sequence[str]) -> Optional[str]:
    """The attribute column most likely to hold the plot identifier."""
    normalised = {}
    for name in field_names:
        if name == DELETED_KEY:
            continue
        key = "".join(ch for ch in name.casefold() if ch.isalnum())
        normalised.setdefault(key, name)
    for candidate in KHASRA_FIELD_CANDIDATES:
        if candidate in normalised:
            return normalised[candidate]
    return None


# --------------------------------------------------------------------------
# .shp  (geometry)
# --------------------------------------------------------------------------

def signed_area(ring: Sequence[Tuple[float, float]]) -> float:
    """
    Shoelace signed area. Positive is counter-clockwise, negative clockwise.

    Used to decide winding, not to measure ground area - that is
    cadastral.polygon_area_m2, which converts degrees to metres properly.
    """
    total = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def read_shp(path: str) -> Tuple[List[List[List[Tuple[float, float]]]], Tuple[float, ...], int]:
    """
    (records, bounding_box, shape_type) from a .shp file.

    Each record is a list of rings; each ring a list of (x, y). Rings are
    returned exactly as stored, winding included - re-winding happens in
    load_parcels so this stays a faithful reader.
    """
    with open(path, "rb") as fh:
        header = fh.read(100)
        if len(header) < 100:
            raise ShapefileError("The .shp file is too short to hold a header.")
        # The file code and length are BIG-endian; everything after is little.
        code = struct.unpack(">i", header[0:4])[0]
        if code != _SHP_FILE_CODE:
            raise ShapefileError(
                f"Not a shapefile (file code {code}, expected {_SHP_FILE_CODE}).")
        shape_type = struct.unpack("<i", header[32:36])[0]
        bbox = struct.unpack("<4d", header[36:68])

        if shape_type not in POLYGON_TYPES:
            found = OTHER_SHAPE_TYPES.get(shape_type, f"type {shape_type}")
            raise ShapefileError(
                f"This shapefile holds {found} geometry. Parcel boundaries have "
                f"to be polygons - if the layer was digitized as lines, close "
                f"them into polygons in ArcGIS or QGIS first (Feature To "
                f"Polygon / Lines to Polygons) and export again.")

        records: List[List[List[Tuple[float, float]]]] = []
        while True:
            record_header = fh.read(8)
            if len(record_header) < 8:
                break
            _number, content_words = struct.unpack(">ii", record_header)
            content = fh.read(content_words * 2)
            if len(content) < 4:
                break
            record_type = struct.unpack("<i", content[0:4])[0]
            if record_type == 0:                      # a Null shape is legal
                records.append([])
                continue
            if record_type not in POLYGON_TYPES:
                continue
            if len(content) < 44:
                continue
            num_parts, num_points = struct.unpack("<ii", content[36:44])
            if num_parts <= 0 or num_points <= 0:
                records.append([])
                continue
            parts_end = 44 + num_parts * 4
            points_end = parts_end + num_points * 16
            if len(content) < points_end:
                raise ShapefileError(
                    "A polygon record is truncated - the .shp file is corrupt "
                    "or was only partially written.")
            parts = list(struct.unpack("<%di" % num_parts,
                                       content[44:parts_end]))
            flat = struct.unpack("<%dd" % (num_points * 2),
                                 content[parts_end:points_end])
            points = [(flat[i * 2], flat[i * 2 + 1]) for i in range(num_points)]

            rings = []
            bounds = parts + [num_points]
            for i in range(num_parts):
                ring = points[bounds[i]:bounds[i + 1]]
                if len(ring) >= 4:                    # a closed ring needs 4
                    rings.append(ring)
            records.append(rings)
        return (records, bbox, shape_type)


def check_lonlat_bbox(bbox: Tuple[float, ...], crs_name: Optional[str],
                      epsg: Optional[int], is_projected: bool) -> None:
    """Refuse a layer whose coordinates cannot be longitude/latitude."""
    if is_projected:
        raise ShapefileError(
            "This layer is in a projected coordinate system"
            + (f" ({crs_name})" if crs_name else "")
            + (f", EPSG:{epsg}" if epsg else "")
            + ", so its coordinates are metres rather than degrees. This "
              "system stores WGS84 longitude/latitude, as RFC 7946 GeoJSON "
              "requires. Reproject the layer to EPSG:4326 (ArcGIS: Project; "
              "QGIS: Export -> Save Features As, CRS EPSG:4326) and import "
              "again.")
    xmin, ymin, xmax, ymax = bbox[:4]
    if not (-180.0 <= xmin <= 180.0 and -180.0 <= xmax <= 180.0
            and -90.0 <= ymin <= 90.0 and -90.0 <= ymax <= 90.0):
        raise ShapefileError(
            f"The layer's extent ({xmin:.1f}, {ymin:.1f}) to "
            f"({xmax:.1f}, {ymax:.1f}) is not longitude/latitude"
            + (f"; the .prj reports {crs_name}" if crs_name else "")
            + ". Reproject it to EPSG:4326 and import again - this system "
              "does not guess a datum shift, because getting it wrong moves "
              "parcels by hundreds of metres.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def load_parcels(shp_path: str) -> Tuple[List[cadastral.Parcel], List[str]]:
    """
    (parcels, warnings) from a digitized parcel layer.

    The returned parcels already carry geo_polygon in lon/lat, so they need
    no georeferencing: shapefile coordinates ARE world coordinates. The pixel
    fields carry no meaning here and are left empty rather than filled with
    invented numbers - area_px in particular would be a lie, since no raster
    was involved and cadastral.polygon_area_m2 computes the real area from
    the geometry anyway.
    """
    if not os.path.exists(shp_path):
        raise ShapefileError(f"No such shapefile: {shp_path}")

    crs_name, epsg, is_projected = read_prj(shp_path)
    records, bbox, _shape_type = read_shp(shp_path)
    check_lonlat_bbox(bbox, crs_name, epsg, is_projected)

    warnings: List[str] = []
    if crs_name is None:
        warnings.append(
            "This layer has no .prj file, so its coordinate system is "
            "undeclared. It was accepted because its extent is a plausible "
            "longitude/latitude, but that is an inference, not a guarantee.")

    attributes: List[Dict[str, str]] = []
    dbf = os.path.splitext(shp_path)[0] + ".dbf"
    if os.path.exists(dbf):
        try:
            attributes = read_dbf(dbf, _dbf_encoding(shp_path))
        except Exception as exc:
            warnings.append(f"Attributes could not be read ({exc}); parcels "
                            f"were imported without khasra numbers.")
    else:
        warnings.append("No .dbf alongside the .shp, so no khasra numbers "
                        "could be attached to these parcels.")

    khasra_field = pick_khasra_field(list(attributes[0].keys())) if attributes else None
    if attributes and khasra_field is None:
        warnings.append(
            "No attribute column looked like a plot identifier (looked for "
            + ", ".join(KHASRA_FIELD_CANDIDATES[:6]) + ", ...). Parcels were "
            "imported without khasra numbers, so records cannot be matched "
            "to them.")

    parcels: List[cadastral.Parcel] = []
    holes_dropped = 0
    multipart = 0
    deleted_rows = 0

    for index, rings in enumerate(records):
        if not rings:
            continue
        # A shapefile polygon record may hold several rings: additional outer
        # rings (a parcel in disjoint pieces) and inner rings (holes),
        # distinguished only by winding. Parcel carries ONE boundary, so the
        # largest outer ring is used and anything else is reported rather
        # than silently discarded.
        outer = [r for r in rings if signed_area(r) < 0]
        inner = [r for r in rings if signed_area(r) >= 0]
        if not outer:
            # Every ring wound counter-clockwise: the file does not follow the
            # ESRI convention. Fall back to the largest ring by magnitude.
            outer = [max(rings, key=lambda r: abs(signed_area(r)))]
            inner = [r for r in rings if r is not outer[0]]
        if len(outer) > 1:
            multipart += 1
        holes_dropped += len(inner)
        ring = max(outer, key=lambda r: abs(signed_area(r)))

        # Shapefile rings are closed (last point repeats the first);
        # parcels_to_geojson closes them itself, so the duplicate is dropped
        # here or the emitted ring carries a doubled vertex.
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        # Shapefile outer rings are clockwise, RFC 7946 exteriors are
        # counter-clockwise. See the module docstring.
        if signed_area(ring) < 0:
            ring = list(reversed(ring))
        if len(ring) < 3:
            continue

        khasra = None
        if khasra_field and index < len(attributes):
            row = attributes[index]
            if not row.get(DELETED_KEY):
                value = (row.get(khasra_field) or "").strip()
                khasra = value or None
            else:
                deleted_rows += 1

        parcels.append(cadastral.Parcel(
            parcel_id=index + 1,
            pixel_polygon=[],
            pixel_centroid=(0.0, 0.0),
            area_px=0.0,
            khasra_number=khasra,
            geo_polygon=[(float(x), float(y)) for x, y in ring],
        ))

    if deleted_rows:
        warnings.append(
            f"{deleted_rows} attribute row(s) are marked deleted in the .dbf; "
            f"those parcels were imported without a khasra number rather than "
            f"being given the next row's. Pack the layer in ArcGIS or QGIS "
            f"(export it afresh) to remove the deleted rows.")
    if multipart:
        warnings.append(
            f"{multipart} parcel(s) are made of several separate pieces; only "
            f"the largest piece of each was imported, because a record here "
            f"carries one boundary.")
    if holes_dropped:
        warnings.append(
            f"{holes_dropped} interior ring(s) (holes) were dropped - a "
            f"parcel is stored as a single boundary, so an enclosed exclusion "
            f"is not represented and the area of those parcels is "
            f"overstated by the size of the hole.")
    if not parcels:
        raise ShapefileError(
            "The shapefile contained no usable polygons. Check that the "
            "layer has features and that they are polygons, not lines.")
    return (parcels, warnings)


def find_shapefile(folder: str) -> Optional[str]:
    """The .shp in a cadastral map folder, if there is exactly one to use."""
    if not os.path.isdir(folder):
        return None
    found = sorted(f for f in os.listdir(folder) if f.lower().endswith(".shp"))
    return os.path.join(folder, found[0]) if found else None


def to_geojson(shp_path: str, disclaimer: Optional[str] = None) -> dict:
    """A digitized parcel layer as the same GeoJSON the raster path emits."""
    parcels, warnings = load_parcels(shp_path)
    geojson = cadastral.parcels_to_geojson(parcels, disclaimer=disclaimer)
    crs_name, epsg, _ = read_prj(shp_path)
    geojson["_georeferencing"] = {
        "method": "shapefile",
        "source_file": os.path.basename(shp_path),
        "crs": crs_name,
        "epsg": epsg,
        # No transform, and that is the point: the operator digitized in world
        # coordinates, so nothing was fitted and nothing can have gone wrong
        # in the fitting.
        "transform": None,
        "parcels": len(parcels),
    }
    if warnings:
        geojson["_warnings"] = warnings
    return geojson
