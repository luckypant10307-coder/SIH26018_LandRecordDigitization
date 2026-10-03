#!/usr/bin/env python3
"""
Rebuild the Vansar parcel base layer from hand-digitised pixel coordinates.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

WHY THIS EXISTS

The 3D work needed a parcel with REAL GROUND under it: every volume until now
stood on a footprint read from a scanned map in pixels, which locates nothing
on the earth. This produces eight parcels in actual coordinates, so a stack
can be declared somewhere real.

WHY BY HAND

Automatic extraction was tried first and measured. `cadastral.vectorize()`
recovers 14 parcels from the synthetic village map and 15 from the real
Bhu-Naksha sheet. Run against this drone orthomosaic it returned 4 polygons -
every one of them the image's own data boundary, and not a single field bund.

That is the expected result rather than a defect: the module traces inked
lines, and a photograph has none, only a texture change along a bund. Doing it
automatically needs a segmentation model and a GPU. Doing it by eye takes
twenty minutes and is how cadastral mapping has always worked.

WHAT IS AND IS NOT KEPT HERE

Only the PIXEL coordinates are stored below. Longitude, latitude, area, and
the clip to the data extent are all derived here from the GeoTIFF's own affine
transform, so the layer can be rebuilt and re-checked rather than taken on
trust. No control points are fitted, so no error is added beyond the drone
survey's own georeferencing.

Usage:
    python3 tools/fetch_drone_sample.py --convert    # the imagery first
    python3 tools/digitise_vansar.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

IMAGE = os.path.join(ROOT, "storage", "drone", "vansar_a_deflate.tif")
OUT = os.path.join(ROOT, "samples", "cadastral", "vansar")

# Vertices read off the orthomosaic at 2.2 cm/px, in SOURCE PIXEL coordinates.
# Verified by drawing them back over the imagery and looking, three times: the
# first pass put P01 outside the data extent, the second left P05 overlapping
# P08 by 91,929 px2.
PARCELS = [
    ("P01", "industrial",   [(110, 900), (820, 120), (1600, 130), (1900, 330),
                             (2030, 1020), (1150, 1450), (330, 1530), (130, 1250)]),
    ("P02", "agricultural", [(2920, 1230), (3420, 1180), (3430, 2060), (2930, 2070)]),
    ("P03", "agricultural", [(3460, 1200), (4700, 1140), (4760, 2040), (3470, 2060)]),
    ("P04", "agricultural", [(3450, 2100), (4740, 2070), (4780, 3400), (3480, 3440)]),
    ("P05", "agricultural", [(2580, 2700), (3370, 2660), (3400, 3930), (2600, 3960)]),
    ("P06", "agricultural", [(2180, 1620), (2890, 1270), (2900, 2530), (2200, 2600)]),
    ("P07", "water_body",   [(560, 2780), (1180, 2700), (1380, 3180), (1250, 3600),
                             (760, 3620), (580, 3200)]),
    ("P08", "agricultural", [(3480, 3490), (4800, 3450), (4900, 4250), (3580, 4330)]),
]

CREDIT = "Imagery (c) Hari Prasad K, via OpenAerialMap, CC-BY 4.0."


def find_tool(name):
    """ogr2ogr, from PATH or from the QGIS install that ships one on Windows."""
    found = shutil.which(name)
    if found:
        return found
    for base in ("C:/Program Files", "C:/Program Files (x86)"):
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base), reverse=True):
            if entry.lower().startswith("qgis"):
                candidate = os.path.join(base, entry, "bin", name + ".exe")
                if os.path.exists(candidate):
                    return candidate
    return None


def data_extent(image_path):
    """
    The orthomosaic's real footprint, so parcels can be clipped to it.

    NODATA HERE IS WHITE, NOT BLACK. The PNG previews render it black, which is
    misleading enough that the first attempt masked on darkness and selected
    the entire canvas - reporting a "data footprint" covering 100% of an image
    that is only 76% data.
    """
    import cv2
    import numpy as np
    from shapely.geometry import Polygon

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = ((image.sum(axis=2) < 735) | (image.std(axis=2) > 6)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((35, 35), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 15, True)
    return Polygon([(float(x), float(y))
                    for x, y in outline.reshape(-1, 2)]).buffer(0)


def main() -> int:
    if not os.path.exists(IMAGE):
        print(f"Imagery not found: {IMAGE}")
        print("Run: python3 tools/fetch_drone_sample.py --convert")
        return 1

    import cadastral
    import cityjson
    import georeference
    from shapely.geometry import Polygon

    coefficients, epsg = georeference.read_geotiff_transform(IMAGE)
    a, b, c, d, e, f = coefficients
    extent = data_extent(IMAGE)
    print(f"CRS EPSG:{epsg}, read from the GeoTIFF - no control points fitted\n")

    features = []
    total = 0.0
    print(f"{'id':<5} {'land use':<14} {'vtx':>4} {'area m2':>9}")
    for name, use, ring in PARCELS:
        clipped = Polygon(ring).intersection(extent)
        if clipped.is_empty:
            print(f"{name:<5} lies outside the imagery - skipped")
            continue
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        pixels = list(clipped.exterior.coords)[:-1]
        geo = [(a * px + b * py + c, d * px + e * py + f) for px, py in pixels]
        area = cadastral.polygon_area_m2(geo)
        total += area
        print(f"{name:<5} {use:<14} {len(geo):>4} {area:>9.0f}")
        features.append({
            "type": "Feature",
            "properties": {"parcel_id": name, "land_use": use,
                           "area_m2": round(area, 1),
                           "source": "digitised from open drone imagery"},
            "geometry": {"type": "Polygon", "coordinates": [
                [[round(x, 8), round(y, 8)] for x, y in geo]
                + [[round(geo[0][0], 8), round(geo[0][1], 8)]]]},
        })

    # A cadastre may not contain overlapping parcels, so this is asserted
    # rather than hoped for: pass two of the digitising had P05 crossing P08.
    polygons = {feat["properties"]["parcel_id"]:
                Polygon(feat["geometry"]["coordinates"][0]) for feat in features}
    names = list(polygons)
    overlaps = [(names[i], names[j])
                for i in range(len(names)) for j in range(i + 1, len(names))
                if polygons[names[i]].intersection(polygons[names[j]]).area > 1e-12]
    invalid = [name for name, polygon in polygons.items() if not polygon.is_valid]
    print(f"\ntopology: {len(overlaps)} overlap(s), {len(invalid)} invalid geometry")
    if overlaps or invalid:
        print(f"  {overlaps} {invalid}")
        return 1
    print(f"total digitised: {total / 10000:.2f} hectares")

    os.makedirs(OUT, exist_ok=True)
    collection = {"type": "FeatureCollection", "name": "vansar_parcels",
                  "crs": {"type": "name",
                          "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                  "features": features}
    geojson_path = os.path.join(OUT, "vansar_parcels.geojson")
    with open(geojson_path, "w", encoding="utf-8") as handle:
        json.dump(collection, handle, indent=1)

    with open(os.path.join(OUT, "vansar_parcels.city.json"), "w",
              encoding="utf-8") as handle:
        json.dump(cityjson.to_cityjson(collection), handle)

    ogr = find_tool("ogr2ogr")
    if ogr:
        shapefile = os.path.join(OUT, "vansar_parcels.shp")
        for suffix in (".shp", ".dbf", ".shx", ".prj"):
            try:
                os.remove(shapefile.replace(".shp", suffix))
            except OSError:
                pass
        result = subprocess.run([ogr, "-f", "ESRI Shapefile", shapefile,
                                 geojson_path], capture_output=True, text=True)
        print("shapefile:", "written" if result.returncode == 0
              else f"FAILED - {result.stderr.strip()[:80]}")
    else:
        print("shapefile: skipped, ogr2ogr not found")

    print(f"\n-> {OUT}")
    print(f"\n{CREDIT}")
    print("NOT a legal cadastre: no survey, no title, no khasra, no owner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
