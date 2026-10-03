#!/usr/bin/env python3
"""
One-time regional import of Google Open Buildings v3 footprints.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

WHY THIS IS A TOOL AND NOT AN API CALL

`building_height` reads a tiled raster, so one parcel costs two HTTP range
reads - about 80 KB against a 270 MB tile, fast enough to run inside a
request. Footprints are gzipped CSV, and gzip cannot be seeked: reaching the
last row means decompressing every row before it.

The files are one per S2 level-6 cell. Median 9 MB across the bucket, but the
Gangetic plain is densely built and the cell covering Jaunpur - where this
project's documents come from - is 988 MB.

So the region is streamed ONCE, filtered to a bounding box, and written to
storage/open_buildings/<cell>.ndjson. Every parcel inside it is then answered
offline, with no network at all. That is also why this belongs beside the
other preparation tools rather than in the server.

Usage:
    python3 tools/import_buildings.py --around 25.75 82.68 --radius-km 5
    python3 tools/import_buildings.py --bbox 82.55 25.90 82.70 26.00
    python3 tools/import_buildings.py --around 18.59 73.74 --radius-km 2 --dry-run

The data is CC-BY 4.0 / ODbL 1.0 and the attribution must be shown wherever a
footprint from it appears.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import open_buildings as ob  # noqa: E402


def bbox_around(lat: float, lon: float, radius_km: float):
    """A degree box around a point. Longitude narrows with latitude."""
    d_lat = radius_km / 111.32
    d_lon = radius_km / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--around", nargs=2, type=float, metavar=("LAT", "LON"),
                       help="centre of the region to import")
    group.add_argument("--bbox", nargs=4, type=float,
                       metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--min-confidence", type=float, default=ob.MIN_CONFIDENCE)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the cell and its size, download nothing")
    args = parser.parse_args()

    if args.around:
        lat, lon = args.around
        bounds = bbox_around(lat, lon, args.radius_km)
    else:
        west, south, east, north = args.bbox
        bounds = (west, south, east, north)
        lat, lon = (south + north) / 2, (west + east) / 2

    token = ob.cell_token(lat, lon)
    size = ob.cell_size(token)

    print("Open Buildings v3 regional import")
    print("-" * 58)
    print(f"  centre     : {lat:.5f}, {lon:.5f}")
    print(f"  bounds     : W {bounds[0]:.5f}  S {bounds[1]:.5f}  "
          f"E {bounds[2]:.5f}  N {bounds[3]:.5f}")
    print(f"  S2 cell    : {token}")
    print(f"  cell file  : {size / 1e6:.0f} MB" if size else "  cell file  : unknown")
    print(f"  confidence : keeping >= {args.min_confidence}")

    if size and size > ob.STREAM_WARN_BYTES:
        print(f"\n  NOTE: this cell is {size / 1e6:.0f} MB. The whole file is")
        print("  decompressed to find the rows inside the bounding box, because")
        print("  gzip cannot be seeked. It is a one-time pass for this region.")

    if args.dry_run:
        print("\n  --dry-run, nothing fetched.")
        return 0

    if not ob.ENABLED:
        print("\n  OPEN_BUILDINGS is not set to 1, so nothing will be fetched.")
        print("  Re-run with OPEN_BUILDINGS=1 to allow the download.")
        return 1

    started = time.time()

    def progress(seen, kept):
        print(f"    {seen:>10,} rows scanned, {kept:>7,} kept "
              f"({time.time() - started:.0f}s)")

    print()
    try:
        buildings = ob.stream_cell(token, bounds,
                                   min_confidence=args.min_confidence,
                                   progress=progress)
    except ob.OpenBuildingsError as exc:
        print(f"  FAILED: {exc}")
        return 1

    if not buildings:
        print("  No buildings found in that box. Check the coordinates: this "
              "dataset covers Africa, South and South-East Asia, Latin America "
              "and the Caribbean.")
        return 1

    written = ob.store_subset(token, buildings)
    areas = sorted(b["area_m2"] for b in buildings if b.get("area_m2"))
    print(f"\n  kept {written:,} buildings in {time.time() - started:.0f}s")
    if areas:
        print(f"  area: median {areas[len(areas) // 2]:.0f} m2, "
              f"largest {areas[-1]:.0f} m2")
    print(f"  -> {os.path.join(ob.CACHE_DIR, token + '.ndjson')}")
    print(f"\n  {ob.ATTRIBUTION}")
    print("  Show that wherever a footprint from this import is displayed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
