#!/usr/bin/env python3
"""
Turn the parcel map inside a plot report into a cadastral map the system can
georeference.

THE GAP THIS CLOSES

The pipeline now finds the parcel map embedded in every Bhu-Naksha plot
report (20 of 20 on the real corpus), reads the highlighted parcel and its
neighbours, and uses them to check the khasra. What it cannot do is say
where any of it is on the earth, because a plot report carries no world file
and no control points - so the geometry stays in pixels and the record falls
back to a place-name geotag measured in tens of kilometres.

server.py already knows how to serve a georeferenced map: a folder under
storage/cadastral/ holding an image plus control_points.json, or a shapefile
on its own. The piece that was missing is getting the document's own map
into that shape. This writes it.

WHAT IT DOES AND DOES NOT DO

It produces a map folder with REAL pixel geometry and PLACEHOLDER
coordinates, clearly marked as such. It does not invent a location: the
control points it writes are nulls with the pixel corners filled in, and the
map is not served until someone supplies the four longitude/latitude values.
Writing plausible-looking coordinates would be worse than writing none,
because a wrong geotag on a land record is indistinguishable from a right
one until somebody visits the field.

THE BETTER ROUTE, IF IT IS OPEN TO YOU

If your state's Bhu-Naksha portal offers a SHAPEFILE or GeoJSON download for
the village, use that instead and skip this entirely: drop the .shp/.dbf/.prj
set into storage/cadastral/<village>/ with a control_points.json carrying
just the village name, and it is served immediately at metre accuracy with
no georeferencing step at all. This tool is for when only the PDF exists.

Run:
    python3 tools/map_from_document.py <document.pdf> [--village NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

# A Windows console defaults to cp1252, which cannot encode Devanagari - so
# printing the village name this tool just read off the document killed it
# with a UnicodeEncodeError after the work was already done. A tool for
# Indian land records has to be able to say "नरहरपुर" out loud.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import ocr_engine          # noqa: E402
import parcel_map          # noqa: E402
import field_extractor     # noqa: E402
import gazetteer           # noqa: E402

STORE = os.path.join(ROOT, "storage", "cadastral")


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9ऀ-ॿ]+", "-", (name or "").strip()).strip("-")
    return (slug or "village").lower()[:48]


def build(pdf_path: str, village: str = None, district: str = None) -> int:
    if not os.path.exists(pdf_path):
        print(f"No such file: {pdf_path}")
        return 1

    result = ocr_engine.extract(pdf_path)
    maps = getattr(result, "embedded_maps", None) or []
    if not maps:
        print("This document embeds no parcel map, so there is nothing to "
              "georeference.")
        return 1

    fields = {f.key: f.to_dict()["value"]
              for f in field_extractor.extract_fields(result.lines)}

    # The same prose recovery the ingest pipeline applies, for the same
    # reason. Reading the extractor alone gave the village as
    # "अमारी, परगना गड़वारा, तहसील बदलापुर" - the name with two further
    # administrative levels glued on, because the owner line carries three
    # label:value pairs and the boundary lands after the last colon. That
    # became the folder name, so a tool that skips the recovery does not
    # merely report a worse value, it files the map under one.
    text = "\n".join(getattr(l, "text", "") for l in result.lines)
    try:
        recovered = gazetteer.places_from_prose(text)
    except Exception:
        recovered = {}
    for key in ("village", "district"):
        raw = fields.get(key)
        confirmed = None
        try:
            confirmed = gazetteer._confirm_place(key, raw) if raw else None
        except Exception:
            confirmed = None
        if confirmed:
            fields[key] = confirmed          # the master's own spelling
        elif recovered.get(key):
            fields[key] = recovered[key]["value"]

    village = village or fields.get("village")
    district = district or fields.get("district")
    if not village:
        print("No village name could be read from the document, and a map "
              "folder needs one - parcel-to-document matching is scoped by "
              "village so two villages reusing the same khasra numbers "
              "cannot cross-match. Pass --village.")
        return 1

    reading = parcel_map.read_map(maps[0])
    folder = os.path.join(STORE, slugify(village))
    os.makedirs(folder, exist_ok=True)

    image_name = "map" + os.path.splitext(maps[0])[1]
    shutil.copyfile(maps[0], os.path.join(folder, image_name))

    try:
        from PIL import Image
        with Image.open(maps[0]) as im:
            width, height = im.size
    except Exception:
        width = height = 0

    control_path = os.path.join(folder, "control_points.json")
    if os.path.exists(control_path):
        print(f"{control_path} already exists - leaving it alone.")
    else:
        # The CORNERS are known exactly; only their coordinates are not. So
        # the pixel side is filled in and the lon/lat left null, which makes
        # the file a form to complete rather than a guess to correct. A null
        # also stops the map being served with a made-up location, because
        # the loader cannot fit a transform from it.
        corners = [[0, 0], [width, 0], [0, height], [width, height]]
        payload = {
            "_comment": (
                "Extracted from the parcel map embedded in "
                f"{os.path.basename(pdf_path)}. The pixel corners are real. "
                "The lon/lat values are NULL and must be filled in before "
                "this map can place anything on the earth - open the image "
                "in QGIS beside satellite imagery or a Bhuvan layer, "
                "identify each corner, and write its coordinates here. "
                "Until then the system keeps using the place-name geotag, "
                "which is honest about being accurate to kilometres."),
            "control_points": [
                {"pixel": corner, "lon": None, "lat": None} for corner in corners
            ],
            "village": village,
            "village_aliases": [v for v in {village, (village or "").title()} if v],
            "district": district,
            "source_document": os.path.basename(pdf_path),
            "map_reading": reading.to_dict(),
        }
        with open(control_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {folder}")
    print(f"  map image        : {image_name}  ({width}x{height} px)")
    print(f"  village          : {village}"
          + (f" / {district}" if district else ""))
    if reading.subject_label:
        print(f"  subject parcel   : khasra {reading.subject_label}, "
              f"{len(reading.subject_polygon)} vertices, "
              f"{reading.subject_area_px:.0f} px")
    if reading.neighbour_labels:
        print(f"  neighbours       : {', '.join(reading.neighbour_labels)}")
    print()
    print("NOT YET GEOREFERENCED. Fill the four lon/lat values in")
    print(f"  {control_path}")
    print("and this village is served at metre accuracy. Until then the "
          "record keeps its")
    print("place-name geotag, which states its own accuracy in kilometres.")
    print()
    print("If your state portal offers the village as a SHAPEFILE, drop the "
          ".shp/.dbf/.prj")
    print("set into this folder instead - it needs no control points at all.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("document", help="a plot report PDF")
    ap.add_argument("--village", help="override the village read from the document")
    ap.add_argument("--district", help="override the district")
    args = ap.parse_args()
    return build(args.document, args.village, args.district)


if __name__ == "__main__":
    raise SystemExit(main())
