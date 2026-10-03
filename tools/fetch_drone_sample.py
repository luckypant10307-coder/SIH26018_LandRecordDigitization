#!/usr/bin/env python3
"""
Fetch openly-licensed drone orthomosaics for development.
Intelligent Land Record Digitization and Validation System - SIH 2026.

WHY A SCRIPT RATHER THAN COMMITTED FILES

The imagery is ~9 MB, it is not ours, and CC-BY attribution travels better in
a file that states the licence than in a blob in git history. `storage/` is
ignored, so this re-creates the samples on any machine instead.

WHAT YOU GET, AND WHAT YOU DO NOT

Two UAV scenes over Vansar, Gujarat at ~2.2 cm per pixel, about a hectare
each. One is farmland with crisply visible field bunds; the other is an
industrial yard with about ten warehouse roofs.

NEITHER CONTAINS A MULTI-STOREY APARTMENT, so neither exercises the vertical
property case PS 26011 is actually about, and neither is anywhere near this
project's corpus - those parcels are in Jaunpur, 1,100 km away. Measured over
OpenAerialMap: 0 open scenes over Jaunpur, 0 over Lucknow, 29 over India as a
whole. These are for developing an extraction pipeline, not for making claims
about Indian land records.

THE FORMAT TRAP, WHICH IS THE REAL REASON THIS SCRIPT EXPLAINS ITSELF

The files are YCbCr JPEG-compressed, internally tiled GeoTIFFs. Both image
libraries this project already depends on refuse them:

    OpenCV 5.0  -> failed TIFFReadRGBATile
    Pillow 12.3 -> OSError: decoder error -2

Only GDAL reads them, so `--convert` shells out to the GDAL CLI to produce a
Deflate GeoTIFF that OpenCV can open. That keeps the existing arrangement
intact: GDAL prepares data as a tool, and the backend still calls no GDAL.

Usage:
    python3 tools/fetch_drone_sample.py
    python3 tools/fetch_drone_sample.py --convert     # also write readable copies
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "storage", "drone")

# Resolved from the OpenAerialMap API (api.openaerialmap.org/meta) on
# 2026-10-03. Pinned rather than re-queried so a run is reproducible; re-query
# if a URL goes stale.
SCENES = [
    {
        "name": "vansar_a",
        "title": "Vansar",
        "url": ("https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/"
                "66c2f1d73bd96f00015aee10/0/66c2f1d73bd96f00015aee11.tif"),
        "bytes": 4489403,
        "gsd_cm": 2.2,
        "content": "farmland; field bunds clearly visible",
    },
    {
        "name": "vansar_b",
        "title": "Vansar, Gujarat",
        "url": ("https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/"
                "66c2eb563bd96f00015aee0c/0/66c2eb563bd96f00015aee0d.tif"),
        "bytes": 4961302,
        "gsd_cm": 2.3,
        "content": "industrial yard; ~10 warehouse roofs",
    },
]

ATTRIBUTION = "Imagery (c) Hari Prasad K, via OpenAerialMap, CC-BY 4.0."


def gdal_tool(name: str) -> str | None:
    """Find a GDAL CLI binary. QGIS ships one on Windows; PATH elsewhere."""
    found = shutil.which(name)
    if found:
        return found
    for base in ("C:/Program Files", "C:/Program Files (x86)"):
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base), reverse=True):
            if not entry.lower().startswith("qgis"):
                continue
            candidate = os.path.join(base, entry, "bin", name + ".exe")
            if os.path.exists(candidate):
                return candidate
    return None


def download(scene: dict) -> str | None:
    """
    Fetch one scene, and REFUSE a short read.

    The size check is not belt-and-braces. The S3 bucket is slow enough that
    the first attempt at this timed out mid-transfer and left two files short
    by about 250 KB each - which is the nastiest possible outcome, because a
    truncated tiled TIFF still opens, still reports its full dimensions, and
    still renders a plausible downsampled preview. It only fails on the tiles
    that were never written:

        TIFFFillTile: Read error at row 3072, col 1024, tile 102;
        got 24538 bytes, expected 34238

    A partial image that looks fine is worse than no image, so an incomplete
    download is deleted rather than kept.
    """
    target = os.path.join(DEST, scene["name"] + ".tif")
    if os.path.exists(target):
        have = os.path.getsize(target)
        if have == scene["bytes"]:
            print(f"  {scene['name']:<10} already complete, skipping")
            return target
        print(f"  {scene['name']:<10} incomplete ({have} of {scene['bytes']} "
              f"bytes), re-fetching")

    print(f"  {scene['name']:<10} fetching {scene['bytes'] / 1e6:.1f} MB ...",
          end="", flush=True)
    try:
        request = urllib.request.Request(
            scene["url"],
            headers={"User-Agent": "LandRecordDigitization/1.0 (SIH 2026)"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f" FAILED ({exc})")
        return None

    if len(payload) != scene["bytes"]:
        print(f" SHORT: got {len(payload)} of {scene['bytes']} bytes, discarding")
        return None

    with open(target, "wb") as handle:
        handle.write(payload)
    print(f" done ({len(payload) / 1e6:.1f} MB)")
    return target


def convert(path: str) -> None:
    """A Deflate copy that OpenCV and Pillow can actually open."""
    translate = gdal_tool("gdal_translate")
    if not translate:
        print("  gdal_translate not found - skipping conversion. Install QGIS "
              "or GDAL to make these readable by OpenCV.")
        return
    out = path.replace(".tif", "_deflate.tif")
    if os.path.exists(out):
        return
    # stderr is captured and the exit code is checked, because the first
    # version of this printed "wrote ..." unconditionally and reported success
    # for a conversion that had failed on a truncated source. A tool that
    # announces work it did not do is worse than one that crashes.
    result = subprocess.run([translate, "-of", "GTiff", "-co", "COMPRESS=DEFLATE",
                             "-co", "TILED=YES", path, out],
                            capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out):
        detail = (result.stderr or "").strip().splitlines()
        print(f"  conversion FAILED for {os.path.basename(path)}"
              + (f": {detail[-1]}" if detail else ""))
        if os.path.exists(out):
            os.remove(out)
        return
    print(f"  wrote {os.path.basename(out)} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert", action="store_true",
                        help="also write Deflate copies OpenCV can read")
    args = parser.parse_args()

    os.makedirs(DEST, exist_ok=True)
    print(f"Fetching {len(SCENES)} UAV scenes into storage/drone/\n")
    for scene in SCENES:
        print(f"  {scene['title']} - {scene['gsd_cm']} cm/px, {scene['content']}")

    print()
    fetched = [p for p in (download(s) for s in SCENES) if p]
    if args.convert:
        print("\nConverting to a format OpenCV can read:")
        for path in fetched:
            convert(path)

    print(f"\n{len(fetched)} of {len(SCENES)} scenes in {DEST}")
    print(f"\n{ATTRIBUTION}")
    print("CC-BY requires that line wherever the imagery, or anything derived")
    print("from it, is shown - slides and screenshots included.")
    print("\nNeither scene contains a multi-storey apartment, so neither")
    print("exercises the vertical-property case; and both are in Gujarat,")
    print("1,100 km from this project's parcels.")
    return 0 if fetched else 1


if __name__ == "__main__":
    sys.exit(main())
