#!/usr/bin/env python3
"""
Unit tests for backend/georeference.py (importing ArcGIS / QGIS georeferencing).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The tests that matter here are the ones covering conventions that fail
SILENTLY - a world file's odd line order and the half-pixel anchor difference
between a world file and a GeoTIFF tiepoint. Both are invisible on a
north-up map and both put every parcel in the wrong place on a rotated one,
which is most scanned cadastral sheets.

Run from anywhere with:
    python3 tests/test_georeference.py -v
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import georeference as gr  # noqa: E402


def write_world_file(path: str, a, b, c, d, e, f) -> str:
    """Write a world file in its real on-disk order: A, D, B, E, C, F."""
    with open(path, "w", encoding="utf-8") as fh:
        for value in (a, d, b, e, c, f):
            fh.write(f"{value!r}\n")
    return path


def write_min_geotiff(path: str, scale_x, scale_y, tie_px, tie_py,
                      tie_x, tie_y, epsg=None) -> str:
    """
    A minimal little-endian TIFF carrying only the georeferencing tags.

    Not a displayable image - georeference.py reads the first IFD for the geo
    tags and nothing else, so a stub is enough to test the tag parsing and
    keeps the fixture readable.
    """
    entries = []
    blobs = b""

    def add(tag, dtype, values, fmt):
        nonlocal blobs, entries
        packed = struct.pack("<" + fmt * len(values), *values)
        if len(packed) <= 4:
            payload = packed.ljust(4, b"\x00")
            entries.append((tag, dtype, len(values), payload, None))
        else:
            entries.append((tag, dtype, len(values), None, len(blobs)))
            blobs += packed

    add(gr._TAG_PIXEL_SCALE, 12, [scale_x, scale_y, 0.0], "d")
    add(gr._TAG_TIEPOINT, 12, [tie_px, tie_py, 0.0, tie_x, tie_y, 0.0], "d")
    if epsg is not None:
        # GeoKeyDirectory: header (version, rev, minor, key count) then one
        # inline key holding the projected-CRS EPSG code.
        keys = [1, 1, 0, 1, gr._GEOKEY_PROJECTED_CS, 0, 1, epsg]
        add(gr._TAG_GEOKEYS, 3, keys, "H")

    ifd_offset = 8
    ifd_size = 2 + len(entries) * 12 + 4
    data_offset = ifd_offset + ifd_size

    out = bytearray()
    out += b"II" + struct.pack("<HI", 42, ifd_offset)
    out += struct.pack("<H", len(entries))
    for tag, dtype, count, payload, blob_at in sorted(entries, key=lambda e: e[0]):
        out += struct.pack("<HHI", tag, dtype, count)
        out += payload if payload is not None else struct.pack("<I", data_offset + blob_at)
    out += struct.pack("<I", 0)
    out += blobs

    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return path


class WorldFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="geo_")

    def test_a_north_up_world_file_round_trips(self):
        path = write_world_file(os.path.join(self.tmp, "m.pgw"),
                                a=0.0001, b=0.0, c=80.9, d=0.0, e=-0.0001, f=26.9)
        a, b, c, d, e, f = gr.read_world_file(path)
        self.assertAlmostEqual(a, 0.0001)
        self.assertAlmostEqual(c, 80.9)
        self.assertAlmostEqual(e, -0.0001)
        self.assertAlmostEqual(f, 26.9)
        self.assertEqual((b, d), (0.0, 0.0))

    def test_the_two_skew_terms_are_not_swapped(self):
        """
        The regression this file exists for. On disk the order is A, D, B, E,
        C, F: line 2 is the Y-skew (d) and line 3 is the X-skew (b). Reading
        them in letter order swaps the rotation and mirrors the map.
        """
        path = os.path.join(self.tmp, "rot.pgw")
        # Distinct values so a swap cannot pass by coincidence.
        write_world_file(path, a=0.0001, b=0.00007, c=80.0,
                         d=0.00003, e=-0.0001, f=27.0)
        a, b, c, d, e, f = gr.read_world_file(path)
        self.assertAlmostEqual(b, 0.00007, msg="x-skew came from the wrong line")
        self.assertAlmostEqual(d, 0.00003, msg="y-skew came from the wrong line")

    def test_extension_is_matched_to_the_raster(self):
        open(os.path.join(self.tmp, "v.png"), "wb").close()
        self.assertIsNone(gr.find_world_file(os.path.join(self.tmp, "v.png")))
        write_world_file(os.path.join(self.tmp, "v.pgw"), 1, 0, 0, 0, -1, 0)
        found = gr.find_world_file(os.path.join(self.tmp, "v.png"))
        self.assertTrue(found and found.endswith(".pgw"))

    def test_generic_wld_is_accepted(self):
        open(os.path.join(self.tmp, "g.png"), "wb").close()
        write_world_file(os.path.join(self.tmp, "g.wld"), 1, 0, 0, 0, -1, 0)
        self.assertTrue(gr.find_world_file(os.path.join(self.tmp, "g.png")))

    def test_a_short_world_file_is_rejected_with_a_count(self):
        path = os.path.join(self.tmp, "short.pgw")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("0.0001\n0.0\n0.0\n")
        with self.assertRaises(gr.GeoreferenceError) as ctx:
            gr.read_world_file(path)
        self.assertIn("three", str(ctx.exception).lower().replace("3", "three"))

    def test_a_non_numeric_line_is_rejected_by_line_number(self):
        path = os.path.join(self.tmp, "junk.pgw")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("0.0001\nnot-a-number\n0.0\n0.0\n0.0\n0.0\n")
        with self.assertRaises(gr.GeoreferenceError) as ctx:
            gr.read_world_file(path)
        self.assertIn("line 2", str(ctx.exception))

    def test_a_zero_x_scale_is_refused(self):
        path = write_world_file(os.path.join(self.tmp, "zero.pgw"),
                                a=0.0, b=0.0, c=80.0, d=0.0, e=-0.0001, f=27.0)
        with self.assertRaises(gr.GeoreferenceError):
            gr.read_world_file(path)


class GeoTiffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="geo_")

    def test_pixel_scale_and_tiepoint_are_read(self):
        path = write_min_geotiff(os.path.join(self.tmp, "m.tif"),
                                 scale_x=0.0002, scale_y=0.0002,
                                 tie_px=0.0, tie_py=0.0, tie_x=80.0, tie_y=27.0)
        (a, b, c, d, e, f), epsg = gr.read_geotiff_transform(path)
        self.assertAlmostEqual(a, 0.0002)
        self.assertAlmostEqual(e, -0.0002)
        self.assertEqual((b, d), (0.0, 0.0))
        self.assertIsNone(epsg)

    def test_the_tiepoint_is_shifted_by_half_a_pixel(self):
        """
        A GeoTIFF tiepoint anchors the CORNER of the upper-left pixel; a world
        file anchors its CENTRE, and so do the pixel coordinates the
        vectoriser emits. Without the half-pixel shift every parcel is offset
        by half a pixel, which on a village sheet is metres on the ground.
        """
        scale = 0.001
        path = write_min_geotiff(os.path.join(self.tmp, "half.tif"),
                                 scale_x=scale, scale_y=scale,
                                 tie_px=0.0, tie_py=0.0, tie_x=80.0, tie_y=27.0)
        (a, b, c, d, e, f), _ = gr.read_geotiff_transform(path)
        self.assertAlmostEqual(c, 80.0 + 0.5 * scale, places=9)
        self.assertAlmostEqual(f, 27.0 - 0.5 * scale, places=9)

    def test_a_tiff_without_geo_tags_says_so(self):
        path = os.path.join(self.tmp, "plain.tif")
        with open(path, "wb") as fh:
            fh.write(b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", 0)
                     + struct.pack("<I", 0))
        with self.assertRaises(gr.GeoreferenceError) as ctx:
            gr.read_geotiff_transform(path)
        self.assertIn("no georeferencing tags", str(ctx.exception))

    def test_a_non_tiff_is_rejected(self):
        path = os.path.join(self.tmp, "fake.tif")
        with open(path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        with self.assertRaises(gr.GeoreferenceError):
            gr.read_geotiff_transform(path)

    def test_bigtiff_is_refused_with_advice(self):
        path = os.path.join(self.tmp, "big.tif")
        with open(path, "wb") as fh:
            fh.write(b"II" + struct.pack("<HI", 43, 8) + b"\x00" * 8)
        with self.assertRaises(gr.GeoreferenceError) as ctx:
            gr.read_geotiff_transform(path)
        self.assertIn("BigTIFF", str(ctx.exception))


class ProjectedCrsTests(unittest.TestCase):
    """
    A projected CRS must be refused, not reinterpreted. UTM coordinates read
    as degrees do not land slightly off - they land off the planet.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="geo_")

    def test_utm_metres_are_refused(self):
        path = write_world_file(os.path.join(self.tmp, "utm.pgw"),
                                a=0.5, b=0.0, c=730000.0,
                                d=0.0, e=-0.5, f=2975000.0)
        with self.assertRaises(gr.GeoreferenceError) as ctx:
            gr.check_lonlat(gr.read_world_file(path), 2000, 2000, None, None)
        self.assertIn("not a longitude/latitude", str(ctx.exception))

    def test_the_refusal_names_the_epsg_code(self):
        with self.assertRaises(gr.GeoreferenceError) as ctx:
            gr.check_lonlat((0.0001, 0, 80.9, 0, -0.0001, 26.9),
                            100, 100, 32644, "WGS_1984_UTM_Zone_44N")
        message = str(ctx.exception)
        self.assertIn("32644", message)
        self.assertIn("UTM_Zone_44N", message)
        self.assertIn("4326", message)

    def test_epsg_4326_is_accepted(self):
        gr.check_lonlat((0.0001, 0, 80.9, 0, -0.0001, 26.9), 100, 100, 4326, None)

    def test_a_corner_off_the_earth_is_caught_even_when_the_origin_looks_fine(self):
        """
        Checked at all four corners on purpose: a bad scale can leave the
        origin looking like a plausible lon/lat and still run off the Earth
        by the far edge of the sheet.
        """
        with self.assertRaises(gr.GeoreferenceError):
            gr.check_lonlat((1.0, 0, 80.9, 0, -1.0, 26.9), 2000, 2000, None, None)


class AuxCrsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="geo_")

    def test_crs_name_is_read_from_an_arcgis_aux_xml(self):
        image = os.path.join(self.tmp, "m.png")
        open(image, "wb").close()
        with open(image + ".aux.xml", "w", encoding="utf-8") as fh:
            fh.write('<PAMDataset><SRS>PROJCS["WGS_1984_UTM_Zone_44N",'
                     'GEOGCS["GCS_WGS_1984"]]</SRS></PAMDataset>')
        self.assertEqual(gr.read_aux_crs(image), "WGS_1984_UTM_Zone_44N")

    def test_a_missing_aux_file_is_not_an_error(self):
        self.assertIsNone(gr.read_aux_crs(os.path.join(self.tmp, "absent.png")))

    def test_malformed_xml_is_not_an_error(self):
        image = os.path.join(self.tmp, "bad.png")
        open(image, "wb").close()
        with open(image + ".aux.xml", "w", encoding="utf-8") as fh:
            fh.write("<PAMDataset><SRS>unclosed")
        self.assertIsNone(gr.read_aux_crs(image))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="geo_")

    def test_no_sidecar_returns_none_so_control_points_still_work(self):
        image = os.path.join(self.tmp, "plain.png")
        open(image, "wb").close()
        self.assertIsNone(gr.discover(image, 100, 100))

    def test_a_world_file_is_preferred_and_reports_its_provenance(self):
        image = os.path.join(self.tmp, "m.png")
        open(image, "wb").close()
        write_world_file(os.path.join(self.tmp, "m.pgw"),
                         a=0.0001, b=0.0, c=80.9, d=0.0, e=-0.0001, f=26.9)
        found = gr.discover(image, 500, 400)
        self.assertIsNotNone(found)
        self.assertEqual(found.source, "world_file")
        self.assertFalse(found.rotated)
        lon, lat = found.apply(0, 0)
        self.assertAlmostEqual(lon, 80.9)
        self.assertAlmostEqual(lat, 26.9)

    def test_rotation_is_reported(self):
        image = os.path.join(self.tmp, "r.png")
        open(image, "wb").close()
        write_world_file(os.path.join(self.tmp, "r.pgw"),
                         a=0.0001, b=0.00002, c=80.9,
                         d=0.00001, e=-0.0001, f=26.9)
        self.assertTrue(gr.discover(image, 500, 400).rotated)

    def test_a_floating_point_crumb_is_not_reported_as_rotation(self):
        """
        Regression: `rotated` was `bool(b or d)`, so the north-up demo map -
        whose least-squares fit leaves an x-skew of -2.5e-13 - was reported
        as rotated. Non-zero is not the same as significant.
        """
        image = os.path.join(self.tmp, "crumb.png")
        open(image, "wb").close()
        write_world_file(os.path.join(self.tmp, "crumb.pgw"),
                         a=4.444444679088419e-06, b=-2.5268676040468563e-13,
                         c=80.949, d=7.783157160934096e-14,
                         e=-4.444444679088419e-06, f=26.855)
        self.assertFalse(gr.discover(image, 900, 900).rotated)

    def test_a_real_rotation_is_still_reported(self):
        image = os.path.join(self.tmp, "tilt.png")
        open(image, "wb").close()
        # ~2 degrees of tilt, typical of a hand-placed scan.
        scale = 4.4e-06
        write_world_file(os.path.join(self.tmp, "tilt.pgw"),
                         a=scale, b=scale * 0.035, c=80.949,
                         d=scale * 0.035, e=-scale, f=26.855)
        self.assertTrue(gr.discover(image, 900, 900).rotated)

    def test_an_unusable_georeferencing_raises_instead_of_returning_none(self):
        """
        None means "there was nothing to find" and sends the caller to
        control_points.json. A georeferencing that exists but is in the wrong
        CRS must not be laundered into that - the surveyor has to be told
        their work was rejected and why.
        """
        image = os.path.join(self.tmp, "utm.png")
        open(image, "wb").close()
        write_world_file(os.path.join(self.tmp, "utm.pgw"),
                         a=0.5, b=0.0, c=730000.0, d=0.0, e=-0.5, f=2975000.0)
        with self.assertRaises(gr.GeoreferenceError):
            gr.discover(image, 2000, 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
