#!/usr/bin/env python3
"""
Unit tests for backend/shapefile_import.py (importing digitized parcel layers).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Fixtures are real binary shapefiles written by the helpers below rather than
checked-in blobs, so what is parsed is visible in the test and a change to the
writer cannot quietly agree with a matching bug in the reader.

The tests that earn their place are the silent-failure ones:
  * WindingTests    - shapefile rings wind the OPPOSITE way to GeoJSON, and a
                      wrongly-wound polygon still draws correctly in Leaflet.
  * AlignmentTests  - attributes are matched to geometry by POSITION, so a
                      deleted .dbf row must not shift khasra numbers onto the
                      wrong parcels.
  * ProjectedCrsTests - a layer in UTM metres must be refused, not reinterpreted.

Run from anywhere with:
    python3 tests/test_shapefile_import.py -v
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cadastral  # noqa: E402
import shapefile_import as shp  # noqa: E402


# --------------------------------------------------------------------------
# Fixture writers
# --------------------------------------------------------------------------

def write_shp(path, records, shape_type=5):
    """
    Write a .shp holding `records`, each a list of rings of (x, y).

    Follows the ESRI layout: a 100-byte header whose file code and length are
    BIG-endian while everything after is little-endian, then one
    variable-length record per feature.
    """
    xs = [x for rings in records for ring in rings for x, _ in ring]
    ys = [y for rings in records for ring in rings for _, y in ring]
    bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)

    body = b""
    for number, rings in enumerate(records, start=1):
        if not rings:
            content = struct.pack("<i", 0)              # Null shape
        else:
            points = [pt for ring in rings for pt in ring]
            parts, offset = [], 0
            for ring in rings:
                parts.append(offset)
                offset += len(ring)
            rxs = [x for x, _ in points]
            rys = [y for _, y in points]
            content = struct.pack("<i", shape_type)
            content += struct.pack("<4d", min(rxs), min(rys), max(rxs), max(rys))
            content += struct.pack("<ii", len(parts), len(points))
            content += struct.pack("<%di" % len(parts), *parts)
            for x, y in points:
                content += struct.pack("<2d", x, y)
            if shape_type == 15:                        # PolygonZ: Z range + array
                content += struct.pack("<2d", 0.0, 0.0)
                content += struct.pack("<%dd" % len(points), *([0.0] * len(points)))
        body += struct.pack(">ii", number, len(content) // 2) + content

    total_words = (100 + len(body)) // 2
    header = struct.pack(">i", 9994) + b"\x00" * 20 + struct.pack(">i", total_words)
    header += struct.pack("<ii", 1000, shape_type)
    header += struct.pack("<4d", *bbox)
    header += struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    with open(path, "wb") as fh:
        fh.write(header + body)
    return path


def write_dbf(path, field_names, rows, deleted=(), width=20):
    """Write a dBASE III .dbf. `deleted` holds row indices to mark deleted."""
    record_length = 1 + width * len(field_names)
    header_length = 32 + 32 * len(field_names) + 1
    out = struct.pack("<BBBB", 0x03, 26, 1, 1)
    out += struct.pack("<IHH", len(rows), header_length, record_length)
    out += b"\x00" * 20
    for name in field_names:
        out += name.encode("ascii")[:11].ljust(11, b"\x00")
        out += b"C" + b"\x00" * 4 + struct.pack("<BB", width, 0) + b"\x00" * 14
    out += b"\x0D"
    for index, row in enumerate(rows):
        out += b"*" if index in deleted else b" "
        for name in field_names:
            out += str(row.get(name, "")).encode("utf-8")[:width].ljust(width, b" ")
    out += b"\x1A"
    with open(path, "wb") as fh:
        fh.write(out)
    return path


def write_prj(path, wkt):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(wkt)
    return path


WGS84_WKT = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
             'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],'
             'AUTHORITY["EPSG","4326"]]')
UTM44_WKT = ('PROJCS["WGS_1984_UTM_Zone_44N",GEOGCS["GCS_WGS_1984"],'
             'PROJECTION["Transverse_Mercator"],UNIT["Meter",1.0],'
             'AUTHORITY["EPSG","32644"]]')


def clockwise_square(x0, y0, size):
    """A closed CLOCKWISE ring - the ESRI convention for an outer boundary."""
    return [(x0, y0), (x0, y0 + size), (x0 + size, y0 + size),
            (x0 + size, y0), (x0, y0)]


def counter_clockwise_square(x0, y0, size):
    return list(reversed(clockwise_square(x0, y0, size)))


class Fixture:
    """A complete .shp/.dbf/.prj set in a temporary directory."""

    def __init__(self, records, field_names=("KHASRA_NO",), rows=None,
                 deleted=(), prj=WGS84_WKT, shape_type=5, name="parcels"):
        self.dir = tempfile.mkdtemp(prefix="shpimp_")
        self.shp = os.path.join(self.dir, name + ".shp")
        write_shp(self.shp, records, shape_type=shape_type)
        if rows is not None:
            write_dbf(os.path.join(self.dir, name + ".dbf"),
                      list(field_names), rows, deleted=deleted)
        if prj is not None:
            write_prj(os.path.join(self.dir, name + ".prj"), prj)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class GeometryTests(unittest.TestCase):
    def test_a_single_polygon_is_imported(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]],
                     rows=[{"KHASRA_NO": "237/4"}])
        parcels, warnings = shp.load_parcels(fx.shp)
        self.assertEqual(len(parcels), 1)
        self.assertEqual(parcels[0].khasra_number, "237/4")
        self.assertEqual(warnings, [])

    def test_several_polygons_keep_their_own_attributes(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)],
                      [clockwise_square(80.95, 26.85, 0.002)],
                      [clockwise_square(80.97, 26.87, 0.001)]],
                     rows=[{"KHASRA_NO": "1"}, {"KHASRA_NO": "2"},
                           {"KHASRA_NO": "3"}])
        parcels, _ = shp.load_parcels(fx.shp)
        self.assertEqual([p.khasra_number for p in parcels], ["1", "2", "3"])

    def test_the_duplicate_closing_vertex_is_dropped(self):
        """
        A shapefile ring repeats its first point to close. Parcel geometry is
        stored OPEN because parcels_to_geojson closes it itself; leaving the
        duplicate in produces a ring with a doubled vertex.
        """
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}])
        parcels, _ = shp.load_parcels(fx.shp)
        ring = parcels[0].geo_polygon
        self.assertEqual(len(ring), 4)
        self.assertNotEqual(ring[0], ring[-1])

    def test_polygonz_geometry_is_read(self):
        """PolygonZ stores Z after the XY block; XY parsing is unchanged."""
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}],
                     shape_type=15)
        parcels, _ = shp.load_parcels(fx.shp)
        self.assertEqual(len(parcels), 1)
        self.assertEqual(len(parcels[0].geo_polygon), 4)

    def test_a_null_shape_is_skipped_without_shifting_the_rest(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)],
                      [],
                      [clockwise_square(80.95, 26.85, 0.001)]],
                     rows=[{"KHASRA_NO": "1"}, {"KHASRA_NO": "2"},
                           {"KHASRA_NO": "3"}])
        parcels, _ = shp.load_parcels(fx.shp)
        # The null record is dropped, and the third polygon keeps ITS OWN
        # attribute row rather than inheriting the null's.
        self.assertEqual([p.khasra_number for p in parcels], ["1", "3"])


class WindingTests(unittest.TestCase):
    """
    Shapefile outer rings are clockwise; RFC 7946 GeoJSON exteriors must be
    counter-clockwise. Leaflet renders either happily, so this cannot be
    caught by looking at the map - only by checking the numbers.
    """

    def test_a_clockwise_shapefile_ring_becomes_counter_clockwise(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}])
        parcels, _ = shp.load_parcels(fx.shp)
        self.assertGreater(shp.signed_area(parcels[0].geo_polygon), 0.0,
                           "exterior ring must be counter-clockwise for RFC 7946")

    def test_an_already_counter_clockwise_ring_is_left_alone(self):
        """A file that ignores the ESRI convention must not be double-flipped."""
        fx = Fixture([[counter_clockwise_square(80.9, 26.8, 0.001)]], rows=[{}])
        parcels, _ = shp.load_parcels(fx.shp)
        self.assertGreater(shp.signed_area(parcels[0].geo_polygon), 0.0)

    def test_the_emitted_geojson_ring_is_closed_and_counter_clockwise(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]],
                     rows=[{"KHASRA_NO": "237/4"}])
        geojson = shp.to_geojson(fx.shp)
        ring = geojson["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(ring[0], ring[-1], "GeoJSON rings must close")
        self.assertGreater(shp.signed_area([tuple(p) for p in ring[:-1]]), 0.0)


class HoleAndMultipartTests(unittest.TestCase):
    def test_a_hole_is_dropped_and_reported(self):
        """
        Parcel carries one boundary, so an interior ring cannot be
        represented. Dropping it silently would overstate the parcel's area
        with no indication, so it is reported.
        """
        outer = clockwise_square(80.9, 26.8, 0.004)
        hole = counter_clockwise_square(80.901, 26.801, 0.001)
        fx = Fixture([[outer, hole]], rows=[{"KHASRA_NO": "9"}])
        parcels, warnings = shp.load_parcels(fx.shp)
        self.assertEqual(len(parcels), 1)
        self.assertTrue(any("hole" in w.lower() for w in warnings), warnings)

    def test_the_outer_ring_is_chosen_not_the_hole(self):
        outer = clockwise_square(80.9, 26.8, 0.004)
        hole = counter_clockwise_square(80.901, 26.801, 0.001)
        fx = Fixture([[outer, hole]], rows=[{}])
        parcels, _ = shp.load_parcels(fx.shp)
        self.assertAlmostEqual(abs(shp.signed_area(parcels[0].geo_polygon)),
                               0.004 * 0.004, places=9)

    def test_a_multipart_parcel_is_reported(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.002),
                       clockwise_square(80.95, 26.85, 0.001)]], rows=[{}])
        parcels, warnings = shp.load_parcels(fx.shp)
        self.assertEqual(len(parcels), 1)
        self.assertTrue(any("separate pieces" in w for w in warnings), warnings)


class AlignmentTests(unittest.TestCase):
    """
    Regression: read_dbf originally skipped deleted rows, which shifted every
    later row up one and relabelled parcels with a neighbour's khasra number.
    """

    def test_a_deleted_row_does_not_shift_khasra_numbers(self):
        fx = Fixture([[clockwise_square(80.90, 26.80, 0.001)],
                      [clockwise_square(80.92, 26.82, 0.001)],
                      [clockwise_square(80.94, 26.84, 0.001)]],
                     rows=[{"KHASRA_NO": "111"}, {"KHASRA_NO": "222"},
                           {"KHASRA_NO": "333"}],
                     deleted={1})
        parcels, warnings = shp.load_parcels(fx.shp)
        self.assertEqual(len(parcels), 3)
        self.assertEqual(parcels[0].khasra_number, "111")
        self.assertIsNone(parcels[1].khasra_number,
                          "a deleted row must yield no khasra, not the next one")
        self.assertEqual(parcels[2].khasra_number, "333",
                         "parcel 3 was relabelled with another parcel's number")
        self.assertTrue(any("deleted" in w for w in warnings), warnings)

    def test_the_deleted_marker_is_not_offered_as_a_khasra_column(self):
        self.assertIsNone(shp.pick_khasra_field([shp.DELETED_KEY]))


class AttributeFieldTests(unittest.TestCase):
    def test_common_column_spellings_are_recognised(self):
        for name in ("KHASRA_NO", "khasra_number", "Khasra", "SURVEY_NO",
                     "PlotNo", "GAT_NO", "KIT_NO", "parcel_id"):
            self.assertEqual(shp.pick_khasra_field([name, "OWNER"]), name, name)

    def test_an_unrecognised_table_is_reported_not_guessed(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]],
                     field_names=("OWNER", "REMARKS"),
                     rows=[{"OWNER": "Ram Prasad", "REMARKS": "x"}])
        parcels, warnings = shp.load_parcels(fx.shp)
        self.assertIsNone(parcels[0].khasra_number)
        self.assertTrue(any("plot identifier" in w for w in warnings), warnings)

    def test_devanagari_attributes_survive_as_utf8(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]],
                     field_names=("KHASRA_NO", "OWNER"),
                     rows=[{"KHASRA_NO": "२३७/४", "OWNER": "रामप्रसाद वर्मा"}])
        parcels, _ = shp.load_parcels(fx.shp)
        self.assertEqual(parcels[0].khasra_number, "२३७/४")

    def test_a_missing_dbf_is_reported(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=None)
        parcels, warnings = shp.load_parcels(fx.shp)
        self.assertEqual(len(parcels), 1)
        self.assertTrue(any(".dbf" in w for w in warnings), warnings)


class ProjectedCrsTests(unittest.TestCase):
    def test_a_utm_prj_is_refused_with_the_crs_named(self):
        fx = Fixture([[clockwise_square(730000.0, 2975000.0, 50.0)]],
                     rows=[{}], prj=UTM44_WKT)
        with self.assertRaises(shp.ShapefileError) as ctx:
            shp.load_parcels(fx.shp)
        message = str(ctx.exception)
        self.assertIn("UTM_Zone_44N", message)
        self.assertIn("4326", message)

    def test_metre_coordinates_are_refused_even_without_a_prj(self):
        fx = Fixture([[clockwise_square(730000.0, 2975000.0, 50.0)]],
                     rows=[{}], prj=None)
        with self.assertRaises(shp.ShapefileError) as ctx:
            shp.load_parcels(fx.shp)
        self.assertIn("not longitude/latitude", str(ctx.exception))

    def test_a_wgs84_prj_is_accepted_and_its_epsg_recorded(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}])
        name, epsg, projected = shp.read_prj(fx.shp)
        self.assertFalse(projected)
        self.assertEqual(epsg, 4326)
        self.assertEqual(name, "GCS_WGS_1984")

    def test_a_layer_with_no_prj_is_accepted_but_flagged(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}], prj=None)
        _parcels, warnings = shp.load_parcels(fx.shp)
        self.assertTrue(any("no .prj" in w for w in warnings), warnings)


class RejectionTests(unittest.TestCase):
    def test_a_point_layer_is_refused_with_advice(self):
        path = os.path.join(tempfile.mkdtemp(prefix="shpimp_"), "pts.shp")
        write_shp(path, [[[(80.9, 26.8)]]], shape_type=1)
        with self.assertRaises(shp.ShapefileError) as ctx:
            shp.read_shp(path)
        self.assertIn("Point", str(ctx.exception))

    def test_a_polyline_layer_says_how_to_fix_it(self):
        path = os.path.join(tempfile.mkdtemp(prefix="shpimp_"), "lines.shp")
        write_shp(path, [[clockwise_square(80.9, 26.8, 0.001)]], shape_type=3)
        with self.assertRaises(shp.ShapefileError) as ctx:
            shp.read_shp(path)
        self.assertIn("polygons", str(ctx.exception))

    def test_a_non_shapefile_is_refused(self):
        path = os.path.join(tempfile.mkdtemp(prefix="shpimp_"), "not.shp")
        with open(path, "wb") as fh:
            fh.write(b"this is not a shapefile" + b"\x00" * 120)
        with self.assertRaises(shp.ShapefileError):
            shp.read_shp(path)

    def test_a_truncated_header_is_refused(self):
        path = os.path.join(tempfile.mkdtemp(prefix="shpimp_"), "short.shp")
        with open(path, "wb") as fh:
            fh.write(struct.pack(">i", 9994))
        with self.assertRaises(shp.ShapefileError):
            shp.read_shp(path)

    def test_an_empty_layer_is_refused(self):
        path = os.path.join(tempfile.mkdtemp(prefix="shpimp_"), "empty.shp")
        write_shp(path, [], shape_type=5)
        with self.assertRaises(shp.ShapefileError) as ctx:
            shp.load_parcels(path)
        self.assertIn("no usable polygons", str(ctx.exception))


class GeoJsonOutputTests(unittest.TestCase):
    def test_output_matches_the_raster_paths_shape(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]],
                     rows=[{"KHASRA_NO": "237/4"}])
        geojson = shp.to_geojson(fx.shp)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertNotIn("crs", geojson, "RFC 7946 forbids a crs member")
        feature = geojson["features"][0]
        self.assertEqual(feature["properties"]["khasra_number"], "237/4")
        self.assertGreater(feature["properties"]["area_m2"], 0.0)
        self.assertIsNotNone(feature["properties"]["centroid_lon"])

    def test_provenance_says_no_transform_was_fitted(self):
        """
        The whole point of this path: the operator digitized in world
        coordinates, so there is no affine fit that could have gone wrong.
        """
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}])
        meta = shp.to_geojson(fx.shp)["_georeferencing"]
        self.assertEqual(meta["method"], "shapefile")
        self.assertIsNone(meta["transform"])
        self.assertEqual(meta["epsg"], 4326)

    def test_area_is_a_real_ground_measurement(self):
        """0.001 degrees square near Lucknow is about 111m x 99m."""
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}])
        area = shp.to_geojson(fx.shp)["features"][0]["properties"]["area_m2"]
        self.assertGreater(area, 9000.0)
        self.assertLess(area, 12000.0)


class DiscoveryTests(unittest.TestCase):
    def test_a_shapefile_is_found_in_a_map_folder(self):
        fx = Fixture([[clockwise_square(80.9, 26.8, 0.001)]], rows=[{}])
        self.assertEqual(shp.find_shapefile(fx.dir), fx.shp)

    def test_a_folder_without_one_returns_none(self):
        self.assertIsNone(shp.find_shapefile(tempfile.mkdtemp(prefix="shpimp_")))

    def test_a_missing_folder_returns_none(self):
        self.assertIsNone(shp.find_shapefile("/definitely/not/here"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
