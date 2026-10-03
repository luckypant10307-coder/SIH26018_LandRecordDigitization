#!/usr/bin/env python3
"""
Unit tests for Google Open Buildings v3 footprints (backend/open_buildings.py).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

NOTHING HERE TOUCHES THE NETWORK.

The S2 tokens are asserted against values verified once against the live
bucket, so a regression in the Hilbert walk is caught without a request. The
important property is not that the arithmetic is self-consistent but that it
names files that actually exist - Jaunpur, Amari, Lucknow, Hinjewadi and
Vansar were each checked against the real listing when these were written.

Run from anywhere with:
    python3 tests/test_open_buildings.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import open_buildings as ob  # noqa: E402


class TestS2Addressing(unittest.TestCase):
    """
    The file name IS the S2 cell token, so this arithmetic decides whether a
    lookup finds anything at all.
    """

    # Verified against the live bucket listing: every one of these exists as
    # <token>_buildings.csv.gz under polygons_s2_level_6_gzip_no_header.
    KNOWN = {
        "Jaunpur": (25.75, 82.68, "3991"),
        "Amari": (25.95, 82.60, "3991"),
        "Lucknow": (26.85, 80.95, "399b"),
        "Hinjewadi": (18.59, 73.74, "3bc3"),
        "Vansar": (22.718, 72.713, "395f"),
    }

    def test_known_places_resolve_to_real_cells(self):
        for name, (lat, lon, token) in self.KNOWN.items():
            self.assertEqual(ob.cell_token(lat, lon), token, name)

    def test_a_token_is_a_level_6_token(self):
        """Three bits of face plus 12 of position: four hex characters."""
        for lat, lon, _ in self.KNOWN.values():
            self.assertLessEqual(len(ob.cell_token(lat, lon)), 4)

    def test_neighbouring_villages_share_a_cell(self):
        """
        Jaunpur and Amari are 25 km apart and fall in the same cell, which is
        why one regional import answers a whole district offline.
        """
        self.assertEqual(ob.cell_token(25.75, 82.68),
                         ob.cell_token(25.95, 82.60))

    def test_distant_places_do_not(self):
        self.assertNotEqual(ob.cell_token(25.75, 82.68),
                            ob.cell_token(18.59, 73.74))

    def test_the_url_is_built_from_the_token(self):
        url = ob.cell_url("3991")
        self.assertTrue(url.endswith("/3991_buildings.csv.gz"))
        self.assertIn("polygons_s2_level_6", url)


class TestWktParsing(unittest.TestCase):

    SQUARE = ("POLYGON((82.2706 25.7575, 82.2708 25.7575, "
              "82.2708 25.7577, 82.2706 25.7577, 82.2706 25.7575))")

    def test_a_real_polygon_parses(self):
        ring = ob.parse_wkt_polygon(self.SQUARE)
        self.assertEqual(len(ring), 4)
        self.assertAlmostEqual(ring[0][0], 82.2706)
        self.assertAlmostEqual(ring[0][1], 25.7575)

    def test_the_closing_vertex_is_dropped(self):
        """
        WKT closes its rings and nothing else in this system does. Keeping the
        repeat would put a zero-length edge in every footprint.
        """
        ring = ob.parse_wkt_polygon(self.SQUARE)
        self.assertNotEqual(ring[0], ring[-1])

    def test_garbage_is_refused(self):
        for bad in ("", "LINESTRING(1 2, 3 4)", "POLYGON(())", "not wkt"):
            with self.assertRaises((ob.OpenBuildingsError, ValueError)):
                ob.parse_wkt_polygon(bad)


class TestGeometry(unittest.TestCase):

    PARCEL = [(82.2700, 25.7570), (82.2720, 25.7570),
              (82.2720, 25.7590), (82.2700, 25.7590)]

    def test_a_point_inside_is_inside(self):
        self.assertTrue(ob.point_in_ring(82.2710, 25.7580, self.PARCEL))

    def test_a_point_outside_is_outside(self):
        self.assertFalse(ob.point_in_ring(82.2800, 25.7580, self.PARCEL))
        self.assertFalse(ob.point_in_ring(82.2710, 25.7700, self.PARCEL))

    def test_bounds(self):
        west, south, east, north = ob.ring_bounds(self.PARCEL)
        self.assertAlmostEqual(west, 82.2700)
        self.assertAlmostEqual(north, 25.7590)


class TestParcelLookup(unittest.TestCase):
    """
    The guards matter as much as the lookup: almost every footprint in this
    system is in map pixels, and a pixel ring must never be used to address a
    cell on the earth.
    """

    def setUp(self):
        self._cache = ob.CACHE_DIR
        ob.CACHE_DIR = tempfile.mkdtemp()

    def tearDown(self):
        ob.CACHE_DIR = self._cache

    PARCEL = [(82.2700, 25.7570), (82.2720, 25.7570),
              (82.2720, 25.7590), (82.2700, 25.7590)]

    def seed(self):
        token = ob.cell_token(25.7580, 82.2710)
        ob.store_subset(token, [
            {"lat": 25.7580, "lon": 82.2710, "area_m2": 96.0,
             "confidence": 0.88,
             "footprint": [[82.2708, 25.7578], [82.2712, 25.7578],
                           [82.2712, 25.7582], [82.2708, 25.7582]]},
            {"lat": 25.7584, "lon": 82.2714, "area_m2": 41.5,
             "confidence": 0.79,
             "footprint": [[82.2713, 25.7583], [82.2715, 25.7583],
                           [82.2715, 25.7585], [82.2713, 25.7585]]},
            # outside the parcel - must not be returned
            {"lat": 25.7700, "lon": 82.2900, "area_m2": 500.0,
             "confidence": 0.95, "footprint": [[82.29, 25.77], [82.291, 25.77],
                                               [82.291, 25.771], [82.29, 25.771]]},
        ])
        return token

    def test_buildings_on_the_parcel_are_found(self):
        self.seed()
        result = ob.buildings_in_parcel(self.PARCEL)
        self.assertIsNotNone(result)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["total_area_m2"], 137.5)

    def test_the_largest_is_identified(self):
        """A parcel's main structure, which is what a stack is declared on."""
        self.seed()
        result = ob.buildings_in_parcel(self.PARCEL)
        self.assertEqual(result["largest"]["area_m2"], 96.0)

    def test_a_building_outside_the_parcel_is_excluded(self):
        self.seed()
        result = ob.buildings_in_parcel(self.PARCEL)
        self.assertTrue(all(b["area_m2"] != 500.0 for b in result["buildings"]))

    def test_the_area_is_in_square_metres(self):
        """
        The point of this dataset for this project: a real area, replacing an
        area in pixels of a scanned map.
        """
        self.seed()
        result = ob.buildings_in_parcel(self.PARCEL)
        self.assertGreater(result["largest"]["area_m2"], 0)

    def test_confidence_travels_with_each_building(self):
        """A 0.7 outline used to settle a boundary is not a 0.9 one."""
        self.seed()
        for building in ob.buildings_in_parcel(self.PARCEL)["buildings"]:
            self.assertTrue(0 <= building["confidence"] <= 1)

    def test_a_pixel_footprint_is_refused(self):
        """(246, 20) is a valid coordinate pair in the Atlantic."""
        self.seed()
        self.assertIsNone(ob.buildings_in_parcel(
            [(246, 20), (358, 103), (228, 333), (145, 268)]))

    def test_a_ring_spanning_degrees_is_refused(self):
        self.assertIsNone(ob.buildings_in_parcel(
            [(20, 30), (80, 30), (80, 85), (20, 85)]))

    def test_no_import_for_the_region_returns_none(self):
        """Not an error: the caller keeps the parcel footprint it had."""
        self.assertIsNone(ob.buildings_in_parcel(self.PARCEL))

    def test_a_degenerate_ring_returns_none(self):
        self.assertIsNone(ob.buildings_in_parcel([(82.27, 25.75)]))
        self.assertIsNone(ob.buildings_in_parcel([]))

    def test_the_attribution_travels_with_the_answer(self):
        """CC-BY is a licence condition, not a courtesy."""
        self.seed()
        result = ob.buildings_in_parcel(self.PARCEL)
        self.assertIn("CC-BY", result["attribution"])
        self.assertIn("not a survey", result["caveat"])


class TestSubsetStore(unittest.TestCase):

    def setUp(self):
        self._cache = ob.CACHE_DIR
        ob.CACHE_DIR = tempfile.mkdtemp()

    def tearDown(self):
        ob.CACHE_DIR = self._cache

    def test_a_subset_round_trips(self):
        rows = [{"lat": 1.0, "lon": 2.0, "area_m2": 10.0, "confidence": 0.9,
                 "footprint": [[2.0, 1.0], [2.1, 1.0], [2.1, 1.1]]}]
        self.assertEqual(ob.store_subset("abcd", rows), 1)
        self.assertEqual(ob.read_subset("abcd"), rows)

    def test_an_absent_cell_reads_as_empty(self):
        self.assertEqual(ob.read_subset("nope"), [])

    def test_cached_cells_are_listed(self):
        ob.store_subset("3991", [])
        self.assertIn("3991", ob.cached_cells())

    def test_describe_states_the_access_pattern(self):
        """
        The thing a reader most needs to know: this is a regional import, not
        a per-parcel fetch like building_height.
        """
        description = ob.describe()
        self.assertIn("regional import", description["access"])
        self.assertIn("CC-BY", description["attribution"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
