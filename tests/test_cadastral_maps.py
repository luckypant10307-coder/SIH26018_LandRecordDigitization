#!/usr/bin/env python3
"""
Unit tests for multi-village cadastral map support (backend/server.py's map
registry and village-scoped record matching).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The point of holding several villages' maps at once is that khasra numbers
are only unique WITHIN a village, so the tests that matter here are the
collision ones: the same number in two villages must resolve to two different
parcels, and a village with no map must resolve to nothing rather than to
someone else's land.

Run from anywhere with:
    python3 tests/test_cadastral_maps.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import server  # noqa: E402


def _fake_geojson(village_khasras, lat, lon):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"parcel_id": i + 1, "khasra_number": k,
                            "area_m2": 10000.0 + i,
                            "centroid_lat": lat, "centroid_lon": lon},
             "geometry": {"type": "Polygon", "coordinates": [[]]}}
            for i, k in enumerate(village_khasras)
        ],
    }


class MapRegistryTests(unittest.TestCase):
    """list_cadastral_maps() reads directories, so these use real ones."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cadmaps_")
        self._real_store = server.CADASTRAL_STORE
        server.CADASTRAL_STORE = self.tmp
        server._CADASTRAL_CACHE.clear()

    def tearDown(self):
        server.CADASTRAL_STORE = self._real_store
        server._CADASTRAL_CACHE.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add_map(self, folder, village, aliases, district="Somewhere"):
        d = os.path.join(self.tmp, folder)
        os.makedirs(d, exist_ok=True)
        # A 1x1 PNG is enough: the registry only checks that an image exists.
        with open(os.path.join(d, "map.png"), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        with open(os.path.join(d, "control_points.json"), "w", encoding="utf-8") as fh:
            # THREE points, not one. An affine fit needs three, so a map
            # carrying fewer can never be georeferenced and listing it
            # advertises something that fails the moment anyone asks for its
            # parcels. The fixture used a single point while the registry
            # checked only that the FILE existed; once the registry started
            # checking that the points were usable, this stub stopped
            # representing a map that could actually be served.
            json.dump({"village": village, "village_aliases": aliases,
                       "district": district,
                       "control_points": [
                           {"pixel": [0, 0], "lon": 1.0, "lat": 1.0},
                           {"pixel": [100, 0], "lon": 1.001, "lat": 1.0},
                           {"pixel": [0, 100], "lon": 1.0, "lat": 0.999},
                       ]},
                      fh, ensure_ascii=False)
        return d

    def test_a_map_awaiting_its_coordinates_is_not_advertised(self):
        """
        tools/map_from_document.py writes the pixel corners with null lon/lat
        - a form to complete, not a guess to correct. Before the registry
        checked, such a map was LISTED and then raised a raw numpy casting
        error when its parcels were requested. Advertising a map that cannot
        be served is worse than not advertising it.
        """
        d = os.path.join(self.tmp, "awaiting")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "map.png"), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        with open(os.path.join(d, "control_points.json"), "w", encoding="utf-8") as fh:
            json.dump({"village": "Awaiting", "control_points": [
                {"pixel": [0, 0], "lon": None, "lat": None},
                {"pixel": [600, 0], "lon": None, "lat": None},
                {"pixel": [0, 600], "lon": None, "lat": None},
            ]}, fh)
        self.assertNotIn("awaiting", [m["id"] for m in server.list_cadastral_maps()])

    def test_added_maps_are_discovered(self):
        self._add_map("barkhedi", "बरखेडी", ["बरखेडी", "Barkhedi"], "Bhopal")
        ids = [m["id"] for m in server.list_cadastral_maps()]
        self.assertIn("barkhedi", ids)

    def test_directory_without_control_points_is_ignored(self):
        d = os.path.join(self.tmp, "broken")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "map.png"), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        self.assertNotIn("broken", [m["id"] for m in server.list_cadastral_maps()])

    def test_directory_without_an_image_is_ignored(self):
        d = os.path.join(self.tmp, "noimage")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "control_points.json"), "w", encoding="utf-8") as fh:
            json.dump({"village": "X", "control_points": []}, fh)
        self.assertNotIn("noimage", [m["id"] for m in server.list_cadastral_maps()])

    def test_aliases_are_casefolded_for_matching(self):
        self._add_map("barkhedi", "बरखेडी", ["बरखेडी", "Barkhedi"])
        entry = [m for m in server.list_cadastral_maps() if m["id"] == "barkhedi"][0]
        self.assertIn("barkhedi", entry["aliases"])
        self.assertIn("बरखेडी", entry["aliases"])


class VillageScopedMatchingTests(unittest.TestCase):
    """
    The collision case, with the geometry stubbed so no OpenCV or map image is
    needed: two villages that both use khasra 213/1.
    """

    NARHARPUR = {"id": "demo", "village": "नरहरपुर", "district": "Lucknow",
                 "aliases": {"नरहरपुर", "narharpur"},
                 "map_path": "x", "control_points_path": "y", "bundled": True}
    BARKHEDI = {"id": "barkhedi", "village": "बरखेडी", "district": "Bhopal",
                "aliases": {"बरखेडी", "barkhedi"},
                "map_path": "x", "control_points_path": "y", "bundled": False}

    def setUp(self):
        self._real_list = server.list_cadastral_maps
        self._real_geo = server._cadastral_geojson
        server.list_cadastral_maps = lambda: [self.NARHARPUR, self.BARKHEDI]
        server._cadastral_geojson = lambda map_id=None: (
            _fake_geojson(["213/1", "200/3"], 26.8516, 80.9497) if map_id == "demo"
            else _fake_geojson(["209/3", "213/1"], 23.2591, 77.4133))

    def tearDown(self):
        server.list_cadastral_maps = self._real_list
        server._cadastral_geojson = self._real_geo

    def _match(self, village, khasra="213/1"):
        return server._parcel_for_record({
            "khasra_number": {"value": khasra},
            "village": {"value": village},
        })

    def test_same_khasra_resolves_to_the_right_village(self):
        north = self._match("नरहरपुर")
        south = self._match("बरखेडी")
        self.assertIsNotNone(north)
        self.assertIsNotNone(south)
        self.assertEqual(north["map"]["id"], "demo")
        self.assertEqual(south["map"]["id"], "barkhedi")
        # ~600 km apart - if scoping broke, these would be the same parcel.
        self.assertNotEqual(north["props"]["centroid_lat"],
                            south["props"]["centroid_lat"])

    def test_latin_alias_matches_the_same_map(self):
        self.assertEqual(self._match("Barkhedi")["map"]["id"], "barkhedi")

    def test_village_with_no_map_does_not_match_anything(self):
        self.assertIsNone(self._match("किशनपुरा"))

    def test_blank_village_is_never_guessed(self):
        self.assertIsNone(self._match(""))

    def test_khasra_absent_from_that_village_does_not_borrow_another(self):
        """200/3 exists on the demo map only - a Barkhedi record quoting it
        must not be handed the Narharpur parcel."""
        self.assertIsNone(self._match("बरखेडी", "200/3"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
