#!/usr/bin/env python3
"""Tests for CityJSON cadastral parcel import and export."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cityjson  # noqa: E402


REFERENCE_SYSTEM = "https://www.opengis.net/def/crs/EPSG/0/4979"


def cityjson_fixture(reference_system=REFERENCE_SYSTEM):
    return {
        "type": "CityJSON",
        "version": "2.0",
        "metadata": {"referenceSystem": reference_system},
        "CityObjects": {
            "plot-237-4": {
                "type": "LandUse",
                "attributes": {"khasra_no": "237/4", "owner": "Example"},
                "geometry": [{
                    "type": "MultiSurface",
                    "lod": "0",
                    "boundaries": [[[0, 1, 2, 3, 0]]],
                }],
            },
        },
        "vertices": [
            [80.9, 26.8, 115.0],
            [80.901, 26.8, 115.0],
            [80.901, 26.801, 115.0],
            [80.9, 26.801, 115.0],
        ],
    }


class CityJSONImportTests(unittest.TestCase):
    def test_import_decodes_transform_and_keeps_attributes_and_height(self):
        document = cityjson_fixture()
        document["transform"] = {
            "scale": [0.000001, 0.000001, 0.1],
            "translate": [80.0, 26.0, 100.0],
        }
        document["vertices"] = [
            [900000, 800000, 150], [901000, 800000, 150],
            [901000, 801000, 150], [900000, 801000, 150],
        ]

        result = cityjson.to_geojson(document, "source.city.json")
        feature = result["features"][0]
        self.assertEqual(feature["properties"]["khasra_number"], "237/4")
        self.assertEqual(feature["properties"]["owner"], "Example")
        self.assertGreater(feature["properties"]["area_m2"], 0)
        self.assertEqual(feature["properties"]["centroid_lon"], 80.9005)
        self.assertEqual(feature["geometry"]["coordinates"][0][0],
                         [80.9, 26.8, 115.0])
        self.assertEqual(result["_georeferencing"]["source_file"], "source.city.json")

    def test_import_solid_uses_the_exterior_shell(self):
        document = cityjson_fixture()
        geometry = document["CityObjects"]["plot-237-4"]["geometry"][0]
        geometry["type"] = "Solid"
        geometry["boundaries"] = [[[[0, 1, 2, 3, 0]]]]
        self.assertEqual(len(cityjson.to_geojson(document)["features"]), 1)

    def test_rejects_non_geographic_reference_system(self):
        document = cityjson_fixture(
            "https://www.opengis.net/def/crs/EPSG/0/32644")
        with self.assertRaisesRegex(cityjson.CityJSONError, "EPSG:4326 or EPSG:4979"):
            cityjson.to_geojson(document)

    def test_accepts_the_standard_unclosed_ring(self):
        """
        THE INTEROPERABILITY TEST, and it used to assert the opposite.

        CityJSON rings are IMPLICITLY CLOSED - the first vertex is not
        repeated, the reverse of GeoJSON. The spec's own cube example lists
        four indices per square face, not five, and it inherits the convention
        from Wavefront OBJ, which it cites.

        Requiring closure rejected every file a standard producer emits (cjio,
        3dfier, the 3D BAG, FME) while still round-tripping with our own
        exporter - so the suite passed and the feature could not actually
        exchange data with anything, which is the only reason to adopt a
        standard format at all.
        """
        document = cityjson_fixture()
        document["CityObjects"]["plot-237-4"]["geometry"][0]["boundaries"] = [
            [[0, 1, 2, 3]]
        ]
        result = cityjson.to_geojson(document)
        ring = result["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(ring[0], ring[-1],
                         "GeoJSON output must be closed even though the input was not")
        self.assertEqual(len(ring), 5, "four distinct corners plus the closure")

    def test_tolerates_a_redundant_repeated_vertex(self):
        """Some producers do repeat it; dropping it loses nothing."""
        document = cityjson_fixture()
        document["CityObjects"]["plot-237-4"]["geometry"][0]["boundaries"] = [
            [[0, 1, 2, 3, 0]]
        ]
        result = cityjson.to_geojson(document)
        ring = result["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(len(ring), 5)
        self.assertEqual(ring[0], ring[-1])

    def test_accepts_a_triangle(self):
        """
        Three indices is a valid ring and the commonest CityJSON primitive -
        the previous four-index minimum refused every triangulated surface.
        """
        document = cityjson_fixture()
        document["CityObjects"]["plot-237-4"]["geometry"][0]["boundaries"] = [
            [[0, 1, 2]]
        ]
        result = cityjson.to_geojson(document)
        ring = result["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(len(ring), 4)            # 3 corners + closure
        self.assertEqual(ring[0], ring[-1])

    def test_a_degenerate_ring_is_still_refused(self):
        """Leniency about closure must not become leniency about nonsense."""
        document = cityjson_fixture()
        document["CityObjects"]["plot-237-4"]["geometry"][0]["boundaries"] = [
            [[0, 1]]
        ]
        with self.assertRaises(cityjson.CityJSONError):
            cityjson.to_geojson(document)


class CityJSONRingConventionTests(unittest.TestCase):
    """What we EMIT has to be readable by other CityJSON tools."""

    def test_exported_rings_are_not_closed(self):
        """
        The exporter used to append the first vertex again, giving every face
        a doubled vertex and a zero-length edge. Our own importer accepted it,
        so nothing in the suite noticed.
        """
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"parcel_id": "89"},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [80.9490, 26.8510], [80.9500, 26.8510],
                    [80.9500, 26.8520], [80.9490, 26.8520],
                    [80.9490, 26.8510],          # GeoJSON closure
                ]]},
            }],
        }
        doc = cityjson.to_cityjson(geojson)
        obj = next(iter(doc["CityObjects"].values()))
        ring = obj["geometry"][0]["boundaries"][0][0]
        self.assertEqual(len(ring), 4, "the GeoJSON closure must be dropped")
        self.assertNotEqual(ring[0], ring[-1])

    def test_a_round_trip_still_preserves_the_polygon(self):
        """Dropping the closure on the way out and adding it back on the way
        in must leave the geometry unchanged."""
        original = [[80.9490, 26.8510], [80.9500, 26.8510],
                    [80.9500, 26.8520], [80.9490, 26.8520],
                    [80.9490, 26.8510]]
        geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {"parcel_id": "89"},
                          "geometry": {"type": "Polygon",
                                       "coordinates": [original]}}],
        }
        back = cityjson.to_geojson(cityjson.to_cityjson(geojson))
        ring = back["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(len(ring), len(original))
        for got, want in zip(ring, original):
            self.assertAlmostEqual(got[0], want[0], places=6)
            self.assertAlmostEqual(got[1], want[1], places=6)


class CityJSONExportTests(unittest.TestCase):
    def test_export_and_reimport_preserve_parcel_attributes_and_heights(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"parcel_id": 7, "khasra_number": "8/2"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [80.9, 26.8, 110.0], [80.901, 26.8, 110.0],
                        [80.901, 26.801, 110.0], [80.9, 26.801, 110.0],
                    ]],
                },
            }],
        }
        exported = cityjson.to_cityjson(geojson)
        self.assertEqual(exported["version"], "2.0")
        self.assertEqual(exported["metadata"]["referenceSystem"], REFERENCE_SYSTEM)
        self.assertNotIn("z_is_placeholder", exported["CityObjects"]["parcel-7"]["attributes"])
        imported = cityjson.to_geojson(exported)
        feature = imported["features"][0]
        self.assertEqual(feature["properties"]["khasra_number"], "8/2")
        self.assertEqual(feature["geometry"]["coordinates"][0][0],
                         [80.9, 26.8, 110.0])

    def test_export_marks_missing_elevations_as_placeholders(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"parcel_id": 1},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [80.9, 26.8], [80.901, 26.8], [80.901, 26.801],
                ]]},
            }],
        }
        result = cityjson.to_cityjson(geojson)
        obj = result["CityObjects"]["parcel-1"]
        self.assertTrue(obj["attributes"]["z_is_placeholder"])
        self.assertTrue(all(vertex[2] == 0 for vertex in result["vertices"]))

    def test_rejects_projected_geojson_coordinates(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [500000, 2960000], [500001, 2960000], [500001, 2960001],
                ]]},
            }],
        }
        with self.assertRaisesRegex(cityjson.CityJSONError, "WGS84 bounds"):
            cityjson.to_cityjson(geojson)


if __name__ == "__main__":
    unittest.main()
