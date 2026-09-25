#!/usr/bin/env python3
"""
Unit tests for backend/cadastral.py (vectorization + georeferencing).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

OpenCV/numpy are optional dependencies for this project (see README S2), so
image-based tests are skipped (not failed) when they are absent. The affine
transform math has no such dependency beyond numpy and always runs when
numpy is present.

Run from anywhere with:
    python3 tests/test_cadastral.py -v
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import cadastral  # noqa: E402

DEMO_DIR = os.path.join(ROOT, "samples", "cadastral")
DEMO_MAP = os.path.join(DEMO_DIR, "village_map_narharpur.png")
DEMO_GROUND_TRUTH = os.path.join(DEMO_DIR, "ground_truth.json")
DEMO_CONTROL_POINTS = os.path.join(DEMO_DIR, "control_points.json")


def _tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return shutil.which("tesseract") is not None


class AffineTransformTests(unittest.TestCase):
    """Pure math - only needs numpy, not OpenCV or any image."""

    @unittest.skipUnless(cadastral.CV_AVAILABLE, "numpy not installed")
    def test_exact_fit_from_three_points(self):
        # A known transform: lon = 0.001*px + 10, lat = -0.001*py + 20
        control_points = [
            (0.0, 0.0, 10.0, 20.0),
            (100.0, 0.0, 10.1, 20.0),
            (0.0, 100.0, 10.0, 19.9),
        ]
        transform, residuals = cadastral.fit_affine_transform(control_points)
        for r in residuals:
            self.assertAlmostEqual(r, 0.0, places=9)
        lon, lat = transform.apply(50.0, 50.0)
        self.assertAlmostEqual(lon, 10.05, places=6)
        self.assertAlmostEqual(lat, 19.95, places=6)

    @unittest.skipUnless(cadastral.CV_AVAILABLE, "numpy not installed")
    def test_least_squares_fit_from_four_points_recovers_consistent_transform(self):
        control_points = [
            (0.0, 0.0, 80.9490, 26.8550),
            (900.0, 0.0, 80.9530, 26.8550),
            (0.0, 900.0, 80.9490, 26.8510),
            (900.0, 900.0, 80.9530, 26.8510),
        ]
        transform, residuals = cadastral.fit_affine_transform(control_points)
        self.assertLess(max(residuals), 1e-6)
        lon, lat = transform.apply(450.0, 450.0)
        self.assertAlmostEqual(lon, 80.9510, places=4)
        self.assertAlmostEqual(lat, 26.8530, places=4)

    @unittest.skipUnless(cadastral.CV_AVAILABLE, "numpy not installed")
    def test_fewer_than_three_points_raises(self):
        with self.assertRaises(ValueError):
            cadastral.fit_affine_transform([(0.0, 0.0, 10.0, 20.0), (1.0, 1.0, 10.1, 20.1)])

    @unittest.skipUnless(cadastral.CV_AVAILABLE, "numpy not installed")
    def test_georeference_parcels_fills_geo_polygon(self):
        transform, _ = cadastral.fit_affine_transform([
            (0.0, 0.0, 10.0, 20.0),
            (10.0, 0.0, 10.1, 20.0),
            (0.0, 10.0, 10.0, 19.9),
        ])
        parcels = [cadastral.Parcel(
            parcel_id=1, pixel_polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            pixel_centroid=(5.0, 5.0), area_px=100.0,
        )]
        cadastral.georeference_parcels(parcels, transform)
        self.assertIsNotNone(parcels[0].geo_polygon)
        self.assertEqual(len(parcels[0].geo_polygon), 4)


class GeoJSONTests(unittest.TestCase):
    def test_feature_collection_shape_and_disclaimer(self):
        p = cadastral.Parcel(
            parcel_id=1, pixel_polygon=[(0, 0), (1, 0), (1, 1)],
            pixel_centroid=(0.5, 0.5), area_px=10.0, khasra_number="99/1",
            geo_polygon=[(10.0, 20.0), (10.1, 20.0), (10.1, 20.1)],
        )
        fc = cadastral.parcels_to_geojson([p], disclaimer="test disclaimer")
        self.assertEqual(fc["type"], "FeatureCollection")
        self.assertEqual(fc["_disclaimer"], "test disclaimer")
        self.assertEqual(len(fc["features"]), 1)
        feat = fc["features"][0]
        self.assertEqual(feat["properties"]["khasra_number"], "99/1")
        ring = feat["geometry"]["coordinates"][0]
        self.assertEqual(ring[0], ring[-1])          # GeoJSON polygons must close

    def test_parcel_without_geo_polygon_is_skipped(self):
        p = cadastral.Parcel(parcel_id=1, pixel_polygon=[(0, 0)], pixel_centroid=(0, 0), area_px=1.0)
        fc = cadastral.parcels_to_geojson([p])
        self.assertEqual(fc["features"], [])


@unittest.skipUnless(cadastral.CV_AVAILABLE, "OpenCV/numpy not installed")
class MapFrameDetectionTests(unittest.TestCase):
    """
    detect_map_frame() exists because real Bhu-Naksha plot reports surround
    the map with a title and an owner/attribute table - without cropping to
    the map's own border, those get picked up as false parcels and (worse)
    skew the median-based area filter enough to wrongly reject real ones.
    """

    def test_no_dominant_contour_returns_none(self):
        cv2, np = cadastral._cv2, cadastral._numpy
        # Two same-sized squares, neither dominant - not a frame situation.
        img = np.full((200, 200), 255, dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (80, 80), 0, 2)
        cv2.rectangle(img, (120, 120), (190, 190), 0, 2)
        self.assertIsNone(cadastral.detect_map_frame(img))

    @unittest.skipUnless(os.path.exists(DEMO_MAP),
                         "bundled demo map missing - run tools/make_cadastral_map.py")
    def test_finds_the_explicit_border_on_the_demo_map(self):
        cv2 = cadastral._cv2
        img = cv2.imread(DEMO_MAP, cv2.IMREAD_GRAYSCALE)
        frame = cadastral.detect_map_frame(img)
        self.assertIsNotNone(frame)
        x, y, w, h = frame
        # The demo map's border sits a few px in from the 900x900 canvas
        # edge (tools/make_cadastral_map.py's border_margin=12) - not the
        # full canvas, and not the parcels' own tighter inner hull.
        self.assertLess(x, 20)
        self.assertLess(y, 20)
        self.assertGreater(w, 850)
        self.assertGreater(h, 850)


@unittest.skipUnless(cadastral.CV_AVAILABLE, "OpenCV/numpy not installed - vectorization skipped")
@unittest.skipUnless(os.path.exists(DEMO_MAP),
                     "bundled demo map missing - run tools/make_cadastral_map.py")
class VectorizationAgainstGroundTruthTests(unittest.TestCase):
    """Verified against the bundled synthetic demo map's known ground truth."""

    @classmethod
    def setUpClass(cls):
        with open(DEMO_GROUND_TRUTH, encoding="utf-8") as fh:
            cls.ground_truth = json.load(fh)["parcels"]
        cls.parcels = cadastral.vectorize(DEMO_MAP)

    def test_parcel_count_matches_ground_truth(self):
        self.assertEqual(len(self.parcels), len(self.ground_truth))

    def test_every_parcel_is_a_simple_polygon(self):
        for p in self.parcels:
            self.assertGreaterEqual(len(p.pixel_polygon), 3)

    def test_areas_match_ground_truth_within_tolerance(self):
        # Match each detected parcel to its nearest ground-truth centroid and
        # check the reported area is close - proves vectorization measured
        # the real parcel, not a fragment or a merged blob. Ground-truth area
        # is the jittered quad's own (shoelace) area, not its axis-aligned
        # bounding box - a skewed quad's true area is smaller than its bbox,
        # so bbox area would be the wrong thing to compare against. A 15%
        # tolerance (not tighter) is deliberate: the map draws boundaries as
        # a 3px line, so a detected contour traces the *interior* edge of
        # that line - measured area is always a little smaller than the
        # ideal quad, proportionally more so for smaller parcels. That is
        # real, expected inset, not noise.
        def shoelace(quad):
            n = len(quad)
            return abs(sum(quad[i][0] * quad[(i + 1) % n][1] - quad[(i + 1) % n][0] * quad[i][1]
                           for i in range(n))) / 2.0

        for p in self.parcels:
            best = min(self.ground_truth,
                      key=lambda g: math.hypot(g["pixel_centroid"][0] - p.pixel_centroid[0],
                                               g["pixel_centroid"][1] - p.pixel_centroid[1]))
            truth_area = shoelace(best["pixel_quad"])
            self.assertLess(abs(p.area_px - truth_area) / truth_area, 0.15)

    @unittest.skipUnless(_tesseract_available(), "tesseract not installed")
    def test_khasra_labels_read_correctly(self):
        parcels = cadastral.vectorize(DEMO_MAP)
        cadastral.read_parcel_labels(DEMO_MAP, parcels)
        correct = 0
        for p in parcels:
            best = min(self.ground_truth,
                      key=lambda g: math.hypot(g["pixel_centroid"][0] - p.pixel_centroid[0],
                                               g["pixel_centroid"][1] - p.pixel_centroid[1]))
            if p.khasra_number == best["khasra_number"]:
                correct += 1
        self.assertGreaterEqual(correct, int(len(parcels) * 0.9))

    def test_full_pipeline_produces_georeferenced_geojson(self):
        control_points = cadastral.load_control_points(DEMO_CONTROL_POINTS)
        geojson = cadastral.vectorize_and_georeference(
            DEMO_MAP, control_points, read_labels=False,
            disclaimer=cadastral.DEMO_DISCLAIMER,
        )
        self.assertEqual(len(geojson["features"]), len(self.ground_truth))
        self.assertEqual(geojson["_disclaimer"], cadastral.DEMO_DISCLAIMER)
        self.assertLess(geojson["_georeferencing"]["max_residual_deg"], 1e-6)
        # Every coordinate should land near the demo map's illustrative bounding
        # box (80.949-80.953 lon, 26.851-26.855 lat) - proves the transform was
        # actually applied, not left as raw pixel coordinates.
        for feat in geojson["features"]:
            for lon, lat in feat["geometry"]["coordinates"][0]:
                self.assertTrue(80.94 <= lon <= 80.96)
                self.assertTrue(26.84 <= lat <= 26.86)


class PolygonMeasurementTests(unittest.TestCase):
    """
    Real-world area and centroid of a georeferenced parcel - the numbers the
    document/map area cross-check (server._add_geo_issues) is built on. Pure
    geometry, so no OpenCV or Tesseract needed.
    """

    # A 0.001 deg box near Lucknow (lat 26.85). Expected sides:
    #   lat 0.001 * 110540           = 110.54 m
    #   lon 0.001 * 111320 * cos(26.85 deg) = 99.31 m
    # -> about 10,978 m2.
    BOX = [(80.950, 26.850), (80.951, 26.850), (80.951, 26.851), (80.950, 26.851)]

    def test_area_of_a_known_box(self):
        area = cadastral.polygon_area_m2(self.BOX)
        self.assertAlmostEqual(area, 10978, delta=120)

    def test_closed_and_open_rings_agree(self):
        """GeoJSON repeats the first point last; the raw polygon does not.
        Both must measure the same, or area depends on which one was passed."""
        closed = self.BOX + [self.BOX[0]]
        self.assertAlmostEqual(cadastral.polygon_area_m2(self.BOX),
                               cadastral.polygon_area_m2(closed), places=6)

    def test_winding_order_does_not_change_area(self):
        self.assertAlmostEqual(cadastral.polygon_area_m2(self.BOX),
                               cadastral.polygon_area_m2(list(reversed(self.BOX))),
                               places=6)

    def test_longitude_scale_shrinks_with_latitude(self):
        """The same degree box is physically smaller further north - if this
        fails, a single national metres-per-degree constant crept back in."""
        north = [(lon, lat + 8.0) for lon, lat in self.BOX]   # ~35N, Ladakh
        self.assertLess(cadastral.polygon_area_m2(north),
                        cadastral.polygon_area_m2(self.BOX))

    def test_degenerate_rings_measure_zero_rather_than_raising(self):
        self.assertEqual(cadastral.polygon_area_m2([]), 0.0)
        self.assertEqual(cadastral.polygon_area_m2([(80.95, 26.85)]), 0.0)
        self.assertEqual(cadastral.polygon_area_m2(
            [(80.95, 26.85), (80.96, 26.85)]), 0.0)

    def test_centroid_of_a_box_is_its_middle(self):
        lon, lat = cadastral.polygon_centroid(self.BOX)
        self.assertAlmostEqual(lon, 80.9505, places=6)
        self.assertAlmostEqual(lat, 26.8505, places=6)

    def test_centroid_of_empty_ring_is_none(self):
        self.assertIsNone(cadastral.polygon_centroid([]))

    def test_geojson_carries_measurements_for_the_cross_check(self):
        if not cadastral.CV_AVAILABLE or not os.path.exists(DEMO_MAP):
            self.skipTest("OpenCV or the demo map is unavailable")
        points = cadastral.load_control_points(DEMO_CONTROL_POINTS)
        geojson = cadastral.vectorize_and_georeference(
            DEMO_MAP, points, read_labels=False)
        self.assertTrue(geojson["features"])
        for feat in geojson["features"]:
            props = feat["properties"]
            self.assertIn("area_m2", props)
            self.assertIsNotNone(props["centroid_lat"])
            self.assertIsNotNone(props["centroid_lon"])
            # Agricultural parcels on this demo map, not stray specks or the
            # whole village: sanity-bound them rather than asserting exact values.
            self.assertGreater(props["area_m2"], 500)
            self.assertLess(props["area_m2"], 200000)



@unittest.skipUnless(cadastral.CV_AVAILABLE, "numpy not installed")
class RobustGeoreferencingTests(unittest.TestCase):
    """
    Outlier removal on the control points.

    fit_affine_transform already reported per-point residuals, but nothing
    acted on them: one mis-clicked GCP was silently least-squared across
    every parcel on the sheet. fit_transform_robust runs the loop a surveyor
    runs by hand in QGIS - fit, look at residuals, delete the obvious
    mis-click, re-fit.
    """

    # A clean, exactly-affine set: pixel -> lon/lat with a simple scale.
    CLEAN = [
        (0.0, 0.0, 77.0000, 23.0000),
        (100.0, 0.0, 77.0100, 23.0000),
        (100.0, 100.0, 77.0100, 22.9900),
        (0.0, 100.0, 77.0000, 22.9900),
        (50.0, 50.0, 77.0050, 22.9950),
    ]

    def _with_outlier(self):
        points = list(self.CLEAN)
        # Move one point a long way off - a mis-click on the reference layer.
        px, py, lon, lat = points[2]
        points[2] = (px, py, lon + 0.02, lat - 0.02)
        return points

    def test_a_clean_set_is_left_alone(self):
        _t, report = cadastral.fit_transform_robust(self.CLEAN)
        self.assertEqual(report["dropped"], [])
        self.assertTrue(report["passed"])
        self.assertEqual(report["control_point_count"], len(self.CLEAN))

    def test_rms_is_reported_in_metres_not_degrees(self):
        """
        Degrees are not a unit anyone can hold a tolerance in, and a 0.02
        degree error is over 2 km - which must not read as "0.02".
        """
        _t, report = cadastral.fit_transform_robust(self._with_outlier())
        self.assertIn("rms_metres", report)
        # The pre-drop figure is the one that shows the mis-click: 0.02
        # degrees is over 2 km, and must not read as "0.02".
        self.assertGreater(report["initial_rms_metres"], 100.0)
        # The post-drop figure is the fit actually being used.
        self.assertLess(report["rms_metres"], report["initial_rms_metres"])

    def test_the_worst_control_point_is_dropped(self):
        _t, report = cadastral.fit_transform_robust(self._with_outlier())
        self.assertTrue(report["dropped"])
        self.assertEqual(report["control_point_count"], len(self.CLEAN) - 1)

    def test_a_dropped_point_is_named_with_its_residual(self):
        """A removal a reviewer cannot inspect is a removal they cannot
        argue with."""
        _t, report = cadastral.fit_transform_robust(self._with_outlier())
        first = report["dropped"][0]
        self.assertIn("point", first)
        self.assertIn("residual_m", first)
        self.assertGreater(first["residual_m"], 0.0)

    def test_dropping_stops_before_redundancy_is_lost(self):
        """
        An affine fit has 6 parameters, so 3 points determine it exactly and
        always report zero residual. Dropping that far would manufacture a
        perfect-looking fit out of a bad one.
        """
        bad = [(0.0, 0.0, 77.0, 23.0), (100.0, 0.0, 77.5, 23.4),
               (100.0, 100.0, 76.2, 22.1), (0.0, 100.0, 78.9, 22.4),
               (50.0, 50.0, 70.0, 20.0)]
        _t, report = cadastral.fit_transform_robust(bad, max_rms_m=0.001)
        self.assertGreaterEqual(report["control_point_count"],
                                cadastral.MIN_GCP_FOR_REDUNDANCY)

    def test_an_unmeetable_tolerance_reports_failure_rather_than_pretending(self):
        bad = [(0.0, 0.0, 77.0, 23.0), (100.0, 0.0, 77.5, 23.4),
               (100.0, 100.0, 76.2, 22.1), (0.0, 100.0, 78.9, 22.4)]
        _t, report = cadastral.fit_transform_robust(bad, max_rms_m=0.001)
        self.assertFalse(report["passed"])

    def test_a_transform_is_still_returned_when_the_tolerance_is_missed(self):
        """A coarse georeferencing a reviewer has been warned about beats
        none at all - but `passed` must say so."""
        bad = [(0.0, 0.0, 77.0, 23.0), (100.0, 0.0, 77.5, 23.4),
               (100.0, 100.0, 76.2, 22.1), (0.0, 100.0, 78.9, 22.4)]
        transform, report = cadastral.fit_transform_robust(bad, max_rms_m=0.001)
        self.assertIsNotNone(transform)
        self.assertFalse(report["passed"])

    def test_a_three_point_fit_declares_it_has_no_redundancy(self):
        three = self.CLEAN[:3]
        _t, report = cadastral.fit_transform_robust(three)
        self.assertFalse(report["redundancy"])
        self.assertIn("not a measurement", report["note"])

    def test_a_redundant_fit_says_so(self):
        _t, report = cadastral.fit_transform_robust(self.CLEAN)
        self.assertTrue(report["redundancy"])

    def test_fewer_than_three_points_is_refused(self):
        with self.assertRaises(ValueError):
            cadastral.fit_transform_robust(self.CLEAN[:2])

    def test_rms_metres_of_no_residuals_is_zero(self):
        self.assertEqual(cadastral.rms_metres([]), 0.0)


if __name__ == "__main__":
    unittest.main()
