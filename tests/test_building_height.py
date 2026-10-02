#!/usr/bin/env python3
"""
Unit tests for measured building heights (backend/building_height.py) and the
declared-vs-measured check in backend/vertical.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

NOTHING HERE TOUCHES THE NETWORK.

The decoder is tested by ENCODING a tile the same way the dataset does -
float32, Deflate, TIFF predictor 3, byte-separated planes - and asserting it
round-trips. That is a real test of the hard part: the predictor's stride is
the one thing in this module that fails silently, because flat ground has zero
deltas and decodes correctly under the wrong stride.

The coordinate guards matter just as much. Most footprints in this system are
in MAP PIXELS, and sampling a raster with pixel coordinates would read a
confident height from the Gulf of Guinea.

Run from anywhere with:
    python3 tests/test_building_height.py -v
"""

from __future__ import annotations

import os
import struct
import sys
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import building_height as bh  # noqa: E402
import vertical as v          # noqa: E402

PARCEL = "UP091223700412"
SQUARE = [(80.9490, 26.8510), (80.9500, 26.8510),
          (80.9500, 26.8520), (80.9490, 26.8520)]


def encode_predictor_3(rows, tw):
    """
    Encode float rows exactly as the dataset's GeoTIFFs do.

    The inverse of _undo_float_predictor: shuffle each row's bytes into
    significance planes (most significant first), then horizontally difference
    over bytes with stride 1.
    """
    out = bytearray()
    for row in rows:
        packed = struct.pack("<" + "f" * tw, *row)
        planes = bytearray(tw * 4)
        for i in range(tw):
            planes[i] = packed[i * 4 + 3]
            planes[tw + i] = packed[i * 4 + 2]
            planes[2 * tw + i] = packed[i * 4 + 1]
            planes[3 * tw + i] = packed[i * 4 + 0]
        diffed = bytearray(planes)
        for i in range(len(planes) - 1, 0, -1):
            diffed[i] = (planes[i] - planes[i - 1]) & 0xFF
        out += diffed
    return bytes(out)


class TestFloatPredictor(unittest.TestCase):
    """The part that fails silently when it is wrong."""

    def test_a_flat_tile_round_trips(self):
        tw = th = 4
        rows = [tuple(0.0 for _ in range(tw)) for _ in range(th)]
        raw = encode_predictor_3(rows, tw)
        self.assertEqual(bh._undo_float_predictor(raw, tw, th), rows)

    def test_a_varying_tile_round_trips(self):
        """
        THE REGRESSION. Under the wrong predictor stride a flat tile still
        decodes perfectly - the deltas are all zero - so only varying data
        catches it. The first version produced -3.4e38 among real values here.
        """
        tw = th = 8
        rows = [tuple(float(r * 10 + c) + 0.25 for c in range(tw))
                for r in range(th)]
        raw = encode_predictor_3(rows, tw)
        decoded = bh._undo_float_predictor(raw, tw, th)
        self.assertEqual(decoded, rows)
        flat = [x for row in decoded for x in row]
        self.assertTrue(all(0 <= x < 1e6 for x in flat), min(flat))

    def test_plausible_building_heights_round_trip(self):
        tw = th = 8
        rows = [tuple((0.0 if (r + c) % 3 else 3.5 * ((r + c) % 7))
                      for c in range(tw)) for r in range(th)]
        raw = encode_predictor_3(rows, tw)
        for got, want in zip(bh._undo_float_predictor(raw, tw, th), rows):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=4)

    def test_it_survives_deflate(self):
        """The real tiles are zlib-compressed on top of the predictor."""
        tw = th = 8
        rows = [tuple(float(c) for c in range(tw)) for _ in range(th)]
        raw = zlib.decompress(zlib.compress(encode_predictor_3(rows, tw)))
        self.assertEqual(bh._undo_float_predictor(raw, tw, th), rows)


class TestProjection(unittest.TestCase):

    def test_uttar_pradesh_lands_in_zone_44(self):
        zone, e, n = bh.utm_forward(80.9495, 26.8515)
        self.assertEqual(zone, 44)
        self.assertTrue(400000 < e < 600000, e)     # near the central meridian
        self.assertTrue(2900000 < n < 3100000, n)

    def test_the_zone_boundary(self):
        self.assertEqual(bh.utm_zone(77.9), 43)
        self.assertEqual(bh.utm_zone(78.1), 44)
        self.assertEqual(bh.utm_zone(84.1), 45)

    def test_the_southern_hemisphere_is_refused_not_mis_projected(self):
        """
        The dataset covers Latin America, so a southern coordinate is a real
        possibility. Without the false northing it would be placed on another
        continent, confidently.
        """
        with self.assertRaises(bh.HeightError):
            bh.utm_forward(-46.6, -23.5)            # Sao Paulo

    def test_nonsense_coordinates_are_refused(self):
        for lon, lat in ((0, 89.0), (500, 20)):
            with self.assertRaises(bh.HeightError):
                bh.utm_forward(lon, lat)


class TestCoordinateGuards(unittest.TestCase):
    """
    sample_footprint must refuse anything that is not lon/lat degrees, because
    almost every footprint in this system is in map pixels.
    """

    def test_a_pixel_footprint_is_refused(self):
        """
        Pixel coordinates from a parcel map. (246, 20) is a valid lon/lat pair
        on its face and sits in the Atlantic off Africa - a height read there
        would be returned with no indication anything was wrong.
        """
        pixels = [(246, 20), (358, 103), (228, 333), (145, 268)]
        self.assertIsNone(bh.sample_footprint(pixels))

    def test_a_small_pixel_ring_inside_the_valid_range_is_still_refused(self):
        """Pixels that happen to look like degrees are caught by extent."""
        pixels = [(20, 30), (80, 30), (80, 85), (20, 85)]
        self.assertIsNone(bh.sample_footprint(pixels))

    def test_a_degenerate_ring_is_refused(self):
        self.assertIsNone(bh.sample_footprint([(80.9, 26.8), (80.91, 26.8)]))
        self.assertIsNone(bh.sample_footprint([]))

    def test_disabled_and_uncached_returns_none_not_an_error(self):
        """Every failure path returns None so the caller keeps its geometry."""
        enabled, cache = bh.ENABLED, bh.CACHE_DIR
        try:
            bh.ENABLED = False
            bh.CACHE_DIR = os.path.join(ROOT, "storage", "_no_such_dir_")
            self.assertIsNone(bh.sample_footprint(SQUARE))
        finally:
            bh.ENABLED, bh.CACHE_DIR = enabled, cache

    def test_it_announces_why_it_is_off(self):
        if not bh.available():
            self.assertIn("BUILDING_HEIGHT", bh.unavailable_reason())

    def test_the_licence_attribution_is_not_optional(self):
        """CC-BY is a condition, so it must travel with the capability line."""
        d = bh.describe()
        self.assertIn("CC-BY", d["attribution"])
        self.assertIn("Open Buildings", d["attribution"])
        self.assertIn("not a survey", d["caveat"])


class TestRepresentativeHeight(unittest.TestCase):

    def test_the_95th_percentile_is_used_not_the_maximum(self):
        """At 4 m resolution the tallest pixel may be a neighbour's roof."""
        reading = {"p95_m": 9.0, "max_m": 31.0, "built_fraction": 0.8}
        self.assertEqual(bh.representative_height(reading), 9.0)

    def test_an_unbuilt_footprint_yields_no_height(self):
        """The tallest pixel of a field is not a building."""
        reading = {"p95_m": 2.0, "max_m": 2.0, "built_fraction": 0.02}
        self.assertIsNone(bh.representative_height(reading))

    def test_no_reading_yields_no_height(self):
        self.assertIsNone(bh.representative_height(None))


class TestDeclaredAgainstMeasured(unittest.TestCase):

    def test_a_wildly_overstated_stack_is_reported(self):
        """Twelve floors declared on a 6 m building."""
        stack = v.stack(PARCEL, SQUARE, floors_above=11)      # top at 36 m
        issues = v.check_against_measured(stack, 6.0)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["rule"], "HEIGHT_CONTRADICTS_FLOORS")
        self.assertEqual(issues[0]["measured_could_hold_floors"], 2)

    def test_a_truthful_stack_is_not_reported(self):
        """Three levels, 9 m declared, 9 m measured."""
        stack = v.stack(PARCEL, SQUARE, floors_above=2)       # top at 9 m
        self.assertEqual(v.check_against_measured(stack, 9.0), [])

    def test_satellite_noise_does_not_trigger_it(self):
        """
        Tolerance is the dataset's 4 m resolution plus a storey. A metre of
        disagreement must not produce a finding, or the check gets ignored.
        """
        stack = v.stack(PARCEL, SQUARE, floors_above=2)       # 9 m
        for measured in (8.0, 7.5, 9.5, 12.0):
            self.assertEqual(v.check_against_measured(stack, measured), [],
                             f"measured {measured} should be within tolerance")

    def test_a_building_taller_than_declared_is_not_a_finding(self):
        """
        Declaring two floors of a six-storey block is an ordinary record. Only
        the overstatement is evidence of an error, so the check is one-directional.
        """
        stack = v.stack(PARCEL, SQUARE, floors_above=1)       # 6 m
        self.assertEqual(v.check_against_measured(stack, 30.0), [])

    def test_no_measurement_means_no_finding(self):
        stack = v.stack(PARCEL, SQUARE, floors_above=5)
        self.assertEqual(v.check_against_measured(stack, None), [])

    def test_basements_are_not_compared_against_a_rooftop_measurement(self):
        """
        A satellite sees the roof. Including basements in the declared top
        would compare an envelope against a depth.
        """
        stack = v.stack(PARCEL, SQUARE, floors_above=1, basements=4)
        self.assertEqual(v.check_against_measured(stack, 6.0), [])

    def test_the_finding_is_a_warning_not_an_error(self):
        """
        It rests on a 4 m satellite estimate, not a survey, so it asks a human
        to look rather than asserting the record is wrong.
        """
        stack = v.stack(PARCEL, SQUARE, floors_above=15)
        self.assertEqual(v.check_against_measured(stack, 5.0)[0]["severity"],
                         "warning")

    def test_the_message_names_the_measurement_as_an_estimate(self):
        stack = v.stack(PARCEL, SQUARE, floors_above=15)
        text = v.check_against_measured(stack, 5.0)[0]["suggestion"]
        self.assertIn("not a survey", text)


class TestFloorsFromHeight(unittest.TestCase):

    def test_it_floors_rather_than_rounds(self):
        """
        8.9 m at 3 m a storey is two floors. Rounding up would turn evidence
        that constrains a claim into evidence that invents a storey.
        """
        self.assertEqual(v.floors_from_height(8.9, 3.0), 2)
        self.assertEqual(v.floors_from_height(9.0, 3.0), 3)

    def test_a_flat_site_holds_nothing(self):
        self.assertEqual(v.floors_from_height(0.0, 3.0), 0)

    def test_a_negative_measurement_is_refused(self):
        with self.assertRaises(v.VerticalError):
            v.floors_from_height(-3.0, 3.0)

    def test_a_bad_storey_height_is_refused_here_too(self):
        with self.assertRaises(v.VerticalError):
            v.floors_from_height(10.0, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
