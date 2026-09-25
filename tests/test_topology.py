#!/usr/bin/env python3
"""
Unit tests for backend/topology.py (cadastral topology validation).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Pure geometry, no optional dependencies, so every test here always runs.

The tests are weighted towards FALSE POSITIVES rather than false negatives,
because that is where a topology checker actually fails in practice. Adjacent
parcels legitimately share entire boundary edges and an enclosing block
boundary legitimately contains every parcel on the sheet; a checker that
reports those as conflicts produces hundreds of findings on a correct village
map, and a reviewer who has been shown hundreds of false findings stops
reading the report. A missed sliver is a bug. A checker nobody reads is worse.

Run from anywhere with:
    python3 tests/test_topology.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import topology as tp  # noqa: E402

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
EAST = [(10, 0), (20, 0), (20, 10), (10, 10)]          # shares the x=10 edge
OVERLAP = [(5, 5), (15, 5), (15, 15), (5, 15)]         # genuinely overlaps SQUARE
ENVELOPE = [(-5, -5), (25, -5), (25, 25), (-5, 25)]    # contains SQUARE
BOWTIE = [(0, 0), (10, 10), (10, 0), (0, 10)]
FAR = [(100, 100), (110, 100), (110, 110), (100, 110)]


class AdjacencyTests(unittest.TestCase):
    """The false positives that would make the checker useless."""

    def test_parcels_sharing_a_whole_edge_are_clean(self):
        """
        The single most important case. Every interior boundary on a village
        map is shared by two parcels, so a checker that flags a shared edge
        reports every correct sheet as broken.
        """
        report = tp.validate({"a": SQUARE, "b": EAST})
        self.assertTrue(report["clean"], report)
        self.assertEqual(report["overlaps"], [])

    def test_a_vertex_lying_on_a_neighbours_edge_is_boundary_not_inside(self):
        self.assertEqual(tp.point_in_ring((10, 5), SQUARE), "boundary")
        self.assertEqual(tp.point_in_ring((5, 5), SQUARE), "inside")
        self.assertEqual(tp.point_in_ring((50, 50), SQUARE), "outside")

    def test_segments_touching_at_a_shared_endpoint_do_not_cross(self):
        self.assertFalse(tp.segments_properly_cross(
            (0, 0), (10, 0), (10, 0), (10, 10)))

    def test_segments_crossing_in_their_interiors_do_cross(self):
        self.assertTrue(tp.segments_properly_cross(
            (0, 0), (10, 10), (0, 10), (10, 0)))

    def test_distant_parcels_are_not_compared(self):
        report = tp.validate({"a": SQUARE, "b": FAR})
        self.assertTrue(report["clean"])


class OverlapTests(unittest.TestCase):

    def test_a_genuine_overlap_is_found(self):
        report = tp.validate({"a": SQUARE, "b": OVERLAP})
        self.assertEqual(len(report["overlaps"]), 1)
        self.assertFalse(report["clean"])

    def test_the_overlap_names_both_parcels_and_a_location(self):
        """A reviewer needs a coordinate, not "these two conflict"."""
        found = tp.find_overlaps({"237/4": SQUARE, "237/5": OVERLAP})[0]
        self.assertEqual({found["a"], found["b"]}, {"237/4", "237/5"})
        self.assertIn("at", found["witness"])
        self.assertIn(found["witness"]["kind"],
                      ("vertex_inside", "edges_cross"))

    def test_overlap_is_symmetric_and_reported_once(self):
        self.assertEqual(len(tp.find_overlaps({"a": SQUARE, "b": OVERLAP})), 1)
        self.assertEqual(len(tp.find_overlaps({"b": OVERLAP, "a": SQUARE})), 1)


class ContainmentTests(unittest.TestCase):
    """
    Containment is a different relation from overlap.

    Measured on the bundled Bhu-Naksha plot report: the vectoriser also picks
    up the outer envelope of the sheet, and before containment was separated
    out, that one shape was reported as overlapping all six parcels inside
    it - 6 findings, none of them real.
    """

    def test_an_enclosing_boundary_is_not_an_overlap(self):
        report = tp.validate({"block": ENVELOPE, "parcel": SQUARE})
        self.assertEqual(report["overlaps"], [])

    def test_containment_does_not_make_the_sheet_unclean(self):
        report = tp.validate({"block": ENVELOPE, "parcel": SQUARE})
        self.assertTrue(report["clean"], report)

    def test_containment_is_reported_on_its_own_channel(self):
        report = tp.validate({"block": ENVELOPE, "parcel": SQUARE})
        self.assertEqual(report["containments"],
                         [{"outer": "block", "inner": "parcel"}])

    def test_containment_is_directional(self):
        self.assertTrue(tp.contains(ENVELOPE, SQUARE))
        self.assertFalse(tp.contains(SQUARE, ENVELOPE))

    def test_a_partial_intrusion_is_not_containment(self):
        self.assertFalse(tp.contains(SQUARE, OVERLAP))
        self.assertFalse(tp.contains(OVERLAP, SQUARE))


class ValidityTests(unittest.TestCase):

    def test_a_clean_square_has_no_problems(self):
        self.assertEqual(tp.polygon_problems(SQUARE), [])

    def test_a_closing_vertex_is_not_a_duplicate(self):
        """GeoJSON rings repeat the first vertex last; that is the format,
        not a fault."""
        self.assertEqual(tp.polygon_problems(SQUARE + [(0, 0)]), [])

    def test_self_intersection_is_found(self):
        problems = tp.polygon_problems(BOWTIE)
        self.assertTrue(any("self-intersection" in p for p in problems), problems)

    def test_a_degenerate_ring_is_found(self):
        self.assertIn("fewer than 3 distinct vertices",
                      tp.polygon_problems([(0, 0), (1, 1)]))

    def test_a_collinear_ring_has_zero_area(self):
        self.assertIn("zero area",
                      tp.polygon_problems([(0, 0), (5, 0), (10, 0)]))

    def test_area_ignores_winding_direction(self):
        self.assertAlmostEqual(tp.area(SQUARE), 100.0)
        self.assertAlmostEqual(tp.area(list(reversed(SQUARE))), 100.0)
        self.assertLess(tp.signed_area(list(reversed(SQUARE))), 0)


class SnapTests(unittest.TestCase):
    """Sliver gaps: two traces of the same boundary landing a fraction apart."""

    GAP = [(10.3, 0), (20, 0), (20, 10), (10.3, 10)]

    def test_a_near_miss_boundary_is_reported(self):
        found = tp.find_unsnapped_vertices({"a": SQUARE, "b": self.GAP},
                                           tolerance=0.5)
        self.assertEqual(len(found), 2)
        self.assertTrue(all(0 < f["distance"] <= 0.5 for f in found))

    def test_a_gap_wider_than_the_tolerance_is_left_alone(self):
        """Beyond the tolerance it is a deliberate strip - a road or a
        watercourse - not a digitising slip."""
        self.assertEqual(
            tp.find_unsnapped_vertices({"a": SQUARE, "b": self.GAP},
                                       tolerance=0.1), [])

    def test_exactly_shared_vertices_are_not_reported(self):
        self.assertEqual(
            tp.find_unsnapped_vertices({"a": SQUARE, "b": EAST},
                                       tolerance=0.5), [])

    def test_snapping_moves_vertices_onto_a_common_point(self):
        rings = {"a": list(SQUARE), "b": list(self.GAP)}
        moved = tp.snap_shared_vertices(rings, 0.5)
        self.assertGreater(moved, 0)
        self.assertEqual(
            tp.find_unsnapped_vertices(rings, 0.5), [])

    def test_snapping_is_idempotent(self):
        """
        Running it twice must not creep the boundary further each time, which
        is why the representative is the first vertex seen rather than a
        moving cluster centroid.
        """
        rings = {"a": list(SQUARE), "b": list(self.GAP)}
        first = tp.snap_shared_vertices(rings, 0.5)
        second = tp.snap_shared_vertices(rings, 0.5)
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)

    def test_a_zero_tolerance_snaps_nothing(self):
        rings = {"a": list(SQUARE), "b": list(self.GAP)}
        self.assertEqual(tp.snap_shared_vertices(rings, 0.0), 0)


class ToleranceUnitTests(unittest.TestCase):
    """
    The unit trap. 0.5 read as degrees is a 55 km tolerance, which would snap
    every parcel in a district onto one point - and it would do it silently.
    """

    def test_degrees_and_metres_are_not_interchangeable(self):
        deg = tp.suggested_tolerance(in_degrees=True, metres=0.5)
        px = tp.suggested_tolerance(in_degrees=False, metres=0.5)
        self.assertAlmostEqual(px, 0.5)
        self.assertLess(deg, 1e-5)
        self.assertGreater(deg, 0)

    def test_half_a_metre_in_degrees_is_not_half_a_degree(self):
        self.assertNotAlmostEqual(
            tp.suggested_tolerance(in_degrees=True, metres=0.5), 0.5)


class ParcelBridgeTests(unittest.TestCase):

    class _P:
        def __init__(self, pid, px, geo=None, khasra=None):
            self.parcel_id = pid
            self.pixel_polygon = px
            self.geo_polygon = geo
            self.khasra_number = khasra

    def test_geographic_rings_are_preferred_when_present(self):
        p = self._P(1, SQUARE, geo=[(77.0, 23.0), (77.1, 23.0), (77.1, 23.1)])
        rings = tp.rings_from_parcels([p])
        self.assertEqual(rings["1"][0], (77.0, 23.0))

    def test_pixel_rings_are_used_when_there_is_no_georeferencing(self):
        rings = tp.rings_from_parcels([self._P(1, SQUARE)], prefer_geo=True)
        self.assertEqual(rings["1"][0], (0, 0))

    def test_the_khasra_number_keys_the_report_when_known(self):
        rings = tp.rings_from_parcels([self._P(1, SQUARE, khasra="237/4")])
        self.assertIn("237/4", rings)

    def test_a_parcel_with_no_geometry_is_skipped(self):
        self.assertEqual(tp.rings_from_parcels([self._P(1, None)]), {})


class ReportTests(unittest.TestCase):

    def test_the_report_states_what_it_does_not_check(self):
        """
        The caveat is load-bearing. Overlaps are detected but not measured,
        and gaps are inferred from unsnapped vertices rather than a true
        union, so an interior unmapped hole would not be found. A reader who
        assumed otherwise would trust the clean flag too far.
        """
        report = tp.validate({"a": SQUARE})
        self.assertIn("caveat", report)
        self.assertIn("not measured", report["caveat"])
        self.assertIn("would not be reported", report["caveat"])

    def test_an_empty_sheet_is_clean(self):
        report = tp.validate({})
        self.assertTrue(report["clean"])
        self.assertEqual(report["parcel_count"], 0)

    def test_the_report_counts_parcels_and_echoes_the_tolerance(self):
        report = tp.validate({"a": SQUARE, "b": EAST}, tolerance=0.25)
        self.assertEqual(report["parcel_count"], 2)
        self.assertEqual(report["tolerance"], 0.25)

    def test_an_invalid_ring_is_named_with_its_fault(self):
        report = tp.validate({"bad": BOWTIE})
        self.assertIn("bad", report["invalid"])
        self.assertFalse(report["clean"])


if __name__ == "__main__":
    unittest.main()
