#!/usr/bin/env python3
"""
Unit tests for 3D ULPIN generation and vertical conflict detection
(backend/vertical.py).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

MOST OF THESE ARE ABOUT WHAT MUST *NOT* BE A CONFLICT.

A conflict detector that flags real buildings is worse than none, because a
reviewer learns to dismiss it. In an ordinary tower, every single storey
touches the one above it - a flat's ceiling IS the next flat's floor - and
every flat on a landing shares its height range with its neighbours. Both
look like overlaps to a careless test and neither is one.

So the cases that matter most here are: stacked flats do not conflict,
side-by-side flats do not conflict, and a basement does not conflict with the
shop above it. Only genuinely co-occupied space does.

Run from anywhere with:
    python3 tests/test_vertical.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import vertical as v  # noqa: E402

PARCEL = "UP091223700412"
SQUARE = [(80.9490, 26.8510), (80.9500, 26.8510),
          (80.9500, 26.8520), (80.9490, 26.8520)]
ELSEWHERE = [(80.9600, 26.8610), (80.9610, 26.8610),
             (80.9610, 26.8620), (80.9600, 26.8620)]


def unit_at(code, unit=1, ring=None, storey=v.DEFAULT_STOREY_M):
    base, top = v.level_elevation(code, storey)
    return v.VerticalParcel(
        ulpin_3d=v.make_3d_ulpin(PARCEL, code, unit), parcel_ulpin=PARCEL,
        level_code=code, unit=unit, footprint=list(ring or SQUARE),
        base_m=base, top_m=top)


class TestUlpinFormat(unittest.TestCase):

    def test_a_3d_ulpin_is_built_from_its_parcel(self):
        self.assertEqual(v.make_3d_ulpin(PARCEL, "F03", 12),
                         "UP091223700412-F03-012")

    def test_it_is_deterministic(self):
        """No allocator, no counter, no clock - a tehsil office with no
        network must produce the same identifier as the state server."""
        a = v.make_3d_ulpin(PARCEL, "F03", 12)
        b = v.make_3d_ulpin(PARCEL, "F03", 12)
        self.assertEqual(a, b)

    def test_it_is_reversible(self):
        parsed = v.parse_3d_ulpin("UP091223700412-F03-012")
        self.assertEqual(parsed["parcel_ulpin"], PARCEL)
        self.assertEqual(parsed["level_kind"], "floor")
        self.assertEqual(parsed["level_number"], 3)
        self.assertEqual(parsed["unit"], 12)

    def test_the_official_part_stays_extractable(self):
        """DILRMP specifies the 14-character parcel ULPIN and not the suffix,
        so the parent must survive unaltered and be recoverable."""
        full = v.make_3d_ulpin(PARCEL, "B02", 7)
        self.assertTrue(full.startswith(PARCEL + "-"))
        self.assertEqual(v.parse_3d_ulpin(full)["parcel_ulpin"], PARCEL)

    def test_the_letters_do_not_sort_physically_and_we_do_not_pretend_they_do(self):
        """
        F sorts before G in ASCII, so a plain sort puts the third floor below
        the ground floor. The letters are mnemonic because that is what gets
        read off a document; the physical order comes from level_sort_key.
        """
        codes = ["B01", "G00", "F01", "F10"]
        self.assertNotEqual(sorted(codes), ["B01", "G00", "F01", "F10"])
        self.assertEqual(sorted(codes, key=v.level_sort_key),
                         ["B01", "G00", "F01", "F10"])

    def test_deeper_basements_sort_lower(self):
        """B02 is BELOW B01: the number counts downward."""
        self.assertEqual(sorted(["B01", "B03", "B02"], key=v.level_sort_key),
                         ["B03", "B02", "B01"])

    def test_a_whole_stack_sorts_bottom_to_top(self):
        codes = ["F02", "B01", "A01", "G00", "S01", "F01", "B02"]
        self.assertEqual(sorted(codes, key=v.level_sort_key),
                         ["S01", "B02", "B01", "G00", "F01", "F02", "A01"])

    def test_ground_has_exactly_one_spelling(self):
        """Two spellings of one level is how the same volume gets registered
        twice."""
        self.assertEqual(v.level_code("G"), "G00")
        self.assertEqual(v.level_code("G", 5), "G00")

    def test_a_malformed_parent_is_refused_not_repaired(self):
        for bad in ("too-short", "", "up091223700412x", None, "UP09122370041"):
            with self.assertRaises(v.VerticalError):
                v.make_3d_ulpin(bad, "F01", 1)

    def test_out_of_range_values_are_refused(self):
        with self.assertRaises(v.VerticalError):
            v.make_3d_ulpin(PARCEL, "F01", 0)
        with self.assertRaises(v.VerticalError):
            v.make_3d_ulpin(PARCEL, "F01", 1000)
        with self.assertRaises(v.VerticalError):
            v.level_code("F", 100)
        with self.assertRaises(v.VerticalError):
            v.level_code("X", 1)

    def test_garbage_does_not_parse(self):
        for bad in ("", "UP091223700412", "UP091223700412-F03",
                    "UP091223700412-Z03-001", "nonsense"):
            with self.assertRaises(v.VerticalError):
                v.parse_3d_ulpin(bad)


class TestElevation(unittest.TestCase):

    def test_ground_is_the_datum(self):
        self.assertEqual(v.level_elevation("G00"), (0.0, 3.0))

    def test_floors_stack_upward(self):
        self.assertEqual(v.level_elevation("F01"), (3.0, 6.0))
        self.assertEqual(v.level_elevation("F02"), (6.0, 9.0))

    def test_basements_go_negative(self):
        """A parking level at -3 m and a shop at 0 m share a footprint; if
        basements did not go negative they would collide."""
        self.assertEqual(v.level_elevation("B01"), (-3.0, 0.0))
        self.assertEqual(v.level_elevation("B02"), (-6.0, -3.0))

    def test_storeys_meet_exactly(self):
        _, g_top = v.level_elevation("G00")
        f1_base, _ = v.level_elevation("F01")
        self.assertEqual(g_top, f1_base)


class TestNotConflicts(unittest.TestCase):
    """The cases a careless detector gets wrong."""

    def test_stacked_flats_are_not_a_conflict(self):
        """One above another, same footprint. This is a building."""
        stack = [unit_at("G00"), unit_at("F01"), unit_at("F02"), unit_at("F03")]
        self.assertEqual(v.find_conflicts(stack), [])

    def test_touching_floors_are_not_an_overlap(self):
        """A flat's ceiling IS the next flat's floor, so top == base on every
        storey of every building."""
        self.assertEqual(v.z_overlap(unit_at("G00"), unit_at("F01")), 0.0)

    def test_flats_on_one_landing_are_not_a_conflict(self):
        """Same height range, different ground."""
        a = unit_at("F02", 1, SQUARE)
        b = unit_at("F02", 2, ELSEWHERE)
        self.assertEqual(v.find_conflicts([a, b]), [])

    def test_a_basement_does_not_conflict_with_the_shop_above_it(self):
        self.assertEqual(v.find_conflicts([unit_at("B01"), unit_at("G00")]), [])

    def test_a_whole_declared_tower_is_clean(self):
        parcels = v.stack(PARCEL, SQUARE, floors_above=12, basements=2,
                          units_per_level=1)
        self.assertEqual(v.find_conflicts(parcels), [])


class TestConflicts(unittest.TestCase):

    def test_two_units_in_the_same_space_conflict(self):
        a = unit_at("F02", 1)
        b = v.VerticalParcel(
            ulpin_3d=v.make_3d_ulpin(PARCEL, "F02", 2), parcel_ulpin=PARCEL,
            level_code="F02", unit=2, footprint=list(SQUARE),
            base_m=a.base_m, top_m=a.top_m)       # same ground, same height
        issues = v.find_conflicts([a, b])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["rule"], "VERTICAL_OVERLAP")
        self.assertEqual(issues[0]["overlap_m"], 3.0)

    def test_a_partial_height_overlap_is_caught(self):
        """A mezzanine sold into the storey above it."""
        a = unit_at("F01")                                   # 3.0 - 6.0
        b = v.VerticalParcel(
            ulpin_3d=v.make_3d_ulpin(PARCEL, "F02", 1), parcel_ulpin=PARCEL,
            level_code="F02", unit=1, footprint=list(SQUARE),
            base_m=4.5, top_m=7.5)                           # overlaps by 1.5
        issues = v.find_conflicts([a, b])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["overlap_m"], 1.5)

    def test_a_repeated_identifier_is_an_error_of_its_own(self):
        a = unit_at("F01", 1)
        b = unit_at("F01", 1, ELSEWHERE)      # different ground, same ULPIN
        issues = v.find_conflicts([a, b])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["rule"], "DUPLICATE_3D_ULPIN")


class TestInvertedGeometry(unittest.TestCase):
    """
    A volume whose top sits below its base encloses nothing.

    This is the nastiest shape of bad data in the module, because an inverted
    range shares no height with ANY real range - so it does not merely pass
    unnoticed, it SUPPRESSES the overlap it genuinely has. Bad input produced a
    building that was structurally incapable of colliding with anything.
    """

    def test_a_negative_storey_height_is_refused(self):
        """It arrives from an HTTP body, so this is reachable, not theoretical."""
        with self.assertRaises(v.VerticalError):
            v.stack(PARCEL, SQUARE, floors_above=2, storey_m=-3.0)

    def test_a_zero_storey_height_is_refused(self):
        with self.assertRaises(v.VerticalError):
            v.stack(PARCEL, SQUARE, floors_above=2, storey_m=0)

    def test_a_non_numeric_or_infinite_storey_is_refused(self):
        for bad in ("tall", None, float("inf"), float("nan")):
            with self.assertRaises(v.VerticalError):
                v.level_elevation("F01", bad)

    def test_an_absurd_storey_height_is_refused(self):
        """Catches centimetres or feet entered where metres were meant."""
        with self.assertRaises(v.VerticalError):
            v.level_elevation("F01", 300.0)

    def test_a_stored_inverted_volume_is_reported_not_crashed_on(self):
        """
        Rows written by an older version must be REPORTED, not refused on load.
        Raising would turn one corrupt row into a dead API endpoint.
        """
        bad = v.VerticalParcel(
            ulpin_3d=v.make_3d_ulpin(PARCEL, "F01", 1), parcel_ulpin=PARCEL,
            level_code="F01", unit=1, footprint=list(SQUARE),
            base_m=6.0, top_m=3.0)
        self.assertFalse(bad.has_valid_range)
        issues = v.find_conflicts([bad])
        self.assertEqual([i["rule"] for i in issues], ["INVALID_Z_RANGE"])
        self.assertEqual(issues[0]["severity"], "error")

    def test_an_inverted_volume_cannot_hide_a_real_overlap(self):
        """
        THE POINT OF ALL THIS. Before the check, these two returned no findings
        whatsoever: the inverted one shares 0 m of height with the good one, so
        the pair looked clean while both claimed the same cubic metres.
        """
        bad = v.VerticalParcel(
            ulpin_3d=v.make_3d_ulpin(PARCEL, "F01", 1), parcel_ulpin=PARCEL,
            level_code="F01", unit=1, footprint=list(SQUARE),
            base_m=6.0, top_m=3.0)
        good = v.VerticalParcel(
            ulpin_3d=v.make_3d_ulpin(PARCEL, "F01", 2), parcel_ulpin=PARCEL,
            level_code="F01", unit=2, footprint=list(SQUARE),
            base_m=3.0, top_m=6.0)
        self.assertEqual(v.z_overlap(bad, good), 0.0)      # silently, as before
        rules = [i["rule"] for i in v.find_conflicts([bad, good])]
        self.assertIn("INVALID_Z_RANGE", rules)            # but no longer silent

    def test_a_generated_stack_never_contains_one(self):
        for storey in (2.5, 3.0, 4.2):
            for p in v.stack(PARCEL, SQUARE, floors_above=4, basements=2,
                             storey_m=storey):
                self.assertTrue(p.has_valid_range, p.ulpin_3d)
                self.assertGreater(p.height_m, 0)


class TestDeclaredGeometry(unittest.TestCase):

    def test_a_stack_has_one_unit_per_level(self):
        parcels = v.stack(PARCEL, SQUARE, floors_above=3, basements=1)
        self.assertEqual(len(parcels), 5)            # B1, G, F1, F2, F3
        self.assertEqual(len({p.ulpin_3d for p in parcels}), 5)

    def test_multiple_units_per_level(self):
        parcels = v.stack(PARCEL, SQUARE, floors_above=1, units_per_level=4)
        self.assertEqual(len(parcels), 8)            # 2 levels x 4 units

    def test_a_block_of_flats_is_not_reported_as_overlapping(self):
        """
        REGRESSION. Four flats on a landing all inherit the whole parcel as
        their footprint, so they share ground AND height range and the first
        version of the detector called all six pairs an overlap - a building
        of 3 flats per floor produced 6 errors before this.
        """
        parcels = v.stack(PARCEL, SQUARE, floors_above=1, units_per_level=4)
        overlaps = [c for c in v.find_conflicts(parcels)
                    if c["rule"] == "VERTICAL_OVERLAP"]
        self.assertEqual(overlaps, [])

    def test_but_an_undivided_level_is_not_declared_clean_either(self):
        """
        The honest third answer. Nobody verified that those four flats are
        separate - the document carries no floor plan - so the level is
        reported as unpartitioned rather than silently passed.
        """
        parcels = v.stack(PARCEL, SQUARE, floors_above=1, units_per_level=4)
        gaps = [c for c in v.find_conflicts(parcels)
                if c["rule"] == "LEVEL_NOT_PARTITIONED"]
        self.assertEqual(len(gaps), 2)               # ground and first floor
        self.assertTrue(all(g["severity"] == "info" for g in gaps))
        self.assertEqual(len(gaps[0]["ulpins"]), 4)

    def test_the_gap_is_reported_once_per_level_not_once_per_pair(self):
        """Ten flats make 45 pairs; 45 copies of one sentence is noise."""
        parcels = v.stack(PARCEL, SQUARE, floors_above=0, units_per_level=10)
        gaps = [c for c in v.find_conflicts(parcels)
                if c["rule"] == "LEVEL_NOT_PARTITIONED"]
        self.assertEqual(len(gaps), 1)

    def test_one_unit_holding_a_whole_level_is_not_a_gap(self):
        """A single owner of the entire floor leaves nothing undetermined."""
        parcels = v.stack(PARCEL, SQUARE, floors_above=3, units_per_level=1)
        self.assertEqual(v.find_conflicts(parcels), [])

    def test_a_real_double_allocation_is_still_caught(self):
        """
        The flag must not become a blanket excuse. Two units with MEASURED
        footprints in the same space are a genuine conflict, and stay one.
        """
        a = unit_at("F02", 1)                        # footprint_is_parcel False
        b = v.VerticalParcel(
            ulpin_3d=v.make_3d_ulpin(PARCEL, "F02", 2), parcel_ulpin=PARCEL,
            level_code="F02", unit=2, footprint=list(SQUARE),
            base_m=a.base_m, top_m=a.top_m)
        issues = v.find_conflicts([a, b])
        self.assertEqual([i["rule"] for i in issues], ["VERTICAL_OVERLAP"])

    def test_a_measured_unit_overlapping_a_declared_one_is_still_caught(self):
        """Only a pair where BOTH sides are unmeasured is unknowable."""
        declared = v.stack(PARCEL, SQUARE, floors_above=2)[1]   # F01, inherited
        measured = v.VerticalParcel(
            ulpin_3d=v.make_3d_ulpin(PARCEL, "F01", 9), parcel_ulpin=PARCEL,
            level_code="F01", unit=9, footprint=list(SQUARE),
            base_m=declared.base_m, top_m=declared.top_m, surveyed=True)
        rules = [i["rule"] for i in v.find_conflicts([declared, measured])]
        self.assertIn("VERTICAL_OVERLAP", rules)

    def test_the_flag_says_the_footprint_was_inherited(self):
        parcel = v.stack(PARCEL, SQUARE, floors_above=1)[0]
        self.assertTrue(parcel.footprint_is_parcel)
        self.assertTrue(parcel.to_dict()["footprint_is_parcel"])

    def test_declared_volumes_say_they_are_declared(self):
        """The one thing that must never be lost: this is not a measurement."""
        parcel = v.stack(PARCEL, SQUARE, floors_above=1)[0]
        self.assertFalse(parcel.surveyed)
        self.assertTrue(any("not surveyed" in n for n in parcel.to_dict()["notes"]))

    def test_a_footprint_too_small_to_be_a_polygon_is_refused(self):
        with self.assertRaises(v.VerticalError):
            v.stack(PARCEL, [(1.0, 2.0), (1.0, 3.0)], floors_above=1)

    def test_height_is_derived_not_stored_twice(self):
        parcel = unit_at("F01")
        self.assertEqual(parcel.height_m, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
