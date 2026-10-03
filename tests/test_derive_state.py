#!/usr/bin/env python3
"""
Unit tests for deriving the state from the district (backend/server.py).
Intelligent Land Record Digitization and Validation System - SIH 2026.

WHY THE FIELD IS EMPTY IN THE FIRST PLACE

A land record almost never prints its state. It is obvious to everyone in the
room - the tehsil office is in the state - so it goes unwritten, and an
extractor that only reads what is on the page will never fill it. Measured on
the real corpus: district reaches 5 of 20 documents and state 0, while every
one of those districts determines its state exactly.

It matters more than it looks: the state is the top of the administrative
hierarchy and the first component of a ULPIN.

MOST OF THESE TESTS ARE ABOUT NOT DOING IT.

A derived value that quietly overwrites a read one, or that invents a state
for a district nobody recognises, is worse than a blank field - a blank is
visibly missing, whereas a wrong state looks like a fact.

Run from anywhere with:
    python3 tests/test_derive_state.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import field_extractor as fe   # noqa: E402
import server                  # noqa: E402
import validator               # noqa: E402


def fields(**values):
    return [fe.ExtractedField(key=spec.key, display=spec.display,
                              value=values.get(spec.key), confidence=0.9)
            for spec in fe.FIELD_SPECS]


def state_of(field_list):
    return [f for f in field_list if f.key == "state"][0]


class TestDeriving(unittest.TestCase):

    def test_a_known_district_fills_the_state(self):
        f = fields(district="Jaunpur")
        self.assertIsNotNone(server._derive_state(f))
        self.assertEqual(state_of(f).value, "Uttar Pradesh")

    def test_it_is_case_insensitive(self):
        for spelling in ("jaunpur", "JAUNPUR", "Jaunpur"):
            f = fields(district=spelling)
            server._derive_state(f)
            self.assertEqual(state_of(f).value, "Uttar Pradesh", spelling)

    def test_the_lgd_codes_come_with_it(self):
        """
        The state and district LGD codes are the first components of a ULPIN,
        so they are carried rather than thrown away once the name is known.
        """
        f = fields(district="Lucknow")
        resolved = server._derive_state(f)
        self.assertEqual(resolved["state_lgd"], "09")
        self.assertEqual(resolved["district_lgd"], "162")
        self.assertEqual(state_of(f).extra["state_lgd"], "09")

    def test_a_district_in_another_state_resolves_to_that_state(self):
        f = fields(district="Pune")
        server._derive_state(f)
        self.assertEqual(state_of(f).value, "Maharashtra")


class TestNotDeriving(unittest.TestCase):
    """The cases where staying quiet is the right answer."""

    def test_a_printed_state_is_never_overwritten(self):
        """
        The document is the authority. If it says Bihar while the district
        says Uttar Pradesh, that disagreement is for the hierarchy rule to
        report - silently 'correcting' it would destroy the evidence of it.
        """
        f = fields(district="Jaunpur", state="Bihar")
        self.assertIsNone(server._derive_state(f))
        self.assertEqual(state_of(f).value, "Bihar")

    def test_an_unknown_district_invents_nothing(self):
        f = fields(district="Atlantis")
        self.assertIsNone(server._derive_state(f))
        self.assertIsNone(state_of(f).value)

    def test_no_district_means_no_state(self):
        f = fields()
        self.assertIsNone(server._derive_state(f))
        self.assertIsNone(state_of(f).value)

    def test_a_blank_district_invents_nothing(self):
        for blank in ("", "   "):
            f = fields(district=blank)
            self.assertIsNone(server._derive_state(f))
            self.assertIsNone(state_of(f).value)


class TestProvenance(unittest.TestCase):
    """
    A reviewer looking at a state they cannot find anywhere on the page needs
    to know why it is there.
    """

    def test_the_value_is_marked_as_derived(self):
        f = fields(district="Jaunpur")
        server._derive_state(f)
        self.assertTrue(state_of(f).extra["derived"])
        self.assertEqual(state_of(f).extra["derived_from"], "district")
        self.assertEqual(state_of(f).extra["derived_from_value"], "Jaunpur")

    def test_the_note_says_it_was_not_printed(self):
        f = fields(district="Jaunpur")
        server._derive_state(f)
        note = " ".join(state_of(f).notes)
        self.assertIn("Jaunpur", note)
        self.assertIn("Not printed", note)

    def test_confidence_stays_below_a_read_value(self):
        """
        An inference from a closed table is strong, and still not something
        anybody wrote down. A district read at 0.99 must not yield a state
        that outranks fields actually present on the page.
        """
        f = fields(district="Jaunpur")
        for field in f:
            if field.key == "district":
                field.confidence = 0.99
        server._derive_state(f)
        self.assertLessEqual(state_of(f).confidence, 0.80)


class TestTheMasterLookup(unittest.TestCase):

    def test_the_master_is_loaded(self):
        self.assertTrue(validator._MASTER.loaded)

    def test_a_known_district_returns_its_state_and_codes(self):
        self.assertEqual(
            validator._MASTER.state_of_district("Lucknow")["state"],
            "Uttar Pradesh")

    def test_an_unknown_district_returns_none(self):
        self.assertIsNone(validator._MASTER.state_of_district("Nowhere"))
        self.assertIsNone(validator._MASTER.state_of_district(""))

    def test_the_master_is_a_demo_extract_and_the_tests_know_it(self):
        """
        8 states and 108 districts, not all of India. Anand - where this
        project's drone sample sits - is absent, and a test that assumed
        full coverage would fail on the day the extract is replaced.
        """
        self.assertGreaterEqual(len(validator._MASTER.states), 8)
        self.assertIsNone(validator._MASTER.state_of_district("Anand"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
