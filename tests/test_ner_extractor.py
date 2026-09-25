#!/usr/bin/env python3
"""
Unit tests for backend/ner_extractor.py (ML-based NER cross-check).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

spaCy (and its en_core_web_sm model) is an optional dependency for this
project, following the same pattern as scikit-learn for fact_checker.py.
Tests split into a degradation group (must pass with or without spaCy) and
a matching group (skipped, not failed, when spaCy/the model is absent).

Run from anywhere with:
    python3 tests/test_ner_extractor.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import ner_extractor as ner  # noqa: E402


def field(value, source_line=None, script="latin"):
    return {
        "value": value,
        "confidence": 0.9,
        "source_line": source_line or value,
        "extra": {"script": script},
    }


class DegradationTests(unittest.TestCase):
    """Must hold whether or not spaCy is installed."""

    @unittest.skipIf(ner.ner_available(), "this asserts the no-spaCy degraded path")
    def test_missing_spacy_reports_unavailable(self):
        values = {"owner_name": field("Ram Prasad")}
        issues = ner.cross_check(values)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule, "NER_UNAVAILABLE")
        self.assertEqual(issues[0].severity, "info")

    def test_missing_value_is_skipped_regardless(self):
        # No value at all for any checked field - nothing to cross-check,
        # and this must not be confused with "checked and unconfirmed".
        if ner.ner_available():
            issues = ner.cross_check({})
            self.assertEqual(issues, [])


@unittest.skipUnless(ner.ner_available(), "spaCy/en_core_web_sm not installed")
class MatchingTests(unittest.TestCase):
    def test_person_name_confirmed(self):
        # Verified empirically (see module docstring): the bare label-value
        # line alone gives the model too little context, which is exactly
        # why the check runs against a wrapped form of the value instead.
        values = {"owner_name": field(
            "Vitthal Bhau Pawar", source_line="Owner Name : Vitthal Bhau Pawar")}
        issues = ner.cross_check(values)
        hit = next(i for i in issues if i.field == "owner_name")
        self.assertEqual(hit.rule, "NER_CONFIRMED")
        self.assertEqual(hit.severity, "info")

    def test_devanagari_script_is_skipped_not_guessed(self):
        values = {"owner_name": field(
            "रामप्रसाद वर्मा", source_line="Owner Name : रामप्रसाद वर्मा",
            script="devanagari")}
        issues = ner.cross_check(values)
        # No NER_* issue for owner_name - the module makes no claim on
        # non-Latin script rather than running an English model on it.
        self.assertFalse(any(i.field == "owner_name" for i in issues))

    def test_date_field_is_checked_against_date_entities(self):
        values = {"registration_date": field(
            "2020-01-01", source_line="Registration Date : 1st January 2020")}
        issues = ner.cross_check(values)
        hit = next(i for i in issues if i.field == "registration_date")
        self.assertIn(hit.rule, ("NER_CONFIRMED", "NER_UNCONFIRMED", "NER_MISMATCH"))

    def test_same_date_different_format_is_not_a_false_mismatch(self):
        # Regression: `value` is ISO ("2031-02-27") while a NER-found DATE
        # span stays in the source's raw format ("27/02/2031"). Comparing
        # those strings directly always looks like a mismatch even when the
        # underlying date is identical - both sides must be parsed to the
        # same representation before comparing.
        values = {"registration_date": field(
            "2031-02-27",
            source_line="Registration Date : 27/02/2031")}
        issues = ner.cross_check(values)
        hit = next(i for i in issues if i.field == "registration_date")
        # Whether or not spaCy tags "27/02/2031" as DATE at all is
        # inconsistent (see module docstring) and not what this test
        # guards - NER_MISMATCH specifically must never fire from a pure
        # formatting difference against the same real date.
        self.assertNotEqual(hit.rule, "NER_MISMATCH")

    def test_khasra_number_is_never_checked(self):
        # No generic NER model has a notion of "khasra number" - this field
        # must never appear in cross_check's output at all, matching or not.
        values = {
            "khasra_number": field("237/4"),
            "owner_name": field("Suresh Chand", source_line="Owner Name : Suresh Chand"),
        }
        issues = ner.cross_check(values)
        self.assertFalse(any(i.field == "khasra_number" for i in issues))


if __name__ == "__main__":
    unittest.main()
