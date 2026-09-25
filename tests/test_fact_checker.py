#!/usr/bin/env python3
"""
Unit tests for backend/fact_checker.py (the ML-based external registry
fact-check pipeline).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

scikit-learn is an optional dependency for this project (see README §2), so
these tests split into two groups:
  - tests that must pass regardless of whether scikit-learn is installed
    (the honest-degradation path), and
  - tests of the actual ML matching/comparison logic, skipped with a clear
    reason when scikit-learn is not present rather than failing.

Run from anywhere with:
    python3 tests/test_fact_checker.py
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import fact_checker as fc  # noqa: E402


def field(value, sqm=None):
    extra = {"area": {"sqm": sqm}} if sqm is not None else {}
    return {"value": value, "confidence": 0.9, "extra": extra}


class DegradationTests(unittest.TestCase):
    """Must hold whether or not scikit-learn is installed."""

    def test_registry_file_loaded(self):
        self.assertTrue(fc._REGISTRY.loaded, "bundled registry_master.json should load")
        self.assertGreater(len(fc._REGISTRY.records), 0)

    @unittest.skipIf(fc.SKLEARN_AVAILABLE, "this asserts the no-sklearn degraded path")
    def test_missing_sklearn_reports_unavailable_even_with_no_identity(self):
        # Degradation is announced up front (like MASTER_UNAVAILABLE in
        # validator.py), not only once a record has fields worth checking -
        # an operator needs to know the capability is missing regardless.
        issues = fc.check({"owner_name": field("Someone")})
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule, "FACT_CHECK_UNAVAILABLE")
        self.assertEqual(issues[0].severity, "info")


@unittest.skipUnless(fc.SKLEARN_AVAILABLE, "scikit-learn not installed - ML matching path skipped")
class MatchingTests(unittest.TestCase):
    def test_missing_identity_fields_returns_no_issues(self):
        # No khasra/village at all -> nothing to look up, and this must not
        # be confused with "looked up and found nothing".
        issues = fc.check({"owner_name": field("Someone")})
        self.assertEqual(issues, [])

    def test_exact_identity_verifies_clean_record(self):
        values = {
            "khasra_number": field("237/4"),
            "khata_number": field("1428"),
            "village": field("नरहरपुर"),
            "district": field("Lucknow"),
            "owner_name": field("रामप्रसाद वर्मा"),
            "area": field("1.2540 hectare", sqm=12540.0),
            "land_classification": field("irrigated_agricultural"),
        }
        issues = fc.check(values)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule, "FACT_CHECK_VERIFIED")
        self.assertEqual(issues[0].severity, "info")

    def test_same_parcel_different_owner_is_flagged(self):
        values = {
            "khasra_number": field("237/4"),
            "khata_number": field("1428"),
            "village": field("नरहरपुर"),
            "district": field("Lucknow"),
            "owner_name": field("मोहन लाल गुप्ता"),
            "area": field("1.2540 hectare", sqm=12540.0),
        }
        issues = fc.check(values)
        hit = next(i for i in issues if i.rule == "FACT_CHECK_OWNER_MISMATCH")
        self.assertEqual(hit.severity, "error")
        self.assertEqual(hit.field, "owner_name")

    def test_large_area_discrepancy_is_error(self):
        values = {
            "khasra_number": field("237/4"),
            "khata_number": field("1428"),
            "village": field("नरहरपुर"),
            "district": field("Lucknow"),
            "owner_name": field("रामप्रसाद वर्मा"),
            "area": field("5 hectare", sqm=50000.0),  # registry has 12,540 sq.m
        }
        issues = fc.check(values)
        hit = next(i for i in issues if i.rule == "FACT_CHECK_AREA_MISMATCH")
        self.assertEqual(hit.severity, "error")

    def test_moderate_area_discrepancy_is_warning(self):
        values = {
            "khasra_number": field("142"),
            "khata_number": field("4471"),
            "village": field("Shirur Kasar"),
            "district": field("Pune"),
            "owner_name": field("Vitthal Bhau Pawar"),
            "area": field("0.6070 hectare", sqm=6070.0),  # registry has 5,100 sq.m (~19%)
        }
        issues = fc.check(values)
        hit = next(i for i in issues if i.rule == "FACT_CHECK_AREA_MISMATCH")
        self.assertEqual(hit.severity, "warning")

    def test_classification_mismatch_is_warning(self):
        values = {
            "khasra_number": field("237/4"),
            "khata_number": field("1428"),
            "village": field("नरहरपुर"),
            "district": field("Lucknow"),
            "owner_name": field("रामप्रसाद वर्मा"),
            "land_classification": field("residential"),  # registry: irrigated_agricultural
        }
        issues = fc.check(values)
        hit = next(i for i in issues if i.rule == "FACT_CHECK_CLASS_MISMATCH")
        self.assertEqual(hit.severity, "warning")

    def test_unknown_parcel_is_not_found(self):
        values = {
            "khasra_number": field("999999/9"),
            "khata_number": field("1"),
            "village": field("Nonexistentpur"),
            "district": field("Nowhereabad"),
            "owner_name": field("Nobody"),
        }
        issues = fc.check(values)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule, "FACT_CHECK_NOT_FOUND")
        self.assertEqual(issues[0].severity, "info")


class IdentifierMatchTests(unittest.TestCase):
    """
    A plot identifier must agree exactly, once script and separators are
    normalised. Fuzzy matching is right for names - they carry OCR and
    transliteration noise - and wrong for plot numbers, where one different
    digit means different land.
    """

    def test_ocr_noise_in_an_identifier_is_forgiven(self):
        for extracted in ("237/4", "2374", "२३७/४", " 237 / 4 ", "237-4"):
            self.assertTrue(fc._identifiers_agree(extracted, "237/4"),
                            f"{extracted!r} should match 237/4")

    def test_different_plots_never_agree(self):
        self.assertFalse(fc._identifiers_agree("213/1", "237/4"))
        self.assertFalse(fc._identifiers_agree("218/1", "213/1"))

    def test_blank_identifier_never_agrees(self):
        self.assertFalse(fc._identifiers_agree("", "237/4"))
        self.assertFalse(fc._identifiers_agree(None, "237/4"))


@unittest.skipUnless(fc.SKLEARN_AVAILABLE, "scikit-learn not installed")
class WrongParcelRegressionTests(unittest.TestCase):
    """
    Regression test for a real, blocking false positive found by running the
    seeded corpus: a valid record for khasra 213/1 in नरहरपुर was matched to
    registry entry 237/4 in नरहरपुर - a DIFFERENT plot - because the identity
    string blends the plot number with the village and district, so an exact
    village match carried the similarity over MATCH_THRESHOLD. The owner of
    that other parcel then failed to match, raising FACT_CHECK_OWNER_MISMATCH
    at *error* severity and blocking a document that had nothing wrong with it.
    """

    def _values(self, khasra):
        return {
            "khasra_number": {"value": khasra, "confidence": 0.95},
            "village": {"value": "नरहरपुर", "confidence": 0.95},
            "district": {"value": "Lucknow", "confidence": 0.95},
            "owner_name": {"value": "सुनीता देवी", "confidence": 0.95},
        }

    def test_unlisted_plot_in_a_listed_village_is_not_found_not_mismatched(self):
        issues = fc.check(self._values("213/1"))
        rules = {i.rule for i in issues}
        self.assertIn("FACT_CHECK_NOT_FOUND", rules)
        self.assertNotIn("FACT_CHECK_OWNER_MISMATCH", rules)
        self.assertNotIn("FACT_CHECK_AREA_MISMATCH", rules)

    def test_it_never_blocks_on_someone_elses_parcel(self):
        for issue in fc.check(self._values("213/1")):
            self.assertNotEqual(issue.severity, "error")

    def test_the_message_names_the_parcel_it_declined_to_use(self):
        """A reviewer must be able to see WHY nothing was compared, or this
        looks identical to 'the registry has no data for this village'."""
        issues = fc.check(self._values("213/1"))
        text = " ".join(i.message for i in issues)
        self.assertIn("213/1", text)
        self.assertIn("different parcel", text)


if __name__ == "__main__":
    unittest.main()
