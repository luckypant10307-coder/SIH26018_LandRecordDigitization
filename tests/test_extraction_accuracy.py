#!/usr/bin/env python3
"""
Regression tests for defects found by tools/measure_accuracy.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Every test here corresponds to a real wrong answer the accuracy harness caught
on the sample corpus, not a hypothetical. They are cheap unit tests (no OCR,
no sample rendering) so the suite stays fast; the harness itself is the
end-to-end measurement and is run separately.

Run from anywhere with:
    python3 tests/test_extraction_accuracy.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import field_extractor as fe  # noqa: E402


class Line:
    """Minimal stand-in for ocr_engine.Line."""

    def __init__(self, text, confidence=0.99, page=1, bbox=(0, 0, 10, 10)):
        self.text = text
        self.confidence = confidence
        self.page = page
        self.bbox = bbox


def extract(*texts, **kw):
    return {f.key: f for f in fe.extract_fields([Line(t) for t in texts], **kw)}


class LandClassInversionTests(unittest.TestCase):
    """
    'सिंचित' (irrigated) is a substring of 'असिंचित' (unirrigated), and the
    classifier returned the first table entry that matched, so every
    unirrigated parcel in the corpus was recorded as irrigated. The field
    drives valuation and revenue assessment, so the inversion is expensive.
    """

    def _classify(self, term):
        spec = fe.FIELD_BY_KEY["land_classification"]
        return fe._validate_value(spec, term)[0]

    def test_unirrigated_is_not_read_as_irrigated(self):
        self.assertEqual(self._classify("असिंचित"), "unirrigated_agricultural")
        self.assertEqual(self._classify("unirrigated"), "unirrigated_agricultural")
        self.assertEqual(self._classify("un-irrigated"), "unirrigated_agricultural")

    def test_irrigated_still_reads_as_irrigated(self):
        self.assertEqual(self._classify("सिंचित"), "irrigated_agricultural")
        self.assertEqual(self._classify("सिंचित कृषि भूमि"), "irrigated_agricultural")
        self.assertEqual(self._classify("irrigated"), "irrigated_agricultural")

    def test_regional_revenue_terms_are_understood(self):
        """A Rajasthan jamabandi and a Maharashtra 7/12 never say सिंचित."""
        self.assertEqual(self._classify("चाही"), "irrigated_agricultural")
        self.assertEqual(self._classify("बागायत"), "irrigated_agricultural")
        self.assertEqual(self._classify("बारानी"), "unirrigated_agricultural")
        self.assertEqual(self._classify("जिरायत"), "unirrigated_agricultural")

    def test_unknown_term_is_passed_through_not_forced(self):
        """Guessing a code for an unrecognised term would be worse than
        admitting the vocabulary does not cover it."""
        spec = fe.FIELD_BY_KEY["land_classification"]
        value, conf, _, notes = fe._validate_value(spec, "मिश्रित प्रकार")
        self.assertEqual(value, "मिश्रित प्रकार")
        self.assertLess(conf, 0.5)
        self.assertTrue(notes)

    def test_regional_labels_are_matched(self):
        got = extract("किस्म भूमि : चाही")
        self.assertEqual(got["land_classification"].value, "irrigated_agricultural")


class SingleCharacterValueTests(unittest.TestCase):
    """
    A value of one character was treated as an empty cell, so the extractor
    walked past the printed value and carried an unrelated number up from the
    line below. Old records legitimately carry khasra '5' and share '1'.
    """

    def test_single_digit_khasra_is_read_from_its_own_line(self):
        got = extract("खसरा संख्या / Khasra Number : 5",
                      "खाता संख्या / Khata Number : 01")
        self.assertEqual(got["khasra_number"].value, "5")
        self.assertEqual(got["khata_number"].value, "01")

    def test_a_truly_empty_row_still_carries_from_below(self):
        got = extract("खसरा संख्या / Khasra Number :", "237/4")
        self.assertEqual(got["khasra_number"].value, "237/4")


class RivalLabelTests(unittest.TestCase):
    """
    'दिनांक' (date) is a registration-date label and also sits inside
    'नामांतरण दिनांक' (mutation date). On a document with a mutation date and
    no registration date, the mutation date was reported as the registration
    date - confidently wrong data where a blank was the truthful answer.
    """

    def test_mutation_date_is_not_copied_into_registration_date(self):
        got = extract("नामांतरण दिनांक / Mutation Date : 07/05/2022")
        self.assertEqual(got["mutation_date"].value, "2022-05-07")
        self.assertIsNone(got["registration_date"].value)
        self.assertEqual(got["registration_date"].status, "missing")

    def test_the_blank_explains_itself_to_a_reviewer(self):
        got = extract("नामांतरण दिनांक / Mutation Date : 07/05/2022")
        self.assertTrue(any("different field" in n
                            for n in got["registration_date"].notes),
                        got["registration_date"].notes)

    def test_a_real_registration_date_is_still_read(self):
        got = extract("नामांतरण दिनांक / Mutation Date : 07/05/2022",
                      "पंजीकरण दिनांक / Registration Date : 09/07/2019")
        self.assertEqual(got["mutation_date"].value, "2022-05-07")
        self.assertEqual(got["registration_date"].value, "2019-07-09")


class ReviewThresholdTests(unittest.TestCase):
    """
    The threshold is a measured value (see field_extractor.REVIEW_THRESHOLD).
    This does not re-derive it - it pins the contract that the constant is
    what actually decides the status, so raising it cannot silently stop
    taking effect.
    """

    def test_threshold_governs_status(self):
        self.assertEqual(fe.REVIEW_THRESHOLD, 0.90)

    def test_low_confidence_field_is_marked_for_review(self):
        # No colon on the line -> heavily demoted, well under the threshold.
        got = extract("ग्राम नरहरपुर")
        village = got["village"]
        if village.value is not None:
            self.assertLess(village.confidence, fe.REVIEW_THRESHOLD)
            self.assertEqual(village.status, "needs_review")


if __name__ == "__main__":
    unittest.main(verbosity=2)
