#!/usr/bin/env python3
"""
Unit tests for backend/doc_type.py (document-type detection and schema scoping).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Pure text matching, no optional dependencies, so every test always runs.

The regression these guard against was found on a real notarised Power of
Attorney: the OCR read every printed value correctly and the record was still
`blocked` on four REQUIRED_MISSING errors, because a GPA structurally carries
no khasra number, no khata number and no area. Zero of that document's twelve
printed field labels existed among the schema's 221 aliases. Demanding them
was asking the paper to be something it is not.

Run from anywhere with:
    python3 tests/test_doc_type.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import doc_type as dt  # noqa: E402
import field_extractor as fe  # noqa: E402

GPA = ("INDIA NON JUDICIAL Government of National Capital Territory of Delhi "
       "e-Stamp Certificate No. IN-DL937006 Certificate Issued Date "
       "Account Reference Unique Doc. Reference Purchased by PUSHPA "
       "Description of Document Article 48(c) Power of attorney - GPA "
       "Property Description VILLAGE NARELA, SABOLI ROAD, DELHI "
       "First Party RAJBIR SINGH Second Party PUSHPA Stamp Duty Paid By")

ROR = ("भूमि का विवरण "
       "Plot Information Khata No: 00620 Area: 0.3120 Hectare "
       "खसरा नंबर : 50 Owner Details")

SALE = ("This SALE DEED is made between the vendor and the vendee. "
        "Consideration price paid. The property is sold and transferred.")

MUTATION = ("Mutation Order दाखिल खारिज "
            "नामांतरण Order Description "
            "तहसीलदार")

CARRIER_ONLY = ("INDIA NON JUDICIAL e-Stamp Certificate No. "
                "Stamp Duty Paid By Account Reference")


class DetectionTests(unittest.TestCase):

    def test_a_record_of_rights_is_recognised(self):
        self.assertEqual(dt.detect(ROR)["type"], "record_of_rights")

    def test_a_power_of_attorney_is_recognised(self):
        self.assertEqual(dt.detect(GPA)["type"], "power_of_attorney")

    def test_a_sale_deed_is_recognised(self):
        self.assertEqual(dt.detect(SALE)["type"], "sale_deed")

    def test_a_mutation_order_is_recognised(self):
        self.assertEqual(dt.detect(MUTATION)["type"], "mutation_order")

    def test_an_unrelated_page_is_not_guessed_at(self):
        """
        "Unknown" is a usable answer. Guessing a type would silently relax the
        required-field rules on a document nobody identified.
        """
        result = dt.detect("a scanned page about something else entirely")
        self.assertEqual(result["type"], "unknown")
        self.assertFalse(result["claimed"])

    def test_empty_input_is_safe(self):
        for empty in ("", None, []):
            self.assertEqual(dt.detect(empty)["type"], "unknown")

    def test_lines_are_accepted_as_well_as_text(self):
        class L:
            def __init__(self, t):
                self.text = t
        lines = [L(part) for part in ROR.split()]
        self.assertEqual(dt.detect(lines)["type"], "record_of_rights")

    def test_the_verdict_carries_its_evidence(self):
        """A type decision changes which fields are demanded, so a reviewer
        who disagrees needs to see what drove it."""
        result = dt.detect(GPA)
        self.assertTrue(result["evidence"])
        self.assertIn("power of attorney", result["evidence"])
        self.assertIn("matched on", result["reason"])


class CarrierTests(unittest.TestCase):
    """
    A stamp certificate is the paper, not the document.

    Competing the carrier against real instruments failed twice on the same
    file: the e-Stamp boilerplate first won outright (19.0 against the GPA's
    17.0), and once OCR improved and the two TIED at 17.0 the zero margin made
    the whole thing "unknown". The medium is not a rival hypothesis about what
    a document is, so it is ranked separately and never enters the margin.
    """

    def test_the_instrument_wins_over_its_carrier(self):
        self.assertEqual(dt.detect(GPA)["type"], "power_of_attorney")

    def test_the_carrier_is_still_reported(self):
        result = dt.detect(GPA)
        self.assertEqual(result["carrier"], "stamp_certificate")
        self.assertEqual(result["carrier_display"], "Stamp Certificate")

    def test_a_carrier_only_page_is_named_not_called_unknown(self):
        result = dt.detect(CARRIER_ONLY)
        self.assertEqual(result["type"], "stamp_certificate")
        self.assertTrue(result["claimed"])

    def test_a_record_of_rights_reports_no_carrier(self):
        self.assertIsNone(dt.detect(ROR)["carrier"])

    def test_the_carrier_never_suppresses_the_instrument_on_a_tie(self):
        """The exact failure: equal scores must not produce 'unknown'."""
        tie = CARRIER_ONLY + " Power of attorney GPA article 48 first party"
        self.assertEqual(dt.detect(tie)["type"], "power_of_attorney")


class SchemaScopingTests(unittest.TestCase):

    ALL = [f.key for f in fe.FIELD_SPECS]

    def test_a_gpa_is_not_asked_for_record_only_fields(self):
        """
        This test previously asserted that a GPA carries no khasra, khata or
        area either, and the real document disproved it: page 2 reads
        "PLOT NO.55, LAND AREA MEAS. 57 SQ.YDS., out OF KHATONI NO.214/248".
        Citing the land record of the property being dealt with is normal
        drafting, so those fields are applicable after all.

        What a GPA genuinely never carries is the record-keeping apparatus -
        a ULPIN, a land classification, a mutation entry. Those belong to the
        register, not to an instrument executed over it.
        """
        allowed = dt.applicable_fields("power_of_attorney", self.ALL)
        for key in ("ulpin", "land_classification", "mutation_number",
                    "mutation_date"):
            self.assertNotIn(key, allowed, key)

    def test_a_gpa_may_cite_the_land_record_it_concerns(self):
        allowed = dt.applicable_fields("power_of_attorney", self.ALL)
        for key in ("khasra_number", "khata_number", "area"):
            self.assertIn(key, allowed, key)

    def test_citable_fields_are_applicable_but_not_forced(self):
        """
        Applicable is not the same as required. A GPA that omits the khatauni
        must not be blocked for it - the required flag still comes from the
        FieldSpec, and this scoping only ever REMOVES demands.
        """
        import validator as v
        vals = {"owner_name": {"value": "RAJBIR SINGH", "confidence": 0.9,
                               "status": "extracted"},
                "village": {"value": "NARELA", "confidence": 0.9,
                            "status": "extracted"},
                "district": {"value": "Delhi", "confidence": 0.9,
                             "status": "extracted"}}
        result = v.validate(dict(vals), existing=[],
                            doc_type="power_of_attorney")
        missing = [i["field"] for i in result["issues"]
                   if i["rule"] == "REQUIRED_MISSING"]
        self.assertNotIn("khasra_number", missing)
        self.assertNotIn("area", missing)

    def test_a_gpa_is_still_asked_for_its_parties_and_place(self):
        allowed = dt.applicable_fields("power_of_attorney", self.ALL)
        for key in ("owner_name", "village", "district"):
            self.assertIn(key, allowed, key)

    def test_a_record_of_rights_gets_the_whole_schema(self):
        self.assertEqual(dt.applicable_fields("record_of_rights", self.ALL),
                         set(self.ALL))

    def test_an_unknown_type_gets_the_whole_schema_not_an_empty_one(self):
        """
        Reporting "khasra missing" on an unidentified paper is honest.
        Reporting nothing would hide that it was processed at all.
        """
        self.assertEqual(dt.applicable_fields("unknown", self.ALL),
                         set(self.ALL))

    def test_a_mutation_order_is_asked_for_mutation_fields(self):
        allowed = dt.applicable_fields("mutation_order", self.ALL)
        self.assertIn("mutation_number", allowed)
        self.assertIn("khasra_number", allowed)

    def test_a_sale_deed_is_asked_for_area_but_not_khata(self):
        allowed = dt.applicable_fields("sale_deed", self.ALL)
        self.assertIn("area", allowed)
        self.assertNotIn("khata_number", allowed)


class ThresholdTests(unittest.TestCase):

    def test_a_single_weak_keyword_is_not_enough(self):
        """"hectare" alone scores 1.0 and must not claim a type."""
        self.assertEqual(dt.detect("area in hectare")["type"], "unknown")

    def test_a_narrow_win_between_instruments_is_refused(self):
        """
        Land papers overlap, so a thin margin means "both apply", not "this
        one". A sale deed reciting an attorney grant should not be filed
        confidently as either.
        """
        both = "sale deed vendor vendee power of attorney attorney holder"
        self.assertEqual(dt.detect(both)["type"], "unknown")

    def test_the_thresholds_are_positive(self):
        self.assertGreater(dt.MIN_SCORE, 0)
        self.assertGreater(dt.MIN_MARGIN, 0)


if __name__ == "__main__":
    unittest.main()
