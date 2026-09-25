#!/usr/bin/env python3
"""
Unit tests for backend/validator.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Run from anywhere with:
    python3 tests/test_validator.py
    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import validator  # noqa: E402


def field(value, confidence=0.95, extra=None):
    """Build one {value, confidence, extra} entry as validator expects."""
    return {"value": value, "confidence": confidence, "extra": extra or {}}


def record(**fields):
    """
    Build a minimal valid land record, overridden by keyword arguments.
    A bare string sets the value at default confidence; pass a dict (from
    `field()`) to control confidence/extra explicitly.
    """
    base = {
        "khasra_number": field("142/2"),
        "khata_number": field("87"),
        "owner_name": field("Ram Prasad Yadav"),
        "area": field("0.4820 hectare", extra={"area": {
            "sqm": 4820.0, "unit": "hectare", "unit_missing": False,
            "regional_unit": False,
        }}),
        "village": field("Rampur"),
        "district": field("Lucknow"),
    }
    for key, val in fields.items():
        base[key] = val if isinstance(val, dict) else field(val)
    return base


class RequiredFieldsTests(unittest.TestCase):
    def test_all_required_present_raises_nothing(self):
        issues = validator.rule_required_fields(record())
        self.assertEqual(issues, [])

    def test_missing_required_field_is_error(self):
        values = record()
        del values["owner_name"]
        issues = validator.rule_required_fields(values)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule, "REQUIRED_MISSING")
        self.assertEqual(issues[0].severity, "error")
        self.assertEqual(issues[0].field, "owner_name")


class AreaSanityTests(unittest.TestCase):
    def test_normal_area_is_clean(self):
        issues = validator.rule_area_sanity(record())
        self.assertEqual(issues, [])

    def test_unparsed_area_is_error(self):
        issues = validator.rule_area_sanity(record(area=field("blah")))
        self.assertTrue(any(i.rule == "AREA_UNPARSED" and i.severity == "error"
                             for i in issues))

    def test_missing_unit_is_error(self):
        values = record(area=field("1200", extra={"area": {
            "sqm": None, "unit": None, "unit_missing": True,
        }}))
        issues = validator.rule_area_sanity(values)
        self.assertTrue(any(i.rule == "AREA_UNIT_MISSING" and i.severity == "error"
                             for i in issues))

    def test_non_positive_area_is_error(self):
        values = record(area=field("0 sq.m", extra={"area": {
            "sqm": 0.0, "unit": "sqm", "unit_missing": False,
        }}))
        issues = validator.rule_area_sanity(values)
        self.assertTrue(any(i.rule == "AREA_NON_POSITIVE" for i in issues))

    def test_mildly_small_area_is_warning(self):
        values = record(area=field("2 sq.m", extra={"area": {
            "sqm": 2.0, "unit": "sqm", "unit_missing": False,
        }}))
        issues = validator.rule_area_sanity(values)
        matches = [i for i in issues if i.rule == "AREA_RANGE"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].severity, "warning")

    def test_impossibly_small_area_is_error(self):
        values = record(area=field("0.1 sq.m", extra={"area": {
            "sqm": 0.1, "unit": "sqm", "unit_missing": False,
        }}))
        issues = validator.rule_area_sanity(values)
        matches = [i for i in issues if i.rule == "AREA_RANGE"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].severity, "error")

    def test_impossibly_large_area_is_error(self):
        values = record(area=field("50000 hectare", extra={"area": {
            "sqm": 500_000_000.0, "unit": "hectare", "unit_missing": False,
        }}))
        issues = validator.rule_area_sanity(values)
        matches = [i for i in issues if i.rule == "AREA_RANGE"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].severity, "error")

    def test_regional_unit_raises_info(self):
        values = record(area=field("2 bigha", extra={"area": {
            "sqm": 5058.58, "unit": "bigha", "unit_missing": False,
            "regional_unit": True,
        }}))
        issues = validator.rule_area_sanity(values)
        self.assertTrue(any(i.rule == "AREA_REGIONAL_UNIT" and i.severity == "info"
                             for i in issues))


class IdentifierFormatTests(unittest.TestCase):
    def test_valid_khasra_is_clean(self):
        issues = validator.rule_identifier_formats(record())
        self.assertEqual(issues, [])

    def test_bad_format_is_warning(self):
        issues = validator.rule_identifier_formats(record(khata_number="ABC!!"))
        self.assertTrue(any(i.rule == "FORMAT_INVALID" and i.severity == "warning"
                             for i in issues))

    def test_all_zero_identifier_is_error(self):
        issues = validator.rule_identifier_formats(record(khata_number="000"))
        self.assertTrue(any(i.rule == "IDENTIFIER_ZERO" and i.severity == "error"
                             for i in issues))


class DateRuleTests(unittest.TestCase):
    def test_future_date_is_error(self):
        issues = validator.rule_dates(record(registration_date="2099-01-01"))
        self.assertTrue(any(i.rule == "DATE_FUTURE" and i.severity == "error"
                             for i in issues))

    def test_pre_1850_date_is_warning(self):
        issues = validator.rule_dates(record(registration_date="1700-01-01"))
        self.assertTrue(any(i.rule == "DATE_TOO_OLD" and i.severity == "warning"
                             for i in issues))

    def test_mutation_before_registration_is_error(self):
        values = record(registration_date="2020-05-01", mutation_date="2019-01-01")
        issues = validator.rule_dates(values)
        self.assertTrue(any(i.rule == "DATE_ORDER" and i.severity == "error"
                             for i in issues))

    def test_unparsable_date_is_warning(self):
        issues = validator.rule_dates(record(mutation_date="not-a-date"))
        self.assertTrue(any(i.rule == "DATE_UNPARSED" and i.severity == "warning"
                             for i in issues))


class AdministrativeHierarchyTests(unittest.TestCase):
    def test_known_district_and_state_is_clean(self):
        values = record(district="Lucknow", state="Uttar Pradesh")
        issues = validator.rule_administrative_hierarchy(values)
        self.assertEqual(issues, [])

    def test_unknown_district_suggests_closest_match(self):
        values = record(district="Kanpurr Nagar")
        issues = validator.rule_administrative_hierarchy(values)
        hit = next(i for i in issues if i.rule == "DISTRICT_UNKNOWN")
        self.assertEqual(hit.severity, "warning")
        self.assertIn("Kanpur Nagar", hit.suggestion)

    def test_state_mismatch_is_error(self):
        values = record(district="Lucknow", state="Bihar")
        issues = validator.rule_administrative_hierarchy(values)
        self.assertTrue(any(i.rule == "STATE_MISMATCH" and i.severity == "error"
                             for i in issues))

    def test_missing_state_is_inferred(self):
        values = record(district="Lucknow")
        del values["district"]
        values["district"] = field("Lucknow")
        issues = validator.rule_administrative_hierarchy(values)
        self.assertTrue(any(i.rule == "STATE_INFERRED" and i.severity == "info"
                             for i in issues))

    def test_tehsil_not_in_district_is_warning(self):
        values = record(district="Lucknow", tehsil="Bilhaur")
        issues = validator.rule_administrative_hierarchy(values)
        self.assertTrue(any(i.rule == "TEHSIL_MISMATCH" and i.severity == "warning"
                             for i in issues))

    def test_village_known_under_different_tehsil_is_warning(self):
        # Real LGD data: "Acharamau" is a real village under Lucknow's
        # Bakshi Ka Talab tehsil, not Sadar.
        values = record(district="Lucknow", tehsil="Sadar", village="Acharamau")
        issues = validator.rule_administrative_hierarchy(values)
        hit = next(i for i in issues if i.rule == "VILLAGE_TEHSIL_MISMATCH")
        self.assertEqual(hit.severity, "warning")
        self.assertEqual(hit.field, "village")
        self.assertIn("Bakshi Ka Talab", hit.message)

    def test_village_matching_its_claimed_tehsil_is_clean(self):
        values = record(district="Lucknow", tehsil="Bakshi Ka Talab", village="Acharamau")
        issues = validator.rule_administrative_hierarchy(values)
        self.assertFalse(any(i.rule == "VILLAGE_TEHSIL_MISMATCH" for i in issues))

    def test_village_absent_from_bundle_is_not_flagged(self):
        # The village gazetteer is a capped, non-exhaustive subset per tehsil
        # (see admin_master.json's _comment) - a real village that simply
        # isn't in our bundle must never be reported as a mismatch.
        values = record(district="Lucknow", tehsil="Sadar",
                        village="ThisVillageIsNotInAnyBundle")
        issues = validator.rule_administrative_hierarchy(values)
        self.assertFalse(any(i.rule == "VILLAGE_TEHSIL_MISMATCH" for i in issues))


class LandClassificationTests(unittest.TestCase):
    def test_known_class_is_clean(self):
        issues = validator.rule_land_classification(record(land_classification="agricultural"))
        self.assertEqual(issues, [])

    def test_unmapped_class_is_warning(self):
        issues = validator.rule_land_classification(record(land_classification="spaceport"))
        self.assertTrue(any(i.rule == "CLASS_UNMAPPED" and i.severity == "warning"
                             for i in issues))


class OwnershipConsistencyTests(unittest.TestCase):
    def test_owner_equals_father_is_warning(self):
        values = record(owner_name="Ram Prasad", father_name="Ram Prasad")
        issues = validator.rule_ownership_consistency(values)
        self.assertTrue(any(i.rule == "OWNER_FATHER_SAME" and i.severity == "warning"
                             for i in issues))

    def test_share_fraction_over_unity_is_error(self):
        issues = validator.rule_ownership_consistency(record(share="7/5"))
        self.assertTrue(any(i.rule == "SHARE_OVER_UNITY" and i.severity == "error"
                             for i in issues))

    def test_share_zero_denominator_is_error(self):
        issues = validator.rule_ownership_consistency(record(share="1/0"))
        self.assertTrue(any(i.rule == "SHARE_INVALID" and i.severity == "error"
                             for i in issues))

    def test_share_percentage_over_100_is_error(self):
        issues = validator.rule_ownership_consistency(record(share="150%"))
        self.assertTrue(any(i.rule == "SHARE_OVER_UNITY" and i.severity == "error"
                             for i in issues))

    def test_share_percentage_zero_is_error(self):
        issues = validator.rule_ownership_consistency(record(share="0%"))
        self.assertTrue(any(i.rule == "SHARE_INVALID" and i.severity == "error"
                             for i in issues))

    def test_share_percentage_within_range_is_clean(self):
        issues = validator.rule_ownership_consistency(record(share="45%"))
        self.assertEqual(issues, [])


class LinkedRecordCompletenessTests(unittest.TestCase):
    def test_number_without_date_is_warning(self):
        values = record(mutation_number="45/2020")
        issues = validator.rule_linked_record_completeness(values)
        self.assertTrue(any(i.rule == "MUTATION_DATE_MISSING" and i.severity == "warning"
                             for i in issues))

    def test_date_without_number_is_warning(self):
        values = record(registration_date="2020-01-01")
        issues = validator.rule_linked_record_completeness(values)
        self.assertTrue(any(i.rule == "REGISTRATION_NUMBER_MISSING" and i.severity == "warning"
                             for i in issues))

    def test_pair_present_is_clean(self):
        values = record(mutation_number="45/2020", mutation_date="2020-01-01")
        issues = validator.rule_linked_record_completeness(values)
        self.assertEqual(issues, [])

    def test_pair_absent_is_clean(self):
        issues = validator.rule_linked_record_completeness(record())
        self.assertEqual(issues, [])


class PlaceholderValueTests(unittest.TestCase):
    def test_placeholder_in_required_field_is_error(self):
        issues = validator.rule_placeholder_values(record(village="N/A"))
        hit = next(i for i in issues if i.rule == "PLACEHOLDER_VALUE" and i.field == "village")
        self.assertEqual(hit.severity, "error")

    def test_placeholder_in_optional_field_is_warning(self):
        issues = validator.rule_placeholder_values(record(tehsil="Unknown"))
        hit = next(i for i in issues if i.rule == "PLACEHOLDER_VALUE" and i.field == "tehsil")
        self.assertEqual(hit.severity, "warning")

    def test_real_value_is_clean(self):
        issues = validator.rule_placeholder_values(record())
        self.assertEqual(issues, [])


class LowConfidenceTests(unittest.TestCase):
    def test_high_confidence_is_clean(self):
        issues = validator.rule_low_confidence(record())
        self.assertEqual(issues, [])

    def test_low_confidence_required_field_is_error(self):
        values = record(owner_name=field("Ram", confidence=0.4))
        issues = validator.rule_low_confidence(values)
        hit = next(i for i in issues if i.field == "owner_name")
        self.assertEqual(hit.severity, "error")

    def test_low_confidence_optional_field_is_warning(self):
        values = record(tehsil=field("Sadar", confidence=0.5))
        issues = validator.rule_low_confidence(values)
        hit = next(i for i in issues if i.field == "tehsil")
        self.assertEqual(hit.severity, "warning")


class DuplicateDetectionTests(unittest.TestCase):
    def test_signature_requires_khasra_and_village(self):
        values = record()
        del values["village"]
        self.assertIsNone(validator.parcel_signature(values))

    def test_same_owner_rescan_is_warning(self):
        values = record()
        sig = validator.parcel_signature(values)
        existing = [{"document_id": 1, "filename": "a.pdf", "signature": sig,
                     "owner_name": "Ram Prasad Yadav"}]
        issues = validator.rule_duplicates(values, existing)
        self.assertTrue(any(i.rule == "DUPLICATE" and i.severity == "warning"
                             for i in issues))

    def test_different_owner_is_conflict_error(self):
        values = record()
        sig = validator.parcel_signature(values)
        existing = [{"document_id": 1, "filename": "a.pdf", "signature": sig,
                     "owner_name": "Someone Else"}]
        issues = validator.rule_duplicates(values, existing)
        self.assertTrue(any(i.rule == "DUPLICATE_CONFLICT" and i.severity == "error"
                             for i in issues))


@unittest.skipUnless(validator.fact_checker.SKLEARN_AVAILABLE,
                     "scikit-learn not installed - fact-check integration path skipped")
class FactCheckIntegrationTests(unittest.TestCase):
    """Confirms rule_fact_check's output reaches validate()'s routing decision."""

    def test_registry_owner_conflict_blocks_the_record(self):
        values = record(
            khasra_number="237/4", khata_number="1428",
            village="नरहरपुर", district="Lucknow",
            owner_name="मोहन लाल गुप्ता",  # registry has रामप्रसाद वर्मा for this parcel
        )
        result = validator.validate(values)
        self.assertEqual(result["decision"], "blocked")
        self.assertTrue(any(i["rule"] == "FACT_CHECK_OWNER_MISMATCH" for i in result["issues"]))

    def test_registry_match_is_a_clean_verification(self):
        values = record(
            khasra_number="237/4", khata_number="1428",
            village="नरहरपुर", district="Lucknow",
            owner_name="रामप्रसाद वर्मा",
            area=field("1.2540 hectare", extra={"area": {"sqm": 12540.0}}),
        )
        result = validator.validate(values)
        self.assertEqual(result["decision"], "auto_approved")
        self.assertTrue(any(i["rule"] == "FACT_CHECK_VERIFIED" for i in result["issues"]))


class ValidateOrchestrationTests(unittest.TestCase):
    def test_clean_record_is_auto_approved(self):
        result = validator.validate(record())
        self.assertEqual(result["decision"], "auto_approved")
        self.assertEqual(result["error_count"], 0)

    def test_warning_only_record_needs_review(self):
        result = validator.validate(record(land_classification="spaceport"))
        self.assertEqual(result["decision"], "needs_review")

    def test_error_blocks_record(self):
        result = validator.validate(record(registration_date="2099-01-01"))
        self.assertEqual(result["decision"], "blocked")
        self.assertGreaterEqual(result["error_count"], 1)

    def test_a_broken_rule_never_crashes_validation(self):
        original = validator.ALL_RULES
        try:
            def boom(_values):
                raise RuntimeError("simulated rule failure")
            validator.ALL_RULES = original + [("BOOM", boom)]
            result = validator.validate(record())
            self.assertTrue(any(i["rule"] == "RULE_ERROR" for i in result["issues"]))
        finally:
            validator.ALL_RULES = original

    def test_trust_score_is_bounded(self):
        result = validator.validate(record())
        self.assertGreaterEqual(result["trust_score"], 0.0)
        self.assertLessEqual(result["trust_score"], 100.0)



class RegionalIdentifierTests(unittest.TestCase):
    """
    "Khasra" is North Indian revenue vocabulary, not a national field.

    Tamil Nadu, Karnataka, Telangana, Andhra Pradesh and Kerala issue no
    khasra at all - the parcel identifier there is the survey number. Before
    this group existed, a South Indian record with every field read at 0.95+
    confidence was still blocked on REQUIRED_MISSING: khasra_number, so the
    system's accuracy on half the country could not matter.
    """

    @staticmethod
    def _record(**extra):
        base = {
            "khata_number": {"value": "1182", "confidence": 0.96, "status": "extracted"},
            "owner_name": {"value": "Ramesh Kumar", "confidence": 0.95, "status": "extracted"},
            "area": {"value": "1.4520 hectare", "confidence": 0.95, "status": "extracted"},
            "village": {"value": "Rampuram", "confidence": 0.95, "status": "extracted"},
            "district": {"value": "Chennai", "confidence": 0.95, "status": "extracted"},
        }
        base.update(extra)
        return base

    @staticmethod
    def _errors(result):
        return [i for i in result["issues"] if i["severity"] == "error"]

    def _missing_identifier(self, result):
        return [i for i in self._errors(result)
                if i["rule"] == "REQUIRED_MISSING"
                and i["field"] in validator.IDENTIFIER_GROUP]

    def test_a_survey_number_satisfies_the_parcel_identifier(self):
        result = validator.validate(self._record(
            survey_number={"value": "237/4", "confidence": 0.97, "status": "extracted"}),
            existing=[])
        self.assertEqual(self._missing_identifier(result), [])
        self.assertNotEqual(result["decision"], "blocked")

    def test_a_khasra_number_still_satisfies_it(self):
        result = validator.validate(self._record(
            khasra_number={"value": "237/4", "confidence": 0.97, "status": "extracted"}),
            existing=[])
        self.assertEqual(self._missing_identifier(result), [])
        self.assertNotEqual(result["decision"], "blocked")

    def test_a_record_naming_no_parcel_at_all_is_still_blocked(self):
        """The group must not become a way to skip the requirement. A record
        that identifies no parcel identifies no land."""
        result = validator.validate(self._record(), existing=[])
        self.assertEqual(result["decision"], "blocked")
        self.assertEqual(len(self._missing_identifier(result)), 1)

    def test_the_error_names_both_forms_rather_than_demanding_one(self):
        """A Tamil Nadu verifier told to supply a 'Khasra Number' has been
        asked for a document that does not exist in their state."""
        result = validator.validate(self._record(), existing=[])
        message = self._missing_identifier(result)[0]["message"].lower()
        self.assertIn("khasra", message)
        self.assertIn("survey", message)

    def test_the_group_is_reported_once_not_per_member(self):
        result = validator.validate(self._record(), existing=[])
        self.assertEqual(len(self._missing_identifier(result)), 1)

    def test_other_required_fields_are_unaffected(self):
        record = self._record(
            survey_number={"value": "237/4", "confidence": 0.97, "status": "extracted"})
        record.pop("village")
        result = validator.validate(record, existing=[])
        missing = [i["field"] for i in self._errors(result)
                   if i["rule"] == "REQUIRED_MISSING"]
        self.assertIn("village", missing)


if __name__ == "__main__":
    unittest.main()
