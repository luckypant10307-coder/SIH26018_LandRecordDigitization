#!/usr/bin/env python3
"""
Unit tests for backend/gazetteer.py (post-OCR correction against vocabularies).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The tests that matter here guard against a CONFIDENT WRONG CORRECTION, which
is worse than no correction at all: rewriting one real village into another
real village silently reassigns a parcel. The measurements behind the design
are in the module docstring; these pin the behaviour those measurements
implied.

  * ScopingTests     - a village is matched only inside its tehsil. Unscoped,
                       28.7% of villages have a near-identical rival
                       somewhere in India; scoped, 1.9%.
  * DangerTests      - the specific pairs that broke a single global
                       threshold (Barabanki/Barabani at 94.1 scores HIGHER
                       than Lucknov/Lucknow at 85.7).
  * ScriptTests      - the bundled LGD extract is entirely romanised, so a
                       Devanagari value has nothing to compare against and
                       must say so rather than score 0 and report "not found".
  * IdentifierTests  - a plot number is shape-checked, never snapped.

Run from anywhere with:
    python3 tests/test_gazetteer.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import gazetteer as gz  # noqa: E402


def values(**kw):
    return {k: {"value": v} for k, v in kw.items()}


def by_field(corrections):
    return {c.field_key: c for c in corrections}


class LoadTests(unittest.TestCase):
    def test_the_bundled_gazetteer_loads(self):
        self.assertTrue(gz.GAZETTEER.loaded)
        report = gz.describe()
        self.assertEqual(report["districts"], 43)
        self.assertEqual(report["tehsils"], 453)
        self.assertEqual(report["villages"], 6516)

    def test_rapidfuzz_presence_is_reported_not_assumed(self):
        self.assertIn("rapidfuzz", gz.describe())


@unittest.skipUnless(gz.RAPIDFUZZ_AVAILABLE, "RapidFuzz not installed")
class DistrictTests(unittest.TestCase):
    def test_an_exact_district_is_confirmed_without_scoring(self):
        out = by_field(gz.correct_record(values(district="Lucknow")))
        self.assertEqual(out["district"].outcome, "confirmed")
        self.assertEqual(out["district"].score, 100.0)
        self.assertFalse(out["district"].applied)

    def test_a_one_character_misread_is_corrected(self):
        """The case the feature exists for: Lucknov -> Lucknow, 85.7."""
        out = by_field(gz.correct_record(values(district="Lucknov")))
        self.assertEqual(out["district"].outcome, "corrected")
        self.assertEqual(out["district"].value, "Lucknow")
        self.assertTrue(out["district"].applied)

    def test_a_corrected_value_is_still_sent_to_a_human(self):
        """
        An applied correction is a machine's guess that happened to clear a
        threshold. It goes in front of a reviewer regardless.
        """
        out = by_field(gz.correct_record(values(district="Lucknov")))
        self.assertTrue(out["district"].needs_review)

    def test_an_unlisted_district_is_not_invented(self):
        out = by_field(gz.correct_record(values(district="Barabanki")))
        self.assertEqual(out["district"].outcome, "not_found")
        self.assertEqual(out["district"].value, "Barabanki")
        self.assertFalse(out["district"].applied)

    def test_absence_is_not_reported_as_proof_of_error(self):
        """The extract is a partial snapshot; saying otherwise would be a lie."""
        out = by_field(gz.correct_record(values(district="Barabanki")))
        self.assertIn("partial", out["district"].message)


@unittest.skipUnless(gz.RAPIDFUZZ_AVAILABLE, "RapidFuzz not installed")
class ScopingTests(unittest.TestCase):
    """
    Scoping is the entire safety argument, so it is tested from both ends:
    it has to ENABLE a correction that is safe, and REFUSE one that is not.
    """

    def test_tehsil_is_matched_within_the_resolved_district(self):
        out = by_field(gz.correct_record(
            values(district="Lucknow", tehsil="Malihahad")))
        self.assertEqual(out["tehsil"].outcome, "corrected")
        self.assertEqual(out["tehsil"].value, "Malihabad")
        self.assertIn("Lucknow", out["tehsil"].scope)

    def test_an_unresolvable_district_blocks_the_levels_below_it(self):
        """
        Without a district there is no list to scope to, and matching
        nationally was measured at 28.7% ambiguity. Degrading to "cannot
        check" is correct; falling back to the national list would not be.
        """
        out = by_field(gz.correct_record(
            values(district="Nowhereabad", village="Acharamau")))
        self.assertEqual(out["village"].outcome, "no_vocabulary")
        self.assertFalse(out["village"].applied)
        self.assertTrue(out["village"].needs_review)

    def test_a_real_village_from_another_tehsil_is_not_accepted(self):
        """
        'Acharamau' is a real village - in Bakshi Ka Talab, not Malihabad.
        Scoped to the wrong tehsil it must come back not_found, because
        accepting it would place the parcel in the wrong tehsil.
        """
        out = by_field(gz.correct_record(
            values(district="Lucknow", tehsil="Malihabad", village="Acharamau")))
        self.assertEqual(out["village"].outcome, "not_found")
        self.assertFalse(out["village"].applied)

    def test_a_village_in_its_own_tehsil_is_confirmed(self):
        village = gz.GAZETTEER.villages("Lucknow", "Bakshi Ka Talab")[0]
        out = by_field(gz.correct_record(
            values(district="Lucknow", tehsil="Bakshi Ka Talab", village=village)))
        self.assertEqual(out["village"].outcome, "confirmed")

    def test_the_scope_is_named_in_the_result(self):
        """A reviewer has to be able to see WHAT the value was checked against."""
        out = by_field(gz.correct_record(
            values(district="Lucknow", tehsil="Sadar", village="Whatever")))
        self.assertIn("Sadar", out["village"].scope)


@unittest.skipUnless(gz.RAPIDFUZZ_AVAILABLE, "RapidFuzz not installed")
class DangerTests(unittest.TestCase):
    """
    The pairs that killed a single global threshold. Similarity alone cannot
    license a correction here, and these are the receipts.
    """

    def test_similarity_does_not_separate_a_misread_from_a_neighbour(self):
        from rapidfuzz import fuzz
        good = fuzz.ratio("Lucknov", "Lucknow")          # same place, misread
        bad = fuzz.ratio("Barabanki", "Barabani")        # DIFFERENT places
        self.assertLess(good, bad,
                        "if this ever reverses, a plain threshold became safe "
                        "and the scoping rationale should be revisited")

    def test_two_close_candidates_are_refused_not_ranked(self):
        result = gz.match_in_scope(
            "Basar", ["Bansar", "Basara", "Kanpur"], "village", "test list")
        self.assertEqual(result.outcome, "ambiguous")
        self.assertFalse(result.applied)
        self.assertGreaterEqual(len(result.candidates), 2)

    def test_an_ambiguous_result_keeps_the_original_value(self):
        result = gz.match_in_scope(
            "Basar", ["Bansar", "Basara"], "village", "test list")
        self.assertEqual(result.value, "Basar")

    def test_a_lone_close_candidate_in_a_narrow_scope_is_applied(self):
        result = gz.match_in_scope(
            "Malihahad", ["Malihabad", "Sadar", "Mohanlalganj"],
            "tehsil", "test list")
        self.assertEqual(result.outcome, "corrected")
        self.assertEqual(result.value, "Malihabad")

    def test_a_weak_best_match_is_only_suggested(self):
        result = gz.match_in_scope(
            "Narharpur", ["Harpur", "Kanpur"], "village", "test list")
        self.assertIn(result.outcome, ("suggested", "not_found"))
        self.assertFalse(result.applied)
        self.assertEqual(result.value, "Narharpur")


@unittest.skipUnless(gz.RAPIDFUZZ_AVAILABLE, "RapidFuzz not installed")
class ScriptTests(unittest.TestCase):
    """
    Crossing scripts.

    These tests used to assert that a Devanagari value reported
    "no comparable vocabulary" and stopped there, because the bundled extract
    is entirely romanised and a direct comparison scores 0.0 against every
    candidate. That is no longer the contract: the value now goes through a
    phonetic key (see TransliterationBridgeTests), so it IS compared.

    What survives unchanged is the important half - a Devanagari name is
    never snapped onto an unrelated romanised one, and a script the
    transliterator cannot handle says so instead of scoring 0 and reporting
    "not found".
    """

    def test_a_devanagari_value_is_compared_through_the_phonetic_key(self):
        out = by_field(gz.correct_record(
            values(district="Lucknow", village="नरहरपुर")))
        self.assertEqual(out["village"].via, "transliteration")
        self.assertIn(out["village"].outcome,
                      ("not_found", "suggested", "corrected", "ambiguous"))

    def test_a_direct_zero_score_match_is_never_returned(self):
        """
        The failure this replaced: scored directly, every romanised candidate
        returns 0.0 for a Devanagari value, and extractOne still names a
        winner ('नरहरपुर' -> 'Acharamau', 0.0). Whatever comes back now, it
        must not be that.
        """
        result = gz.match_in_scope(
            "नरहरपुर", ["Acharamau", "Harpur", "Adhar Khera"],
            "village", "test list")
        self.assertNotEqual(result.value, "Acharamau")
        self.assertFalse(result.applied and result.score == 0.0)

    def test_devanagari_is_never_snapped_to_an_unrelated_name(self):
        result = gz.match_in_scope(
            "नरहरपुर", ["Acharamau", "Adhar Khera", "Bahadurganj"],
            "village", "test list")
        self.assertNotEqual(result.outcome, "corrected")
        self.assertEqual(result.value, "नरहरपुर")

    def test_a_script_without_a_transliterator_reports_no_vocabulary(self):
        """
        Han script has no entry in _TRANSLITERABLE, so there is no key to
        compare with. "cannot compare" and "does not exist" are different
        findings and collapsing them would tell a reviewer their village is
        unknown when it was never checked.
        """
        result = gz.match_in_scope(
            "北京市", ["Lucknow", "Bhopal"], "district", "district list")
        self.assertEqual(result.outcome, "no_vocabulary")
        self.assertIn("NOT evidence", result.message)
        self.assertEqual(result.value, "北京市")


@unittest.skipUnless(gz.RAPIDFUZZ_AVAILABLE, "RapidFuzz not installed")
class StateCrossCheckTests(unittest.TestCase):
    def test_a_state_contradicting_its_district_is_flagged(self):
        out = by_field(gz.correct_record(
            values(district="Lucknow", state="Madhya Pradesh")))
        self.assertEqual(out["state"].outcome, "ambiguous")
        self.assertIn("Uttar Pradesh", out["state"].message)

    def test_neither_side_is_silently_overwritten(self):
        """
        The district and the state disagree; the text cannot say which was
        misread, so nothing is changed.
        """
        out = by_field(gz.correct_record(
            values(district="Lucknow", state="Madhya Pradesh")))
        self.assertEqual(out["state"].value, "Madhya Pradesh")
        self.assertFalse(out["state"].applied)

    def test_a_consistent_state_is_confirmed(self):
        out = by_field(gz.correct_record(
            values(district="Lucknow", state="Uttar Pradesh")))
        self.assertEqual(out["state"].outcome, "confirmed")


class IdentifierTests(unittest.TestCase):
    """
    A plot number has a shape but no vocabulary. 213/1 and 218/1 are both
    valid and denote different land, so it is checked and never repaired.
    """

    def test_plain_and_subdivided_khasra_shapes_pass(self):
        for value in ("237", "237/4", "123/2/1", "1"):
            self.assertEqual(
                gz.check_identifier(value, "khasra_number").outcome, "confirmed", value)

    def test_a_letter_suffixed_survey_number_passes(self):
        """142/2B is normal on a Maharashtra 7/12."""
        self.assertEqual(
            gz.check_identifier("142/2B", "survey_number").outcome, "confirmed")

    def test_devanagari_digits_are_folded_not_rejected(self):
        result = gz.check_identifier("२३७/४", "khasra_number")
        self.assertEqual(result.outcome, "confirmed")
        self.assertEqual(result.value, "237/4")

    def test_a_malformed_khasra_is_flagged_but_left_alone(self):
        result = gz.check_identifier("23-7-4", "khasra_number")
        self.assertEqual(result.outcome, "invalid_shape")
        self.assertEqual(result.value, "23-7-4")
        self.assertTrue(result.needs_review)

    def test_the_message_says_what_the_shape_should_be(self):
        result = gz.check_identifier("oops", "khasra_number")
        self.assertIn("123/2", result.message)

    def test_a_plot_number_is_never_snapped_to_a_neighbour(self):
        """
        The guard against repeating fact_checker.py's original bug, where
        213/1 was matched to 237/4 and the record blocked on the wrong
        parcel's owner.
        """
        result = gz.check_identifier("213/1", "khasra_number")
        self.assertEqual(result.value, "213/1")
        self.assertFalse(result.applied)

    def test_khata_must_be_a_plain_number(self):
        self.assertEqual(gz.check_identifier("1428", "khata_number").outcome, "confirmed")
        self.assertEqual(gz.check_identifier("14/28", "khata_number").outcome, "invalid_shape")

    def test_ulpin_shape(self):
        self.assertEqual(
            gz.check_identifier("UP091223700412", "ulpin").outcome, "confirmed")
        self.assertEqual(
            gz.check_identifier("0912237004", "ulpin").outcome, "invalid_shape")


class ControlledVocabularyTests(unittest.TestCase):
    def test_a_known_land_class_synonym_is_canonicalised(self):
        result = gz.match_land_class("सिंचित")
        self.assertEqual(result.outcome, "confirmed")
        self.assertEqual(result.value, "irrigated_agricultural")

    def test_unirrigated_is_not_read_as_irrigated(self):
        """
        'असिंचित' CONTAINS 'सिंचित'. Matching the shorter synonym first
        inverts the land use of every unirrigated parcel - a bug this project
        has already been bitten by once in the extractor.
        """
        result = gz.match_land_class("असिंचित")
        self.assertEqual(result.value, "unirrigated_agricultural")

    def test_a_regional_term_is_recognised(self):
        self.assertEqual(gz.match_land_class("जिरायत").value,
                         "unirrigated_agricultural")

    def test_an_unknown_term_is_passed_through_not_forced(self):
        result = gz.match_land_class("मिश्रित प्रकार")
        self.assertEqual(result.outcome, "not_found")
        self.assertEqual(result.value, "मिश्रित प्रकार")
        self.assertIn("information, not an error", result.message)

    def test_mutation_types_are_recognised_across_wordings(self):
        for text, code in (("Sale deed", "sale"), ("वरासत", "inheritance"),
                           ("बटवारा", "partition"), ("Mortgage", "mortgage"),
                           ("अदालत", "court_decree")):
            self.assertEqual(gz.match_mutation_type(text).value, code, text)

    def test_an_unknown_mutation_type_is_not_forced(self):
        self.assertEqual(gz.match_mutation_type("zzz unknown").outcome, "not_found")


@unittest.skipUnless(gz.RAPIDFUZZ_AVAILABLE and gz.TRANSLITERATION_AVAILABLE,
                     "RapidFuzz / indic-transliteration not installed")
class TransliterationBridgeTests(unittest.TestCase):
    """
    RapidFuzz is script-agnostic; the bundled LGD extract is not - it is
    entirely romanised. Both sides are therefore reduced to a phonetic key
    before comparison.

    Measured on 16 district pairs whose romanisation is independently known,
    at the district bar of 84: 13 auto-applied, 13 correct, 0 wrong, 3
    refused.
    """

    def test_a_devanagari_district_resolves_to_its_romanised_entry(self):
        out = by_field(gz.correct_record(values(district="भोपाल")))
        self.assertEqual(out["district"].outcome, "corrected")
        self.assertEqual(out["district"].value, "Bhopal")
        self.assertEqual(out["district"].via, "transliteration")

    def test_a_devanagari_village_resolves_inside_its_own_tehsil(self):
        out = by_field(gz.correct_record(
            values(district="भोपाल", tehsil="Berasia", village="अजबपुरा")))
        self.assertEqual(out["village"].outcome, "corrected")
        self.assertEqual(out["village"].value, "Ajabpura")

    def test_scoping_still_refuses_a_village_from_the_wrong_tehsil(self):
        """
        Ajabpura is in Berasia. Transliteration must not let it in through
        Huzur - crossing scripts is allowed, crossing tehsils is not.
        """
        out = by_field(gz.correct_record(
            values(district="भोपाल", tehsil="Huzur", village="अजबपुरा")))
        self.assertNotEqual(out["village"].outcome, "corrected")
        self.assertFalse(out["village"].applied)

    def test_a_transliterated_match_is_always_sent_to_a_human(self):
        out = by_field(gz.correct_record(values(district="भोपाल")))
        self.assertTrue(out["district"].needs_review)
        self.assertIn("approximate", out["district"].message)

    def test_schwa_is_deleted_the_way_hindi_writes_it(self):
        """
        'नरहरपुर' romanises as Narharpur, not naraharapura: Hindi drops the
        inherent 'a'. The deletion runs BEFORE diacritics are stripped, since
        IAST is what distinguishes inherent 'a' from long 'aa' - getting that
        order wrong turned कानपुर into 'knpur' and cost three test pairs.
        """
        self.assertEqual(gz.phonetic_key("कानपुर"), "kanpur")
        self.assertEqual(gz._skeleton(gz.phonetic_key("नरहरपुर")),
                         gz._skeleton(gz.phonetic_key("Narharpur")))

    def test_romanisation_variants_fold_together(self):
        for a, b in (("Aurangabad", "Orangabad"), ("Lakshmipur", "Lakshmipoor"),
                     ("Vardha", "Wardha")):
            self.assertEqual(gz.phonetic_key(a), gz.phonetic_key(b), (a, b))

    def test_a_latin_value_does_not_go_through_the_bridge(self):
        out = by_field(gz.correct_record(values(district="Lucknow")))
        self.assertEqual(out["district"].via, "direct")

    def test_an_unrelated_devanagari_name_is_not_snapped(self):
        result = gz.match_in_scope(
            "क्षेत्रपालनगर", ["Bhopal", "Lucknow", "Jaipur"],
            "district", "district list")
        self.assertNotEqual(result.outcome, "corrected")
        self.assertFalse(result.applied)

    def test_the_verifiable_district_pairs_have_no_wrong_corrections(self):
        """
        The headline guarantee, run as a test rather than quoted from a
        docstring: across every pair whose romanisation is independently
        known, the bridge must never apply a WRONG correction. Refusing is
        acceptable; being confidently wrong is not.
        """
        pairs = [("लखनऊ", "Lucknow"), ("भोपाल", "Bhopal"), ("जयपुर", "Jaipur"),
                 ("पुणे", "Pune"), ("इंदौर", "Indore"), ("नागपुर", "Nagpur"),
                 ("पटना", "Patna"), ("आगरा", "Agra"), ("कानपुर", "Kanpur Nagar"),
                 ("वाराणसी", "Varanasi"), ("जबलपुर", "Jabalpur"),
                 ("उदयपुर", "Udaipur"), ("अमृतसर", "Amritsar"),
                 ("सूरत", "Surat"), ("लुधियाना", "Ludhiana")]
        applied = correct = 0
        for deva, expected in pairs:
            result = by_field(gz.correct_record(values(district=deva)))["district"]
            if result.applied:
                applied += 1
                self.assertEqual(result.value, expected,
                                 f"{deva} was corrected to the WRONG district")
                correct += 1
        self.assertEqual(applied, correct)
        self.assertGreaterEqual(applied, 10,
                                "the bridge stopped resolving anything useful")


class DegradationTests(unittest.TestCase):
    def test_an_empty_value_is_unusable_not_corrected(self):
        self.assertEqual(gz.check_identifier("", "khasra_number").outcome, "unusable")
        self.assertEqual(
            gz.match_in_scope("", ["Lucknow"], "district", "x").outcome, "unusable")

    def test_an_empty_candidate_list_says_so(self):
        result = gz.match_in_scope("Lucknow", [], "district", "district list")
        self.assertEqual(result.outcome, "no_vocabulary")
        self.assertEqual(result.value, "Lucknow")

    def test_a_record_with_no_vocabulary_fields_yields_nothing(self):
        self.assertEqual(gz.correct_record(values(owner_name="Ram Prasad")), [])

    def test_every_correction_serialises(self):
        for c in gz.correct_record(values(district="Lucknov", khasra_number="237/4")):
            payload = c.to_dict()
            self.assertIn("outcome", payload)
            self.assertIn("applied", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
