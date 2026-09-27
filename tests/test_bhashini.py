#!/usr/bin/env python3
"""
Unit tests for backend/bhashini.py (Indic script -> Latin place-name bridge).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

NO TEST HERE TOUCHES THE NETWORK. The suite has to pass on a demo laptop with
no connectivity and no Bhashini credentials, so the transport is stubbed and
what is actually tested is the logic around it: script detection, the exonym
table, candidate ordering, the cache, and the rule that the reference data -
never the model - decides which candidate is right.

The live API is exercised separately by tools/measure_transliteration.py, which
is where a number that depends on a third party belongs.

Run from anywhere with:
    python3 tests/test_bhashini.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import bhashini  # noqa: E402


class CacheIsolated(unittest.TestCase):
    """Each test gets an empty in-memory cache and no disk writes."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._cache_path = bhashini._CACHE_PATH
        bhashini._CACHE_PATH = os.path.join(self._dir.name, "cache.json")
        bhashini._cache = {}
        bhashini._cache_dirty = False
        self._available = bhashini.AVAILABLE

    def tearDown(self):
        bhashini._CACHE_PATH = self._cache_path
        bhashini._cache = None
        bhashini.AVAILABLE = self._available
        self._dir.cleanup()


class TestScriptDetection(unittest.TestCase):

    def test_devanagari_is_detected_as_hindi(self):
        self.assertEqual(bhashini.script_language("लखनऊ"), "hi")

    def test_each_supported_script_maps_to_its_own_language(self):
        # Sending Tamil text as 'hi' returns nothing useful, so the mapping
        # being right is load-bearing, not cosmetic.
        cases = {"বাংলা": "bn", "ਪੰਜਾਬੀ": "pa", "ગુજરાતી": "gu", "ଓଡ଼ିଆ": "or",
                 "தமிழ்": "ta", "తెలుగు": "te", "ಕನ್ನಡ": "kn", "മലയാളം": "ml"}
        for text, expected in cases.items():
            self.assertEqual(bhashini.script_language(text), expected, text)

    def test_latin_text_needs_no_transliteration(self):
        self.assertIsNone(bhashini.script_language("Lucknow"))
        self.assertFalse(bhashini.is_indic("Kanpur Nagar"))

    def test_mixed_text_is_detected_by_its_first_indic_character(self):
        self.assertEqual(bhashini.script_language("Plot 213/1 नरहरपुर"), "hi")

    def test_digits_and_punctuation_are_not_indic(self):
        for text in ("213/1", "", "  ", "0.81 ha", "-"):
            self.assertFalse(bhashini.is_indic(text), repr(text))


class TestExonyms(unittest.TestCase):

    def test_the_shipped_table_covers_the_measured_failures(self):
        """
        These ten are in the table because the live model provably cannot
        produce them. If the table stops covering one, a real district silently
        stops resolving.
        """
        for name, expected in (("लखनऊ", "Lucknow"), ("मेरठ", "Meerut"),
                               ("आगरा", "Agra"), ("इंदौर", "Indore"),
                               ("ग्वालियर", "Gwalior"), ("सूरत", "Surat"),
                               ("मैसूरु", "Mysuru"), ("बेलगावी", "Belagavi"),
                               ("रीवा", "Rewa")):
            self.assertEqual(bhashini.exonym(name), expected, name)

    def test_an_unknown_name_has_no_exonym(self):
        self.assertIsNone(bhashini.exonym("नरहरपुर"))
        self.assertIsNone(bhashini.exonym(""))

    def test_surrounding_whitespace_does_not_defeat_the_lookup(self):
        self.assertEqual(bhashini.exonym("  लखनऊ  "), "Lucknow")

    def test_the_table_is_valid_json_with_no_empty_targets(self):
        with open(os.path.join(ROOT, "backend", "data", "place_exonyms.json"),
                  encoding="utf-8") as fh:
            payload = json.load(fh)
        table = payload["exonyms"]
        self.assertGreater(len(table), 10)
        for key, value in table.items():
            self.assertTrue(key.strip(), "empty key")
            self.assertTrue(value.strip(), f"empty target for {key}")
            self.assertTrue(bhashini.is_indic(key), f"{key} is not Indic script")
            self.assertFalse(bhashini.is_indic(value), f"{value} is not Latin")


class TestCandidateOrdering(CacheIsolated):

    def test_the_exonym_is_offered_before_any_transliteration(self):
        bhashini.AVAILABLE = False      # no network, exonym only
        self.assertEqual(bhashini.candidates("लखनऊ"), ["Lucknow"])

    def test_cached_transliterations_are_used_without_a_network_call(self):
        bhashini.AVAILABLE = False
        bhashini._cache[bhashini._cache_key("नरहरपुर", True)] = ["narharpur"]
        self.assertEqual(bhashini.candidates("नरहरपुर"), ["narharpur"])

    def test_exonym_leads_and_transliterations_follow(self):
        bhashini.AVAILABLE = False
        bhashini._cache[bhashini._cache_key("लखनऊ", True)] = ["lakhanau"]
        bhashini._cache[bhashini._cache_key("लखनऊ", False)] = ["lakhnau"]
        self.assertEqual(bhashini.candidates("लखनऊ"),
                         ["Lucknow", "lakhanau", "lakhnau"])

    def test_duplicate_candidates_collapse_but_keep_order(self):
        bhashini.AVAILABLE = False
        bhashini._cache[bhashini._cache_key("भोपाल", True)] = ["bhopal", "bhopal"]
        bhashini._cache[bhashini._cache_key("भोपाल", False)] = ["bhopal", "bhopaal"]
        self.assertEqual(bhashini.candidates("भोपाल"), ["bhopal", "bhopaal"])

    def test_an_empty_name_yields_nothing(self):
        self.assertEqual(bhashini.candidates(""), [])
        self.assertEqual(bhashini.candidates("   "), [])


class TestResolveToReference(CacheIsolated):
    """
    The point of the whole module: the reference data decides, not the model.
    """

    def setUp(self):
        super().setUp()
        bhashini.AVAILABLE = False
        bhashini._cache.update({
            bhashini._cache_key("नरहरपुर", True): ["narharpur"],
            bhashini._cache_key("जयपुर", True): ["jaipur"],
            bhashini._cache_key("कोईभीनहीं", True): ["koibhinahin"],
        })

    def test_a_name_the_authority_accepts_is_resolved(self):
        known = {"narharpur", "jaipur", "Lucknow"}
        out = bhashini.resolve_to_reference(["नरहरपुर", "जयपुर"], known.__contains__)
        self.assertEqual(out, {"नरहरपुर": "narharpur", "जयपुर": "jaipur"})

    def test_a_name_the_authority_rejects_is_left_alone(self):
        """
        A village the master does not know must keep the spelling the OCR read.
        Substituting a plausible transliteration would be exactly the confident
        wrong answer this project exists to avoid.
        """
        out = bhashini.resolve_to_reference(["कोईभीनहीं"], lambda c: False)
        self.assertEqual(out, {})

    def test_latin_input_is_skipped_entirely(self):
        calls = []

        def authority(candidate):
            calls.append(candidate)
            return True

        out = bhashini.resolve_to_reference(["Lucknow", "Kanpur Nagar"], authority)
        self.assertEqual(out, {})
        self.assertEqual(calls, [], "Latin names must not be transliterated")

    def test_the_exonym_wins_over_a_transliteration_the_authority_also_accepts(self):
        bhashini._cache[bhashini._cache_key("लखनऊ", True)] = ["lakhanau"]
        out = bhashini.resolve_to_reference(["लखनऊ"], lambda c: True)
        self.assertEqual(out, {"लखनऊ": "Lucknow"})

    def test_an_authority_that_raises_does_not_break_resolution(self):
        def flaky(candidate):
            if candidate == "narharpur":
                raise RuntimeError("master unavailable")
            return candidate == "jaipur"

        out = bhashini.resolve_to_reference(["नरहरपुर", "जयपुर"], flaky)
        self.assertEqual(out, {"जयपुर": "jaipur"})

    def test_empty_input_is_handled(self):
        self.assertEqual(bhashini.resolve_to_reference([], lambda c: True), {})
        self.assertEqual(bhashini.resolve_to_reference(["", None], lambda c: True), {})


class TestDegradation(CacheIsolated):

    def test_transliterate_returns_cached_entries_even_when_unavailable(self):
        bhashini.AVAILABLE = False
        bhashini._cache[bhashini._cache_key("पुणे", True)] = ["pune"]
        self.assertEqual(bhashini.transliterate(["पुणे"]), {"पुणे": ["pune"]})

    def test_transliterate_never_raises_without_credentials(self):
        bhashini.AVAILABLE = False
        self.assertEqual(bhashini.transliterate(["कोईभीनहीं"]), {})

    def test_capabilities_reports_a_reason_when_off(self):
        caps = bhashini.capabilities()
        self.assertIn("available", caps)
        if not caps["available"]:
            self.assertTrue(caps["reason"], "an unavailable service must say why")

    def test_unavailable_reason_names_the_missing_piece(self):
        reason = bhashini.unavailable_reason()
        if reason is not None:
            self.assertTrue(
                "BHASHINI_PROJECT_ID" in reason or "BHASHINI_CONSENT" in reason,
                reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
