#!/usr/bin/env python3
"""
Unit tests for backend/llm_extractor.py (optional, off-by-default LLM
field-suggestion helper).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

No test here ever makes a real network call - every provider call is
monkeypatched. Enabling this feature for real sends document text to a
third-party API, which is exactly why the test suite must never do it
implicitly just by importing this module, on any machine, with or without
an API key sitting in the environment.

Run from anywhere with:
    python3 tests/test_llm_extractor.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import llm_extractor as llm  # noqa: E402


class ActivationGateTests(unittest.TestCase):
    """The two-flag gate is the whole safety story here - it must never
    activate on an API key alone."""

    def test_api_key_without_consent_is_unavailable(self):
        old_key, old_consent = llm.ANTHROPIC_API_KEY, llm.CONSENT_GIVEN
        try:
            llm.ANTHROPIC_API_KEY = "sk-fake-for-test"
            llm.CONSENT_GIVEN = False
            self.assertIsNone(llm._provider())
        finally:
            llm.ANTHROPIC_API_KEY, llm.CONSENT_GIVEN = old_key, old_consent

    def test_consent_without_any_key_is_unavailable(self):
        old_a, old_o, old_consent = llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY, llm.CONSENT_GIVEN
        try:
            llm.ANTHROPIC_API_KEY = None
            llm.OPENAI_API_KEY = None
            llm.CONSENT_GIVEN = True
            self.assertIsNone(llm._provider())
        finally:
            llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY, llm.CONSENT_GIVEN = old_a, old_o, old_consent

    def test_both_flags_present_selects_anthropic_first(self):
        old_a, old_o, old_consent = llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY, llm.CONSENT_GIVEN
        try:
            llm.ANTHROPIC_API_KEY = "sk-fake"
            llm.OPENAI_API_KEY = "sk-fake-2"
            llm.CONSENT_GIVEN = True
            self.assertEqual(llm._provider(), "anthropic")
        finally:
            llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY, llm.CONSENT_GIVEN = old_a, old_o, old_consent

    def test_nvidia_key_alone_selects_nvidia(self):
        old_a, old_o, old_n, old_consent = (llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY,
                                            llm.NVIDIA_API_KEY, llm.CONSENT_GIVEN)
        try:
            llm.ANTHROPIC_API_KEY = None
            llm.OPENAI_API_KEY = None
            llm.NVIDIA_API_KEY = "nvapi-fake"
            llm.CONSENT_GIVEN = True
            self.assertEqual(llm._provider(), "nvidia")
        finally:
            (llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY,
             llm.NVIDIA_API_KEY, llm.CONSENT_GIVEN) = old_a, old_o, old_n, old_consent

    def test_nvidia_key_without_consent_is_unavailable(self):
        old_n, old_consent = llm.NVIDIA_API_KEY, llm.CONSENT_GIVEN
        try:
            llm.NVIDIA_API_KEY = "nvapi-fake"
            llm.CONSENT_GIVEN = False
            self.assertIsNone(llm._provider())
        finally:
            llm.NVIDIA_API_KEY, llm.CONSENT_GIVEN = old_n, old_consent

    def test_suggest_fields_makes_no_call_when_unavailable(self):
        old_a, old_o, old_consent = llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY, llm.CONSENT_GIVEN
        called = []
        real_call = llm._call_anthropic
        llm._call_anthropic = lambda prompt: called.append(1) or "{}"
        try:
            llm.ANTHROPIC_API_KEY = None
            llm.OPENAI_API_KEY = None
            llm.CONSENT_GIVEN = False
            result = llm.suggest_fields("some document text", ["owner_name"])
            self.assertEqual(result, {})
            self.assertEqual(called, [])
        finally:
            llm._call_anthropic = real_call
            llm.ANTHROPIC_API_KEY, llm.OPENAI_API_KEY, llm.CONSENT_GIVEN = old_a, old_o, old_consent


class JsonExtractionTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(llm._extract_json('{"khasra_number": "213/1"}'),
                         {"khasra_number": "213/1"})

    def test_json_wrapped_in_prose(self):
        text = 'Sure, here you go:\n{"owner_name": "Ram Lal"}\nHope that helps.'
        self.assertEqual(llm._extract_json(text), {"owner_name": "Ram Lal"})

    def test_unparsable_text_returns_none(self):
        self.assertIsNone(llm._extract_json("not json at all"))

    def test_empty_text_returns_none(self):
        self.assertIsNone(llm._extract_json(""))
        self.assertIsNone(llm._extract_json(None))


class GroundingTests(unittest.TestCase):
    """
    Regression tests built from a REAL observed fabrication, not an invented
    scenario. On the first end-to-end run against a degraded Madhya Pradesh
    scan, the model returned a fluent, complete, entirely fictional Rajasthan
    land record while the document's real values sat in the prompt it was
    given. These are the exact strings from both that failing run and the
    run where the same model read the same document correctly.
    """

    # Trimmed from the actual OCR output of scan_02_khasra_mp_hindi_faded.png,
    # OCR noise ('§', mangled labels, Devanagari digits) left intact on purpose.
    OCR_TEXT = (
        "खाता क्रमांक + ९०७३\n"
        "खसर। करमाकि ; §42/4\n"
        "Yc का नाम : सुनीता बाई\n"
        "Rebel : 0.8090 हेक्टेयर\n"
        "ग्राम : बरखेडी\n"
        "जलि : Bhopal\n"
        "राज्य : Madhya Pradesh\n"
    )

    def test_fabricated_values_are_rejected(self):
        for bogus in ["Khasra No.  1234", "Shri Ram Kumar", "District - Alwar",
                      "State - Rajasthan", "Mutation No.  1111"]:
            self.assertFalse(llm.is_grounded(bogus, self.OCR_TEXT),
                             f"fabrication {bogus!r} should not be grounded")

    def test_genuine_values_are_accepted(self):
        for real in ["42/4", "सुनीता बाई", "0.8090 हेक्टेयर", "बरखेडी",
                     "Bhopal", "Madhya Pradesh"]:
            self.assertTrue(llm.is_grounded(real, self.OCR_TEXT),
                            f"genuine value {real!r} should be grounded")

    def test_indic_digits_are_bridged(self):
        """A model that helpfully converts '९०७३' to '9073' is doing the right
        thing and must not be punished for it by a naive substring check."""
        self.assertTrue(llm.is_grounded("9073", self.OCR_TEXT))

    def test_whitespace_differences_do_not_break_grounding(self):
        self.assertTrue(llm.is_grounded("Madhya   Pradesh", self.OCR_TEXT))

    def test_empty_inputs_are_not_grounded(self):
        self.assertFalse(llm.is_grounded("", self.OCR_TEXT))
        self.assertFalse(llm.is_grounded("anything", ""))

    def test_suggest_fields_drops_ungrounded_values(self):
        old_a, old_c, old_call = llm.ANTHROPIC_API_KEY, llm.CONSENT_GIVEN, llm._call_anthropic
        try:
            llm.ANTHROPIC_API_KEY = "sk-fake"
            llm.OPENAI_API_KEY = None
            llm.CONSENT_GIVEN = True
            llm._call_anthropic = lambda prompt: (
                '{"owner_name": "Shri Ram Kumar", "district": "Bhopal"}')
            out = llm.suggest_fields(self.OCR_TEXT, ["owner_name", "district"])
            # 'Bhopal' is in the document; 'Shri Ram Kumar' is not.
            self.assertEqual(set(out.keys()), {"district"})
        finally:
            llm.ANTHROPIC_API_KEY, llm.CONSENT_GIVEN = old_a, old_c
            llm._call_anthropic = old_call


class SuggestFieldsParsingTests(unittest.TestCase):
    """Exercises suggest_fields()'s filtering logic with the provider call
    monkeypatched - never a real request."""

    def setUp(self):
        self._old_a = llm.ANTHROPIC_API_KEY
        self._old_o = llm.OPENAI_API_KEY
        self._old_c = llm.CONSENT_GIVEN
        self._old_call = llm._call_anthropic
        llm.ANTHROPIC_API_KEY = "sk-fake-for-test"
        llm.OPENAI_API_KEY = None
        llm.CONSENT_GIVEN = True

    def tearDown(self):
        llm.ANTHROPIC_API_KEY = self._old_a
        llm.OPENAI_API_KEY = self._old_o
        llm.CONSENT_GIVEN = self._old_c
        llm._call_anthropic = self._old_call

    def test_only_requested_and_present_fields_are_returned(self):
        # Source text must actually contain the value now - see GroundingTests.
        source = "Owner Name : Ram Lal\nFather Name :\n"
        llm._call_anthropic = lambda prompt: (
            '{"owner_name": "Ram Lal", "father_name": "", "unrequested_field": "x"}')
        out = llm.suggest_fields(source, ["owner_name", "father_name", "area"])
        self.assertEqual(set(out.keys()), {"owner_name"})
        self.assertEqual(out["owner_name"]["suggested_value"], "Ram Lal")
        self.assertTrue(out["owner_name"]["source"].startswith("llm:"))

    def test_transport_failure_propagates_rather_than_looking_like_no_results(self):
        """Regression test for a real bug found by end-to-end testing, not by
        this suite: a 20s timeout on a full 17-field request was being
        swallowed into {}, so a failed call and a model that honestly found
        nothing were indistinguishable and the reviewer saw neither. The
        caller needs the exception to raise LLM_SUGGESTION_FAILED."""
        def boom(prompt):
            raise TimeoutError("The read operation timed out")
        llm._call_anthropic = boom
        with self.assertRaises(TimeoutError):
            llm.suggest_fields("doc text", ["owner_name"])

    def test_non_json_response_yields_empty_result(self):
        llm._call_anthropic = lambda prompt: "I could not find that field."
        out = llm.suggest_fields("doc text", ["owner_name"])
        self.assertEqual(out, {})

    def test_no_field_keys_makes_no_call(self):
        called = []
        llm._call_anthropic = lambda prompt: called.append(1) or "{}"
        out = llm.suggest_fields("doc text", [])
        self.assertEqual(out, {})
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
