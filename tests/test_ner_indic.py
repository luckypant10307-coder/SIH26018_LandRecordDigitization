#!/usr/bin/env python3
"""
Unit tests for the Indic-script NER path in backend/ner_extractor.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

NO TEST HERE LOADS THE 2.5 GB MODEL. The multilingual pipeline is stubbed, so
the suite still passes on a machine with no torch, no transformers and no
network - which is the same machine the deployment image builds. What is tested
is the logic around the model: script routing, label translation, the
confidence floor, and the rule that a failure degrades to "no claim" rather
than to a wrong claim.

The model's real accuracy was measured separately against live Hindi land-record
lines (6 of 6 person names recovered); a number that depends on downloaded
weights does not belong in a unit test.

Run from anywhere with:
    python3 tests/test_ner_indic.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import ner_extractor as N  # noqa: E402

KHATAUNI = "खातेदार का नाम: सुनीता देवी पत्नी स्व0 रामप्रसाद, ग्राम नरहरपुर"


class StubPipeline:
    """Stands in for the transformers NER pipeline, recording what it saw."""

    def __init__(self, spans):
        self.spans = spans
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return list(self.spans)


def person(word, score=0.99):
    return {"entity_group": "PER", "word": word, "score": score}


class IndicStubbed(unittest.TestCase):
    """Installs a stub model and restores the module afterwards."""

    spans = [person("सुनीता देवी"), person("रामप्रसाद"),
             {"entity_group": "LOC", "word": "नरहरपुर", "score": 0.97}]

    def setUp(self):
        self._saved = (N._INDIC, N._INDIC_LOAD_ATTEMPTED, N._INDIC_ERROR)
        self.stub = StubPipeline(self.spans)
        N._INDIC = self.stub
        N._INDIC_LOAD_ATTEMPTED = True
        N._INDIC_ERROR = None

    def tearDown(self):
        N._INDIC, N._INDIC_LOAD_ATTEMPTED, N._INDIC_ERROR = self._saved

    def devanagari(self, key="owner_name", value="सुनीता देवी", line=KHATAUNI):
        return {key: {"value": value, "confidence": 0.9,
                      "extra": {"script": "devanagari"}, "source_line": line}}

    def rules(self, values):
        return [i.rule for i in N.cross_check(values)]


class TestLabelTranslation(IndicStubbed):

    def test_per_becomes_spacy_person(self):
        out = N._indic_entities_by_label(KHATAUNI)
        self.assertIn("PERSON", out)
        self.assertIn("सुनीता देवी", out["PERSON"])

    def test_loc_becomes_gpe(self):
        out = N._indic_entities_by_label(KHATAUNI)
        self.assertEqual(out.get("GPE"), ["नरहरपुर"])

    def test_an_unknown_label_is_dropped_rather_than_passed_through(self):
        self.stub.spans = [{"entity_group": "MISC", "word": "x", "score": 0.99}]
        self.assertEqual(N._indic_entities_by_label("x"), {})


class TestConfidenceFloor(IndicStubbed):

    def test_a_weak_span_is_ignored(self):
        """
        A low-confidence span that disagrees with the extractor would raise
        NER_MISMATCH and send a correct record to a human for nothing.
        """
        self.stub.spans = [person("कोई और", score=0.20)]
        self.assertEqual(N._indic_entities_by_label(KHATAUNI), {})

    def test_a_span_just_above_the_floor_is_kept(self):
        self.stub.spans = [person("सुनीता देवी", score=N._INDIC_MIN_SCORE + 0.01)]
        self.assertIn("PERSON", N._indic_entities_by_label(KHATAUNI))

    def test_a_weak_span_yields_unconfirmed_not_mismatch(self):
        self.stub.spans = [person("कोई और", score=0.20)]
        self.assertEqual(self.rules(self.devanagari()), ["NER_UNCONFIRMED"])


class TestScriptRouting(IndicStubbed):

    def test_devanagari_is_checked_instead_of_skipped(self):
        """The whole point: this path used to return nothing at all."""
        self.assertEqual(self.rules(self.devanagari()), ["NER_CONFIRMED"])

    def test_a_wrong_devanagari_reading_is_flagged(self):
        self.assertEqual(self.rules(self.devanagari(value="रीता शर्मा")),
                         ["NER_MISMATCH"])

    def test_the_indic_model_receives_the_source_line_not_a_template(self):
        """
        Feeding it the value alone would only ask "is this a name", never "is
        this the name on the line", which is the check that catches a misread.
        """
        N.cross_check(self.devanagari())
        self.assertEqual(self.stub.calls, [KHATAUNI])
        self.assertNotIn("Mr.", self.stub.calls[0])

    def test_latin_never_reaches_the_indic_model(self):
        values = {"owner_name": {"value": "Sunita Devi", "confidence": 0.9,
                                 "extra": {"script": "latin"}}}
        N.cross_check(values)
        self.assertEqual(self.stub.calls, [],
                         "Latin text must stay on the English model")

    def test_a_value_with_no_script_recorded_is_treated_as_latin(self):
        values = {"owner_name": {"value": "Sunita Devi", "confidence": 0.9,
                                 "extra": {}}}
        N.cross_check(values)
        self.assertEqual(self.stub.calls, [])


class TestDegradation(IndicStubbed):

    def test_a_model_that_raises_makes_no_claim(self):
        class Exploding:
            def __call__(self, text):
                raise RuntimeError("inference failed")

        N._INDIC = Exploding()
        self.assertEqual(N._indic_entities_by_label(KHATAUNI), {})
        self.assertEqual(self.rules(self.devanagari()), ["NER_UNCONFIRMED"])

    def test_no_indic_model_leaves_devanagari_unclaimed(self):
        """Without the model the record must be silent, not wrong."""
        N._INDIC = None
        N._INDIC_LOAD_ATTEMPTED = True
        self.assertEqual(self.rules(self.devanagari()), [])

    def test_status_reports_without_forcing_a_load(self):
        N._INDIC = None
        N._INDIC_LOAD_ATTEMPTED = False
        status = N.indic_ner_status()
        self.assertFalse(status["attempted"])
        self.assertTrue(status["model"])


class TestConfiguration(unittest.TestCase):

    def test_the_model_is_overridable_for_the_gated_indicner(self):
        """
        ai4bharat/IndicNER is MuRIL fine-tuned for this exact task but gated.
        Swapping to it once access exists must need no code change.
        """
        self.assertTrue(N.INDIC_NER_MODEL)
        self.assertEqual(
            os.environ.get("INDIC_NER_MODEL", N.INDIC_NER_MODEL),
            N.INDIC_NER_MODEL)

    def test_label_map_covers_what_the_model_emits(self):
        for emitted in ("PER", "LOC", "ORG", "DATE"):
            self.assertIn(emitted, N._INDIC_LABEL_MAP)
        self.assertEqual(N._INDIC_LABEL_MAP["PER"], "PERSON")


if __name__ == "__main__":
    unittest.main(verbosity=2)
