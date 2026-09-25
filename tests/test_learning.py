#!/usr/bin/env python3
"""
Unit tests for backend/learning.py (the AI-driven learning loop).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Two of these classes are regression tests for defects that made the loop
inert or unsafe rather than merely imperfect:

  * ConfusionRepairTests - the mined character-confusion map was written to
    the model file and shown in the Learning tab, but apply_model() only ever
    read `aliases` and `calibration`, so no confusion ever changed anything.
  * StatusRecomputeTests - apply_model() deflated confidence but left the
    status extract_fields() had already stamped, so recalibration (which the
    module docstring calls the highest-value part of the loop) never actually
    moved a field into the review queue.

Run from anywhere with:
    python3 tests/test_learning.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import field_extractor as fe  # noqa: E402
import learning  # noqa: E402


def make_field(key, value, confidence=0.95, status="extracted"):
    spec = fe.FIELD_BY_KEY[key]
    return fe.ExtractedField(key=key, display=spec.display, value=value,
                             confidence=confidence, status=status)


def model(confusions=(), aliases=(), calibration=()):
    """A model dict with active_rules set consistently, as build_model would."""
    return {
        "version": 1, "samples": 99,
        "confusions": list(confusions),
        "aliases": list(aliases),
        "calibration": list(calibration),
        "active_rules": (sum(1 for c in confusions if c.get("auto_apply"))
                         + sum(1 for a in aliases if a.get("auto_apply"))
                         + len(calibration)),
    }


CONF_4_TO_1 = {"from": "4", "to": "1", "support": 6,
               "field_kinds": ["number"], "auto_apply": True}


class ConfusionVariantTests(unittest.TestCase):
    def test_one_substitution_at_a_time(self):
        """
        The whole safety argument. A learned confusion says the engine
        sometimes reads 4 where the truth is 1; it does not say WHICH 4 was
        wrong. Replacing every occurrence turns khata 4474 into 1171 and
        invents a parcel.
        """
        got = set(learning._confusion_variants("4474", [CONF_4_TO_1]))
        self.assertEqual(got, {"1474", "4174", "4471"})
        self.assertNotIn("1171", got)

    def test_rules_not_marked_auto_apply_are_ignored(self):
        rule = dict(CONF_4_TO_1, auto_apply=False)
        self.assertEqual(learning._confusion_variants("442", [rule]), {})

    def test_value_without_the_character_yields_nothing(self):
        self.assertEqual(learning._confusion_variants("237/9", [CONF_4_TO_1]), {})


class ConfusionRepairTests(unittest.TestCase):
    """
    Regression: mined confusions were never applied by apply_model().
    """

    def _apply(self, value, known, key="khata_number"):
        f = make_field(key, value)
        applied = learning.apply_model(
            [f], model(confusions=[CONF_4_TO_1]),
            corroborate=lambda k, v: v in known)
        return f, applied

    def test_a_single_corroborated_candidate_is_repaired(self):
        f, applied = self._apply("4474", {"4471"})
        self.assertEqual(f.value, "4471")
        self.assertEqual([a["type"] for a in applied], ["confusion_repair"])

    def test_a_repair_is_always_sent_to_a_human(self):
        """A learned guess about a plot number is never silently trusted."""
        f, _ = self._apply("4474", {"4471"})
        self.assertEqual(f.status, "needs_review")

    def test_the_note_explains_the_evidence(self):
        f, _ = self._apply("4474", {"4471"})
        note = " ".join(f.notes)
        self.assertIn("4474", note)
        self.assertIn("4471", note)
        self.assertIn("6", note)          # the support count

    def test_ambiguity_is_never_resolved_by_guessing(self):
        """Two equally plausible repairs must leave the value alone."""
        f, applied = self._apply("4474", {"4471", "1474"})
        self.assertEqual(f.value, "4474")
        self.assertEqual(f.status, "needs_review")
        self.assertEqual([a["type"] for a in applied], ["confusion_ambiguous"])

    def test_no_candidate_corroborated_changes_nothing(self):
        f, applied = self._apply("4474", set())
        self.assertEqual(f.value, "4474")
        self.assertEqual(applied, [])

    def test_a_value_that_already_checks_out_is_not_second_guessed(self):
        f, applied = self._apply("4471", {"4471", "1471"})
        self.assertEqual(f.value, "4471")
        self.assertEqual(applied, [])

    def test_without_an_authority_no_repair_is_attempted(self):
        f = make_field("khata_number", "4474")
        applied = learning.apply_model([f], model(confusions=[CONF_4_TO_1]))
        self.assertEqual(f.value, "4474")
        self.assertEqual(applied, [])

    def test_repair_is_restricted_to_identifier_fields(self):
        """
        Confusion repair is only defensible where the alphabet is closed and
        an authority exists. A person's name has neither.
        """
        f = make_field("owner_name", "Vitthat")
        learning.apply_model([f], model(confusions=[CONF_4_TO_1]),
                             corroborate=lambda k, v: True)
        self.assertEqual(f.value, "Vitthat")


class StatusRecomputeTests(unittest.TestCase):
    """
    Regression: recalibration changed the number but not where the field went.
    """

    CALIB = [{"field_key": "owner_name", "display": "Landowner Name",
              "reviewed": 40, "stated_confidence": 0.95,
              "observed_precision": 0.60, "multiplier": 0.60,
              "direction": "deflate"}]

    def test_deflated_confidence_moves_a_field_into_review(self):
        f = make_field("owner_name", "Someone", confidence=0.95, status="extracted")
        learning.apply_model([f], model(calibration=self.CALIB))
        self.assertLess(f.confidence, fe.REVIEW_THRESHOLD)
        self.assertEqual(f.status, "needs_review")

    def test_a_field_that_stays_above_threshold_is_left_alone(self):
        calib = [dict(self.CALIB[0], multiplier=1.0, observed_precision=0.95,
                      direction="stable")]
        f = make_field("owner_name", "Someone", confidence=0.99)
        learning.apply_model([f], model(calibration=calib))
        self.assertEqual(f.status, "extracted")

    def test_a_missing_field_is_never_resurrected(self):
        """'missing' means nothing was found; recalibration must not relabel it."""
        f = make_field("owner_name", None, confidence=0.0, status="missing")
        learning.apply_model([f], model(calibration=self.CALIB))
        self.assertEqual(f.status, "missing")

    def test_an_empty_model_does_nothing_at_all(self):
        f = make_field("owner_name", "Someone", confidence=0.10, status="extracted")
        applied = learning.apply_model([f], model())
        self.assertEqual(applied, [])
        # No rules means no opinion - the field keeps the status extraction
        # gave it rather than being re-judged by an untrained model.
        self.assertEqual(f.status, "extracted")


class MiningThresholdTests(unittest.TestCase):
    def _corrections(self, n, ai="442", human="142", key="khasra_number"):
        return [{"field_key": key, "ai_value": ai, "human_value": human}
                for _ in range(n)]

    def test_a_confusion_below_support_is_not_promoted(self):
        rows = self._corrections(learning.MIN_CONFUSION_SUPPORT - 1)
        self.assertEqual(learning.mine_confusions(rows), [])

    def test_a_confusion_at_support_is_promoted_and_auto_applies(self):
        rows = self._corrections(learning.MIN_CONFUSION_SUPPORT)
        got = learning.mine_confusions(rows)
        self.assertEqual(len(got), 1)
        self.assertEqual((got[0]["from"], got[0]["to"]), ("4", "1"))
        self.assertTrue(got[0]["auto_apply"])

    def test_a_confusion_learned_from_names_never_auto_applies(self):
        """Closed alphabet only: a letter swap has no authority to check it."""
        rows = self._corrections(learning.MIN_CONFUSION_SUPPORT,
                                 ai="Vitthat", human="Vitthal", key="owner_name")
        got = learning.mine_confusions(rows)
        self.assertTrue(got)
        self.assertFalse(any(c["auto_apply"] for c in got))

    def test_a_digit_confusion_survives_being_seen_in_prose_too(self):
        """
        Regression: `auto_apply` demanded that a confusion had been seen in
        NOTHING but numeric fields. On the sample corpus '4' -> '1' earned 14
        supporting corrections and was still refused, because the same misread
        also appeared once in a village name - so the repair path could never
        open. Where a rule was learned is not where it may be used.
        """
        rows = (self._corrections(6, ai="442", human="142", key="khasra_number")
                + self._corrections(1, ai="Bark4edi", human="Bark1edi", key="village"))
        got = {(c["from"], c["to"]): c for c in learning.mine_confusions(rows)}
        rule = got[("4", "1")]
        self.assertIn("number", rule["field_kinds"])
        self.assertIn("text", rule["field_kinds"])
        self.assertTrue(rule["auto_apply"])

    def test_aliases_need_support_and_agreement(self):
        rows = [{"field_key": "village", "ai_value": "Lucknov",
                 "human_value": "Lucknow"} for _ in range(learning.MIN_ALIAS_SUPPORT)]
        got = learning.mine_aliases(rows)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0]["auto_apply"])

    def test_a_contested_alias_does_not_auto_apply(self):
        rows = ([{"field_key": "village", "ai_value": "Lucknov",
                  "human_value": "Lucknow"} for _ in range(3)]
                + [{"field_key": "village", "ai_value": "Lucknov",
                    "human_value": "Lakhnau"} for _ in range(3)])
        got = learning.mine_aliases(rows)
        self.assertEqual(len(got), 1)
        self.assertFalse(got[0]["auto_apply"])

    def test_calibration_ignores_a_sample_too_small_to_mean_anything(self):
        stats = [{"field_key": "village", "reviewed": learning.MIN_CALIBRATION_SAMPLE - 1,
                  "precision": 0.2, "avg_conf": 0.95}]
        self.assertEqual(learning.calibrate(stats), [])

    def test_calibration_deflates_an_overconfident_field(self):
        stats = [{"field_key": "village", "reviewed": 40,
                  "precision": 0.50, "avg_conf": 0.95}]
        got = learning.calibrate(stats)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["direction"], "deflate")
        self.assertLess(got[0]["multiplier"], 1.0)

    def test_calibration_cannot_inflate_without_bound(self):
        """A field that looks perfect must not be handed unlimited confidence."""
        stats = [{"field_key": "village", "reviewed": 400,
                  "precision": 1.0, "avg_conf": 0.30}]
        got = learning.calibrate(stats)
        self.assertLessEqual(got[0]["multiplier"], 1.15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
