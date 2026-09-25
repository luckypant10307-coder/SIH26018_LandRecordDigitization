#!/usr/bin/env python3
"""
Unit tests for backend/handwriting.py (handwriting detection).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHAT THESE TESTS DO AND DO NOT ESTABLISH
----------------------------------------
The "handwritten" lines below are SYNTHETIC - characters drawn with jittered
baselines, mixed sizes and mixed stroke weights. They prove that each feature
responds in the direction it is supposed to, and that the scoring logic and
its safeguards behave. They do NOT establish that this detects real Patwari
handwriting; nothing available in this repository can establish that, and the
module docstring says so. Treat these as tests of the mechanism, not
evidence of recall.

The safeguards, on the other hand, are fully testable and are where the real
risk lies:
  * OneSidedScoringTests - a line MORE regular than print must never be
    flagged. Scoring absolute deviation would flag the cleanest printed page
    in the corpus as handwritten.
  * ProfileTests - a near-constant feature must not produce a division that
    flags everything.

Run from anywhere with:
    python3 tests/test_handwriting.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import handwriting as hw  # noqa: E402

try:
    import cv2
    import numpy as np
    CV_OK = True
except Exception:                                    # pragma: no cover
    CV_OK = False


def printed_line(text="KHASRA 237/4 RAM PRASAD", scale=1.0):
    """A machine-set line: one baseline, one size, one stroke weight."""
    w, h = int(760 * scale), int(64 * scale)
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.putText(img, text, (int(8 * scale), int(44 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, 0,
                max(1, int(round(2 * scale))), cv2.LINE_AA)
    return img


def handwritten_line(text="KHASRA 237/4 RAM PRASAD", seed=0):
    """
    A synthetic hand-written line: every character gets its own baseline
    offset, size and stroke weight, and the gaps between them vary.
    """
    rng = np.random.default_rng(seed)
    img = np.full((64, 760), 255, dtype=np.uint8)
    x = 8
    for ch in text:
        if ch == " ":
            x += int(rng.integers(8, 26))
            continue
        scale = float(rng.uniform(0.65, 1.15))
        thickness = int(rng.integers(1, 4))
        dy = int(rng.integers(-7, 8))
        cv2.putText(img, ch, (x, 44 + dy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    scale, 0, thickness, cv2.LINE_AA)
        x += int(18 * scale) + int(rng.integers(0, 9))
        if x > 730:
            break
    return img


def profile_from(images):
    """
    A print profile fitted on SEGMENTS, matching how scoring works.

    Fitting on whole lines while scoring segments would compare each value
    against the statistics of a different kind of object.
    """
    samples = []
    for img in images:
        samples.extend(hw.segment_features(img))
    return hw.fit_profile(samples)


PRINTED_CORPUS = ("KHASRA 237/4 RAM PRASAD", "VILLAGE NARHARPUR LUCKNOW",
                  "AREA 1.2540 HECTARE", "MUTATION MUT-2023-004182",
                  "REGISTRATION REG-2019-77120", "SHARE 1/1 IRRIGATED",
                  "KHATA 1428 TEHSIL SADAR", "STATE UTTAR PRADESH")


def printed_form_row(gap=380, label="KHASRA NO", value="237/4 RAM"):
    """
    An ENTIRELY PRINTED form row: label, a wide blank, a typeset value.
    This is what most of an old khatauni looks like.
    """
    img = np.full((64, 760), 255, dtype=np.uint8)
    cv2.putText(img, label, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    cv2.putText(img, value, (8 + gap, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2,
                cv2.LINE_AA)
    return img


def mixed_form_row(seed=3, gap=330, label="KHASRA NO", value="237/4 RAM PRASAD"):
    """A printed label with a HANDWRITTEN entry - the real mixed-record case."""
    img = np.full((64, 760), 255, dtype=np.uint8)
    cv2.putText(img, label, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    x = gap
    for ch in value:
        if ch == " ":
            x += int(rng.integers(8, 26))
            continue
        scale = float(rng.uniform(0.65, 1.15))
        thickness = int(rng.integers(1, 4))
        dy = int(rng.integers(-7, 8))
        cv2.putText(img, ch, (x, 44 + dy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    scale, 0, thickness, cv2.LINE_AA)
        x += int(18 * scale) + int(rng.integers(0, 9))
        if x > 730:
            break
    return img


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class FeatureTests(unittest.TestCase):
    def test_a_crop_too_small_to_judge_returns_none(self):
        self.assertIsNone(hw.line_features(np.full((6, 6), 255, np.uint8)))

    def test_a_blank_crop_returns_none(self):
        """No ink is not 'perfectly regular print', it is nothing to measure."""
        self.assertIsNone(hw.line_features(np.full((40, 400), 255, np.uint8)))

    def test_all_features_are_produced_for_a_printed_line(self):
        features = hw.line_features(printed_line())
        self.assertIsNotNone(features)
        for name in hw.FEATURE_NAMES:
            self.assertIn(name, features)

    def test_features_are_scale_free(self):
        """
        A profile fitted at one scan resolution has to apply at another, so
        every feature is a ratio. Doubling the rendering must not move them
        much - an absolute pixel measurement would make the profile a
        property of the scanner instead of of print.
        """
        small = hw.line_features(printed_line(scale=1.0))
        large = hw.line_features(printed_line(scale=2.0))
        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        for name in ("stroke_width_cv", "height_cv"):
            self.assertLess(abs(small[name] - large[name]), 0.25,
                            f"{name} changed too much with scale")

    def test_a_rule_line_is_not_counted_as_a_character(self):
        """
        A component spanning the crop is a table border, not a glyph.
        Counting it would wreck the height statistics of every ruled row on a
        land-record form - which is most of them.
        """
        img = printed_line()
        cv2.line(img, (0, 60), (img.shape[1] - 1, 60), 0, 2)
        features = hw.line_features(img)
        self.assertIsNotNone(features)
        self.assertLess(features["height_cv"], 0.6)


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class DirectionTests(unittest.TestCase):
    """
    Each feature must move the right way on synthetic handwriting. This is a
    test of the mechanism, NOT evidence about real handwriting.
    """

    def setUp(self):
        self.printed = hw.line_features(printed_line())
        self.written = hw.line_features(handwritten_line())
        self.assertIsNotNone(self.printed)
        self.assertIsNotNone(self.written)

    def test_baseline_is_less_straight(self):
        self.assertGreater(self.written["baseline_residual"],
                           self.printed["baseline_residual"])

    def test_character_heights_vary_more(self):
        self.assertGreater(self.written["height_cv"], self.printed["height_cv"])

    def test_synthetic_handwriting_is_flagged_against_a_print_profile(self):
        profile = profile_from([printed_line(t) for t in PRINTED_CORPUS])
        verdict = hw.score(self.written, profile)
        self.assertIsNotNone(verdict)
        self.assertTrue(verdict["is_handwriting_suspected"],
                        f"not flagged: {verdict}")

    def test_the_explanation_names_the_feature_that_stood_out(self):
        profile = profile_from([printed_line(t) for t in PRINTED_CORPUS])
        verdict = hw.score(self.written, profile)
        text = hw.explain(verdict)
        self.assertIn(verdict["worst_feature"], text)
        self.assertIn("unreliable", text)

    def test_no_explanation_when_nothing_is_suspected(self):
        self.assertEqual(hw.explain({"is_handwriting_suspected": False}), "")


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class MixedRecordTests(unittest.TestCase):
    """
    Old land records are printed FORMS with handwritten ENTRIES, frequently on
    the same row. These are the tests for that case, and the first one is a
    regression for a real false positive.
    """

    def setUp(self):
        self.profile = profile_from([printed_line(t) for t in PRINTED_CORPUS])

    def test_a_row_splits_at_the_label_to_value_gap(self):
        segments = hw.segment_features(printed_form_row())
        self.assertGreaterEqual(len(segments), 2,
                                "label and value should be separate text runs")

    def test_an_entirely_printed_form_row_is_not_flagged(self):
        """
        Regression. Scored as ONE line, an all-printed form row was flagged as
        handwriting at 9-15 standard deviations - purely because the wide
        blank between label and value made its spacing look wildly irregular.
        Land-record forms are made of rows like this, so that false positive
        would have swamped the review queue with correctly-read printed
        fields.
        """
        for gap in (200, 300, 420, 520):
            verdict = hw.inspect_line(printed_form_row(gap=gap), self.profile)
            self.assertIsNotNone(verdict)
            self.assertFalse(verdict["is_handwriting_suspected"],
                             f"printed row with a {gap}px gap was flagged: "
                             f"{verdict['worst_feature']} z={verdict['worst_z']}")

    def test_a_handwritten_entry_beside_a_printed_label_is_flagged(self):
        verdict = hw.inspect_line(mixed_form_row(), self.profile)
        self.assertIsNotNone(verdict)
        self.assertTrue(verdict["is_handwriting_suspected"],
                        f"mixed row not flagged: {verdict}")

    def test_the_flagged_region_is_the_entry_not_the_label(self):
        """
        The whole point of segmenting: say WHICH part of the row is suspect,
        so it can be lined up with the value the extractor actually read.
        """
        verdict = hw.inspect_line(mixed_form_row(gap=330), self.profile)
        self.assertGreater(verdict["x0"], 200,
                           "the printed label on the left was blamed")

    def test_not_every_run_on_a_mixed_row_is_condemned(self):
        """The printed label must still be recognised as printed."""
        verdict = hw.inspect_line(mixed_form_row(), self.profile)
        self.assertLess(verdict["flagged_segments"], verdict["segments"])

    def test_a_single_run_line_still_works(self):
        verdict = hw.inspect_line(printed_line("STATE UTTAR PRADESH"), self.profile)
        self.assertIsNotNone(verdict)
        self.assertFalse(verdict["is_handwriting_suspected"])


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class SegmentSplitTests(unittest.TestCase):
    def test_no_boxes_gives_no_segments(self):
        self.assertEqual(hw.split_segments([]), [])

    def test_one_box_is_one_segment(self):
        box = (0.0, 0.0, 10.0, 20.0, 5.0)
        self.assertEqual(hw.split_segments([box]), [[box]])

    def test_evenly_spaced_boxes_stay_one_segment(self):
        boxes = [(float(i * 22), 0.0, 14.0, 20.0, float(i * 22 + 7))
                 for i in range(8)]
        self.assertEqual(len(hw.split_segments(boxes)), 1)

    def test_a_wide_blank_creates_a_break(self):
        left = [(float(i * 22), 0.0, 14.0, 20.0, float(i * 22 + 7)) for i in range(5)]
        right = [(float(400 + i * 22), 0.0, 14.0, 20.0, float(400 + i * 22 + 7))
                 for i in range(5)]
        self.assertEqual(len(hw.split_segments(left + right)), 2)


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class OneSidedScoringTests(unittest.TestCase):
    """
    Every feature measures IRREGULARITY, so only deviations ABOVE the print
    mean are evidence of handwriting. A line that is more regular than the
    printed average is unusually clean print.
    """

    PROFILE = {
        "version": 1, "lines": 100,
        "features": {
            "stroke_width_cv": {"mean": 0.20, "std": 0.02, "observed_std": 0.02},
            "baseline_residual": {"mean": 0.10, "std": 0.01, "observed_std": 0.01},
            "height_cv": {"mean": 0.25, "std": 0.03, "observed_std": 0.03},
            "gap_cv": {"mean": 1.00, "std": 0.20, "observed_std": 0.20},
        },
    }

    def test_a_more_regular_line_than_print_is_not_flagged(self):
        spotless = {"stroke_width_cv": 0.01, "baseline_residual": 0.001,
                    "height_cv": 0.01, "gap_cv": 0.05}
        verdict = hw.score(spotless, self.PROFILE)
        self.assertFalse(verdict["is_handwriting_suspected"],
                         "unusually clean print was flagged as handwriting")
        self.assertLess(verdict["worst_z"], 0.0)

    def test_a_less_regular_line_is_flagged(self):
        messy = {"stroke_width_cv": 0.60, "baseline_residual": 0.40,
                 "height_cv": 0.80, "gap_cv": 3.00}
        self.assertTrue(hw.score(messy, self.PROFILE)["is_handwriting_suspected"])

    def test_the_threshold_is_respected(self):
        borderline = {"stroke_width_cv": 0.20 + 3 * 0.02,
                      "baseline_residual": 0.10, "height_cv": 0.25, "gap_cv": 1.0}
        self.assertFalse(hw.score(borderline, self.PROFILE, z_threshold=4.0)
                         ["is_handwriting_suspected"])
        self.assertTrue(hw.score(borderline, self.PROFILE, z_threshold=2.5)
                        ["is_handwriting_suspected"])

    def test_per_feature_z_scores_are_reported(self):
        verdict = hw.score({"stroke_width_cv": 0.30, "baseline_residual": 0.10,
                            "height_cv": 0.25, "gap_cv": 1.0}, self.PROFILE)
        self.assertAlmostEqual(verdict["z_scores"]["stroke_width_cv"], 5.0, places=1)
        self.assertEqual(verdict["worst_feature"], "stroke_width_cv")


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class ProfileTests(unittest.TestCase):
    def test_a_near_constant_feature_gets_a_floored_deviation(self):
        """
        Without a floor, a feature that barely varies across the training
        corpus divides by almost zero, and then EVERY line scores an enormous
        z-score - the detector would flag everything, most confidently on the
        cleanest pages.
        """
        samples = [{"stroke_width_cv": 0.2, "baseline_residual": 0.1,
                    "height_cv": 0.25, "gap_cv": 1.0} for _ in range(20)]
        profile = hw.fit_profile(samples)
        stats = profile["features"]["stroke_width_cv"]
        self.assertEqual(stats["observed_std"], 0.0)
        self.assertGreater(stats["std"], 0.0)
        # And a normal line must therefore NOT be flagged.
        verdict = hw.score({"stroke_width_cv": 0.21, "baseline_residual": 0.1,
                            "height_cv": 0.25, "gap_cv": 1.0}, profile)
        self.assertFalse(verdict["is_handwriting_suspected"])

    def test_round_trip_through_disk(self):
        profile = hw.fit_profile([{"stroke_width_cv": 0.2, "baseline_residual": 0.1,
                                   "height_cv": 0.25, "gap_cv": 1.0}] * 12)
        path = os.path.join(tempfile.mkdtemp(prefix="hw_"), "p.json")
        hw.save_profile(profile, path)
        back = hw.load_profile(path)
        self.assertEqual(back["lines"], profile["lines"])
        self.assertIn("stroke_width_cv", back["features"])

    def test_a_missing_profile_disables_detection(self):
        path = os.path.join(tempfile.gettempdir(), "definitely_absent_hw.json")
        self.assertIsNone(hw.load_profile(path))
        self.assertFalse(hw.available(path))

    def test_a_malformed_profile_is_rejected(self):
        path = os.path.join(tempfile.mkdtemp(prefix="hw_"), "bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{ not json")
        self.assertIsNone(hw.load_profile(path))

    def test_a_profile_without_features_is_rejected(self):
        path = os.path.join(tempfile.mkdtemp(prefix="hw_"), "empty.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "features": {}}, fh)
        self.assertIsNone(hw.load_profile(path))

    def test_scoring_without_a_profile_returns_none_not_a_verdict(self):
        """
        None means "cannot judge". Returning a negative verdict instead would
        silently assert every line is printed, which is the confident-wrong
        answer this project refuses to give.
        """
        self.assertIsNone(hw.score({"stroke_width_cv": 9.0}, {"features": {}}))


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class CropTests(unittest.TestCase):
    def test_a_box_is_padded_vertically(self):
        page = np.full((400, 800), 255, np.uint8)
        crop = hw.crop_line(page, (100, 100, 700, 140))
        self.assertGreater(crop.shape[0], 40, "no vertical margin was added")

    def test_padding_is_clamped_to_the_page(self):
        page = np.full((400, 800), 255, np.uint8)
        crop = hw.crop_line(page, (0, 0, 800, 40))
        self.assertLessEqual(crop.shape[0], 400)
        self.assertLessEqual(crop.shape[1], 800)

    def test_a_degenerate_box_returns_none(self):
        page = np.full((400, 800), 255, np.uint8)
        self.assertIsNone(hw.crop_line(page, (100, 100, 100, 100)))
        self.assertIsNone(hw.crop_line(page, (100, 100, 105, 103)))

    def test_no_page_returns_none(self):
        self.assertIsNone(hw.crop_line(None, (0, 0, 10, 10)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
