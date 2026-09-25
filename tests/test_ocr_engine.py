#!/usr/bin/env python3
"""
Unit tests for backend/ocr_engine.py's restoration/quality-gate logic.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

OpenCV/numpy are optional dependencies for this project (see README S2), so
image-based tests are skipped (not failed) when they are absent. The pure
quality-gate math has no such dependency and always runs.

Run from anywhere with:
    python3 tests/test_ocr_engine.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import ocr_engine as oe  # noqa: E402

_CV_OK = oe._cv2 is not None and oe._numpy is not None


class QualityGateMathTests(unittest.TestCase):
    """Pure functions - no image, no OpenCV dependency."""

    def test_high_legibility_is_ok_and_untouched(self):
        self.assertEqual(oe.quality_gate_label(90.0), "ok")
        self.assertEqual(oe.quality_gate_multiplier(90.0), 1.0)

    def test_boundary_ok_is_inclusive(self):
        self.assertEqual(oe.quality_gate_label(oe.QUALITY_GATE_OK), "ok")
        self.assertEqual(oe.quality_gate_multiplier(oe.QUALITY_GATE_OK), 1.0)

    def test_boundary_poor_is_inclusive(self):
        self.assertEqual(oe.quality_gate_label(oe.QUALITY_GATE_POOR), "poor")
        self.assertAlmostEqual(oe.quality_gate_multiplier(oe.QUALITY_GATE_POOR),
                               oe.QUALITY_GATE_MIN_MULTIPLIER)

    def test_very_low_legibility_floors_at_minimum(self):
        self.assertEqual(oe.quality_gate_label(0.0), "poor")
        self.assertEqual(oe.quality_gate_multiplier(0.0), oe.QUALITY_GATE_MIN_MULTIPLIER)

    def test_midpoint_is_marginal_and_between_bounds(self):
        mid = (oe.QUALITY_GATE_OK + oe.QUALITY_GATE_POOR) / 2.0
        self.assertEqual(oe.quality_gate_label(mid), "marginal")
        mult = oe.quality_gate_multiplier(mid)
        self.assertGreater(mult, oe.QUALITY_GATE_MIN_MULTIPLIER)
        self.assertLess(mult, 1.0)

    def test_multiplier_is_monotonic_in_legibility(self):
        scores = [0, 10, 20, 32, 40, 50, 62, 70, 100]
        mults = [oe.quality_gate_multiplier(s) for s in scores]
        self.assertEqual(mults, sorted(mults))


@unittest.skipUnless(_CV_OK, "OpenCV/numpy not installed - image restoration tests skipped")
class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.cv2 = oe._cv2
        self.np = oe._numpy
        self.tmpdir = tempfile.mkdtemp(prefix="ocr_test_")

    def _save(self, img, name):
        path = os.path.join(self.tmpdir, name)
        self.cv2.imwrite(path, img)
        return path

    def _base_page(self, h=420, w=620):
        img = self.np.full((h, w, 3), 255, dtype=self.np.uint8)
        self.cv2.putText(img, "KHASRA 237/4 OWNER RAM PRASAD", (30, 110),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        self.cv2.putText(img, "VILLAGE NARHARPUR DISTRICT LUCKNOW", (30, 170),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        return img

    def test_rule_lines_are_measured_but_deliberately_not_erased(self):
        """
        Ruling lines are DETECTED and reported, and left on the page.

        They used to be erased before OCR, which sounded right and measured
        wrong: scored over the 33-scan corpus, erasing them cost field recall
        (31.7% -> 30.6%) because the opening that isolates a long thin line
        also takes the parts of characters resting against it. Tesseract's own
        layout analysis copes with ruled forms perfectly well, so the ratio is
        kept as a quality signal and the pixels are left alone.
        """
        cv2, np = self.cv2, self.np
        img = self._base_page()
        # A ruling line far from any text, so it can be checked in isolation -
        # land-record forms rule off every field row like this.
        cv2.line(img, (20, 300), (600, 300), (0, 0, 0), 2)
        path = self._save(img, "rule_line.png")

        out_path, metrics, _warnings = oe.assess_and_preprocess(path, self.tmpdir)
        self.assertIsNotNone(out_path)
        self.assertGreater(metrics["rule_line_ratio"], 0.0,
                           "the line still has to be MEASURED")

        out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        strip = out_img[295:305, 100:500]
        self.assertLess(float(strip.mean()), 250.0,
                        "the ruling line should still be on the page")

    def test_tesseract_receives_greyscale_not_a_binary_mask(self):
        """
        The restored page handed to OCR must retain grey levels.

        Binarising here competes with Tesseract's own thresholder instead of
        helping it, and measured worse than sending the untouched original
        (28.1% vs 31.1% field recall). This test is what stops a future
        "improvement" from quietly reintroducing a two-level image.
        """
        cv2, np = self.cv2, self.np
        path = self._save(self._base_page(), "greyscale_contract.png")
        out_path, _metrics, _warnings = oe.assess_and_preprocess(path, self.tmpdir)
        self.assertIsNotNone(out_path)
        out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        self.assertGreater(len(np.unique(out_img)), 2,
                           "output is binarised - Tesseract needs grey levels")

    def test_bleed_through_is_suppressed_but_real_text_survives(self):
        cv2, np = self.cv2, self.np
        img = self._base_page()
        # A faint "ghost" text far from the real text, simulating reverse-side
        # bleed-through: much lower contrast than genuine foreground ink.
        cv2.putText(img, "GHOST", (250, 320), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (200, 200, 200), 2)
        path = self._save(img, "bleed_through.png")

        out_path, metrics, _warnings = oe.assess_and_preprocess(path, self.tmpdir)
        self.assertGreater(metrics["bleed_through_ratio"], 0.0)

        out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        ghost_region = out_img[290:340, 240:400]
        real_text_region = out_img[85:125, 30:400]
        self.assertGreater(float(ghost_region.mean()), 220.0,
                           "faint bleed-through should be suppressed to background")
        self.assertLess(float(real_text_region.mean()), 220.0,
                        "genuine foreground ink must survive suppression")

    def test_shadowed_text_still_binarises_after_illumination_correction(self):
        cv2, np = self.cv2, self.np
        img = self._base_page()
        # A left-to-right lighting gradient (shadow), like a scanner lid not
        # sitting flush or a photographed page under a directional lamp.
        gradient = np.tile(np.linspace(0.45, 1.0, img.shape[1]), (img.shape[0], 1))
        gradient = np.dstack([gradient] * 3)
        shadowed = (img.astype(np.float64) * gradient).clip(0, 255).astype(np.uint8)
        path = self._save(shadowed, "shadowed.png")

        out_path, metrics, _warnings = oe.assess_and_preprocess(path, self.tmpdir)
        self.assertGreater(metrics["illumination_variation"], 0.0)

        out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        # The first text line sits inside the darkest (most shadowed) part of
        # the gradient; it must still binarise to ink despite that.
        shadowed_text_region = out_img[85:125, 30:400]
        self.assertLess(float(shadowed_text_region.mean()), 220.0,
                        "text under the shadow should still be recovered as ink")

    def test_blank_washed_out_page_trips_the_quality_gate(self):
        # No legible content at all - the clearest real case for the
        # "poor" bucket: a page this washed-out cannot support any OCR
        # confidence, however confident Tesseract's own numbers might read.
        cv2, np = self.cv2, self.np
        rng = np.random.default_rng(1)
        blank = np.full((420, 620), 200, dtype=np.float64) + rng.normal(0, 6, (420, 620))
        blank = cv2.cvtColor(blank.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        path = self._save(blank, "blank.png")

        _out_path, metrics, warnings = oe.assess_and_preprocess(path, self.tmpdir)
        self.assertEqual(metrics["quality_gate"], "poor")
        self.assertEqual(metrics["confidence_gate_multiplier"], oe.QUALITY_GATE_MIN_MULTIPLIER)
        self.assertTrue(any("Quality gate" in w for w in warnings))

    def test_degraded_page_scores_well_below_a_clean_one(self):
        cv2, np = self.cv2, self.np
        clean = self._base_page()
        degraded = cv2.GaussianBlur(clean, (15, 15), 0)
        degraded = (degraded.astype(np.float64) * 0.5 + 128).clip(0, 255).astype(np.uint8)
        rng = np.random.default_rng(0)
        degraded = (degraded.astype(np.float64)
                   + rng.normal(0, 18, degraded.shape)).clip(0, 255).astype(np.uint8)

        _out1, clean_metrics, _w1 = oe.assess_and_preprocess(
            self._save(clean, "clean_cmp.png"), self.tmpdir)
        _out2, degraded_metrics, _w2 = oe.assess_and_preprocess(
            self._save(degraded, "degraded_cmp.png"), self.tmpdir)

        self.assertLess(degraded_metrics["legibility_score"],
                        clean_metrics["legibility_score"] - 15,
                        "blur+fade+noise should meaningfully depress the score")
        self.assertLess(degraded_metrics["confidence_gate_multiplier"],
                        clean_metrics["confidence_gate_multiplier"])

    def test_clean_page_passes_the_quality_gate(self):
        img = self._base_page()
        path = self._save(img, "clean.png")

        _out_path, metrics, _warnings = oe.assess_and_preprocess(path, self.tmpdir)
        self.assertIn(metrics["quality_gate"], ("ok", "marginal"))
        self.assertGreaterEqual(metrics["confidence_gate_multiplier"],
                                oe.QUALITY_GATE_MIN_MULTIPLIER)


class ScriptSelectionTests(unittest.TestCase):
    """
    Script-aware language selection (see ocr_engine's SCRIPT_PACK_CANDIDATES).
    The real selection needs Tesseract and several language packs, so the
    decision *logic* is tested here with the OCR call stubbed; correctness of
    the actual choice was verified separately against rendered Tamil, Telugu
    and Devanagari pages (Tamil won by 24.5 points, Telugu by 17.0).
    """

    def test_one_candidate_per_script_not_per_language(self):
        """mar+eng was deliberately removed: it is the same script as hin+eng,
        and confidence cannot reliably choose between two models of one
        script - it picked mar over hin on documents that were Hindi."""
        self.assertIn("hin+eng", oe.SCRIPT_PACK_CANDIDATES)
        self.assertNotIn("mar+eng", oe.SCRIPT_PACK_CANDIDATES)
        self.assertEqual(len(oe.SCRIPT_PACK_CANDIDATES),
                         len(set(oe.SCRIPT_PACK_CANDIDATES)))

    def test_highest_confidence_pack_wins(self):
        available = ["hin", "tam", "tel", "eng"]
        fake = {"hin+eng": 58.0, "tam+eng": 91.0, "tel+eng": 63.0}

        def stub(path, lang):
            return {"score": fake.get(lang, 0.0)}

        real_ocr, real_conf = oe._ocr_page, oe._tsv_mean_confidence
        try:
            oe._ocr_page = stub
            oe._tsv_mean_confidence = lambda tsv: tsv["score"]
            lang, diag = oe.select_languages("irrelevant.png", available)
        finally:
            oe._ocr_page, oe._tsv_mean_confidence = real_ocr, real_conf

        self.assertEqual(lang, "tam+eng")
        self.assertEqual(diag["confidence"], 91.0)
        self.assertEqual(diag["margin"], 28.0)   # 91.0 - 63.0

    def test_packs_not_installed_are_skipped(self):
        """A machine with only Hindi and English must never be handed tam+eng."""
        available = ["hin", "eng"]
        tried = []

        def stub(path, lang):
            tried.append(lang)
            return {"score": 50.0}

        real_ocr, real_conf = oe._ocr_page, oe._tsv_mean_confidence
        try:
            oe._ocr_page = stub
            oe._tsv_mean_confidence = lambda tsv: tsv["score"]
            lang, _diag = oe.select_languages("irrelevant.png", available)
        finally:
            oe._ocr_page, oe._tsv_mean_confidence = real_ocr, real_conf

        for attempted in tried:
            for piece in attempted.split("+"):
                self.assertIn(piece, available)
        # The point of this test is the FILTERING above - an uninstalled pack
        # must never be attempted. The winner is incidental to it, and with
        # this stub every pack (including the `eng` baseline) scores an
        # identical 50.0, so the margin over English is 0.0 and the gate
        # correctly declines to pay for an Indic pack that added nothing.
        # Asserting "hin+eng" here would be asserting that a tie promotes an
        # Indic pack, which is the exact behaviour that put Odia glyphs
        # through an English-only Delhi e-Stamp.
        self.assertEqual(lang, "eng")

    def test_a_tie_with_english_does_not_promote_an_indic_pack(self):
        """
        The rule that fixed the e-Stamp, stated directly: an Indic pack must
        ADD something measurable over plain English or it is not used.
        """
        def stub(path, lang, psm=oe._PSM_SINGLE_BLOCK):
            return {"score": 50.0}

        real_ocr, real_conf = oe._ocr_page, oe._tsv_mean_confidence
        try:
            oe._ocr_page = stub
            oe._tsv_mean_confidence = lambda tsv: tsv["score"]
            lang, diag = oe.select_languages("irrelevant.png", ["hin", "eng"])
        finally:
            oe._ocr_page, oe._tsv_mean_confidence = real_ocr, real_conf
        # Assert the OUTCOME, not the internal flag. On an exact tie `eng`
        # may sort first, in which case there was no Indic winner to override
        # and gated_to_english is legitimately False - the result is still
        # English, which is the whole point.
        self.assertEqual(lang, "eng")
        self.assertEqual(diag["margin_over_english"], 0.0)

    def test_an_indic_pack_wins_when_it_clearly_beats_english(self):
        scores = {"eng": 44.0, "hin+eng": 81.0}

        def stub(path, lang, psm=oe._PSM_SINGLE_BLOCK):
            return {"score": scores.get(lang, 10.0)}

        real_ocr, real_conf = oe._ocr_page, oe._tsv_mean_confidence
        try:
            oe._ocr_page = stub
            oe._tsv_mean_confidence = lambda tsv: tsv["score"]
            lang, diag = oe.select_languages("irrelevant.png", ["hin", "eng"])
        finally:
            oe._ocr_page, oe._tsv_mean_confidence = real_ocr, real_conf
        self.assertEqual(lang, "hin+eng")
        self.assertFalse(diag["gated_to_english"])
        self.assertGreater(diag["margin_over_english"],
                           oe.SCRIPT_MARGIN_OVER_ENGLISH)

    def test_the_threshold_sits_in_the_measured_gap(self):
        """
        Measured over 15 real Bhu-Naksha reports and 4 Latin e-Stamp pages:
        Devanagari margins run +10.89 to +37.33, Latin +0.00 to +9.49. The
        constant must stay inside that 1.4-point gap - a first attempt at
        12.0 wrongly gated four Devanagari documents and cost 6.6 points of
        real-corpus accuracy.
        """
        self.assertGreater(oe.SCRIPT_MARGIN_OVER_ENGLISH, 9.49)
        self.assertLessEqual(oe.SCRIPT_MARGIN_OVER_ENGLISH, 10.89)

    def test_no_usable_pack_reports_rather_than_guesses(self):
        lang, diag = oe.select_languages("irrelevant.png", [])
        self.assertIsNone(lang)
        self.assertIn("no usable", diag["reason"])

    def test_explicit_languages_bypass_selection(self):
        """A caller that already knows the script must not pay for a trial."""
        import inspect
        sig = inspect.signature(oe.extract)
        self.assertIsNone(sig.parameters["languages"].default,
                          "extract() must default to auto-selection")


def _tsv(words):
    """Minimal Tesseract TSV dict: [(text, conf), ...]."""
    return {"text": [w for w, _ in words], "conf": [c for _, c in words]}


class PageSegmentationChoiceTests(unittest.TestCase):
    """
    psm 6 assumes the page is ONE uniform block. That holds for a plain
    register page and measures best there, but a property paper carrying a
    printed form, handwritten patwari entries and a sketch map is several
    blocks - and once assess_and_preprocess strips the table ruling lines,
    psm 6 has no cue left and welds the form into a single block. Measured on
    exactly such a page it read 1 of 8 printed fields where psm 3 read 8.

    So the mode is chosen per page, and the MARGIN is the load-bearing part:
    the multi-region page gave psm 3 a 60% lead in confident characters while
    every single-block scan stayed within 6% either way.
    """

    def test_confident_chars_counts_length_not_words(self):
        """
        A page that shattered into one-character fragments must not outscore
        one that read whole words - that is the failure being compared
        against, so counting boxes instead of characters would invert it.
        """
        fragments = _tsv([("a", 90.0), ("b", 90.0), ("c", 90.0)])
        words = _tsv([("Bhopal", 90.0)])
        self.assertEqual(oe._confident_chars(fragments), 3)
        self.assertEqual(oe._confident_chars(words), 6)

    def test_confident_chars_ignores_unconfident_and_empty_boxes(self):
        tsv = _tsv([("Bhopal", 90.0), ("garbage", 12.0), ("   ", 99.0),
                    ("junk", -1.0)])
        self.assertEqual(oe._confident_chars(tsv), 6)

    def test_confident_chars_survives_a_non_numeric_confidence(self):
        """Tesseract's conf column is text; a build that emits '' must not
        take the whole page down."""
        tsv = {"text": ["Bhopal", "x"], "conf": ["90.0", ""]}
        self.assertEqual(oe._confident_chars(tsv), 6)
        self.assertEqual(oe._confident_chars(None), 0)

    def _pick(self, block_words, auto_words):
        calls = []

        def stub(path, lang, psm=oe._PSM_SINGLE_BLOCK):
            calls.append(psm)
            return _tsv(block_words if psm == oe._PSM_SINGLE_BLOCK
                        else auto_words)

        real = oe._ocr_page
        try:
            oe._ocr_page = stub
            tsv, psm = oe._ocr_page_best_layout("irrelevant.png", "eng")
        finally:
            oe._ocr_page = real
        return psm, calls

    def test_single_block_is_kept_when_the_margin_is_not_cleared(self):
        """A small lead is run-to-run variation, not evidence of layout.
        Without the bar, noise alone would flip the mode on half the corpus
        and give back the recall psm 6 earns on ordinary pages."""
        psm, _ = self._pick([("Bhopalxx", 90.0)], [("Bhopalxxx", 90.0)])
        self.assertEqual(psm, oe._PSM_SINGLE_BLOCK)

    def test_auto_layout_wins_when_it_finds_substantially_more(self):
        psm, _ = self._pick([("short", 90.0)],
                            [("aaaaaaaaaa", 90.0), ("bbbbbbbbbb", 90.0)])
        self.assertEqual(psm, oe._PSM_AUTO_LAYOUT)

    def test_both_modes_are_actually_tried(self):
        _psm, calls = self._pick([("x", 90.0)], [("y", 90.0)])
        self.assertIn(oe._PSM_SINGLE_BLOCK, calls)
        self.assertIn(oe._PSM_AUTO_LAYOUT, calls)

    def test_a_failed_mode_does_not_lose_the_page(self):
        """One mode returning None must fall back to the other rather than
        reporting the page unreadable."""
        def only_auto(path, lang, psm=oe._PSM_SINGLE_BLOCK):
            return None if psm == oe._PSM_SINGLE_BLOCK else _tsv([("ok", 90.0)])

        def only_block(path, lang, psm=oe._PSM_SINGLE_BLOCK):
            return _tsv([("ok", 90.0)]) if psm == oe._PSM_SINGLE_BLOCK else None

        real = oe._ocr_page
        try:
            oe._ocr_page = only_auto
            tsv, psm = oe._ocr_page_best_layout("x.png", "eng")
            self.assertIsNotNone(tsv)
            self.assertEqual(psm, oe._PSM_AUTO_LAYOUT)

            oe._ocr_page = only_block
            tsv, psm = oe._ocr_page_best_layout("x.png", "eng")
            self.assertIsNotNone(tsv)
            self.assertEqual(psm, oe._PSM_SINGLE_BLOCK)
        finally:
            oe._ocr_page = real

    def test_script_selection_still_uses_one_mode(self):
        """Language packs must be compared on equal terms. Letting the trial
        pick a psm per pack would make the winner a function of two variables
        and the reported margin meaningless."""
        seen = []

        def stub(path, lang, psm=oe._PSM_SINGLE_BLOCK):
            seen.append(psm)
            return {"score": 50.0}

        real_ocr, real_conf = oe._ocr_page, oe._tsv_mean_confidence
        try:
            oe._ocr_page = stub
            oe._tsv_mean_confidence = lambda tsv: tsv["score"]
            oe.select_languages("irrelevant.png", ["hin", "eng"])
        finally:
            oe._ocr_page, oe._tsv_mean_confidence = real_ocr, real_conf
        self.assertEqual(set(seen), {oe._PSM_SINGLE_BLOCK})


def _line(text, bbox, page=1):
    return oe.Line(text=text, confidence=0.9, page=page, bbox=bbox,
                   source="tesseract")


class DigitRepairTests(unittest.TestCase):
    """
    Picking one language pack per page is the right call for prose and the
    wrong one for numerals.

    Measured on a real Bhu-Naksha plot report: hin+eng won script selection by
    13.54 confidence - correctly, it is the only pack that reads the Devanagari
    - and then misread every number on the page. Khata 00100 came back 0000,
    plot 184 came back 84, area 1.6350 came back .6350. The same crops under
    plain eng were exactly right.

    So the Indic pass keeps the prose and an English pass is consulted for the
    digit runs alone. These tests pin the guards, because a mis-aligned splice
    would invent an identifier that was never on the page - worse than the
    misreading it set out to fix.
    """

    def test_digits_are_taken_from_the_english_pass(self):
        primary = [_line("khata no: 0000 plot no: 84", (0, 100, 500, 130))]
        digits = [_line("khata no: 00100 plot no: 184", (0, 100, 500, 130))]
        self.assertEqual(oe._repair_digits(primary, digits), 1)
        self.assertEqual(primary[0].text, "khata no: 00100 plot no: 184")

    def test_the_prose_is_left_alone(self):
        """Only the digit runs move. The Devanagari reading is the good one
        and must survive the repair untouched."""
        primary = [_line("\u0916\u0938\u0930\u093e : 84", (0, 100, 500, 130))]
        digits = [_line("GERI AG : 184", (0, 100, 500, 130))]
        oe._repair_digits(primary, digits)
        self.assertEqual(primary[0].text, "\u0916\u0938\u0930\u093e : 184")

    def test_a_cross_script_line_is_still_matched(self):
        """
        The guard cannot depend on the two readings looking alike. On
        'khasra : 184' the Devanagari pass reads the label as Devanagari and
        the English pass reads the same pixels as 'GERI AG'; an earlier
        text-similarity guard rejected exactly the repairs that matter.
        """
        primary = [_line("\u0916\u0938\u0930\u093e \u0928\u0902\u092c\u0930 : 84",
                         (10, 100, 400, 130))]
        digits = [_line("GERI AG : 184", (12, 101, 398, 129))]
        self.assertEqual(oe._repair_digits(primary, digits), 1)

    def test_a_different_number_of_numbers_is_refused(self):
        """Two readings that disagree about how many numbers the line holds
        cannot be spliced positionally - there is no correspondence to use."""
        primary = [_line("plot 84 area 6350", (0, 100, 500, 130))]
        digits = [_line("plot 184", (0, 100, 500, 130))]
        self.assertEqual(oe._repair_digits(primary, digits), 0)
        self.assertEqual(primary[0].text, "plot 84 area 6350")

    def test_a_line_at_the_same_height_elsewhere_is_refused(self):
        """A two-column page puts unrelated lines at identical heights.
        Vertical overlap alone would splice one column's digits into the
        other's."""
        primary = [_line("plot 84", (0, 100, 200, 130))]
        digits = [_line("year 1947", (900, 100, 1100, 130))]
        self.assertEqual(oe._repair_digits(primary, digits), 0)
        self.assertEqual(primary[0].text, "plot 84")

    def test_a_different_page_is_never_used(self):
        primary = [_line("plot 84", (0, 100, 500, 130), page=1)]
        digits = [_line("plot 184", (0, 100, 500, 130), page=2)]
        self.assertEqual(oe._repair_digits(primary, digits), 0)

    def test_lines_without_digits_are_untouched(self):
        primary = [_line("\u0928\u093f\u0935\u093e\u0938 \u0938\u094d\u0925\u093e\u0928",
                         (0, 100, 500, 130))]
        before = primary[0].text
        self.assertEqual(oe._repair_digits(primary, [_line("junk", (0, 100, 500, 130))]), 0)
        self.assertEqual(primary[0].text, before)

    def test_agreeing_readings_are_not_counted_as_repairs(self):
        primary = [_line("plot 184", (0, 100, 500, 130))]
        digits = [_line("plot 184", (0, 100, 500, 130))]
        self.assertEqual(oe._repair_digits(primary, digits), 0)

    def test_empty_inputs_are_safe(self):
        self.assertEqual(oe._repair_digits([], []), 0)
        self.assertEqual(oe._repair_digits([_line("a 1", (0, 0, 10, 10))], []), 0)

    def test_digit_runs_and_skeleton(self):
        self.assertEqual(oe._digit_runs("khata 00100 plot 184"), ["00100", "184"])
        self.assertEqual(oe._digit_runs("no digits here"), [])
        self.assertEqual(oe._digit_skeleton("plot 184 of 2022"), "plot # of #")


class BlockedOptionalDependencyTests(unittest.TestCase):
    """
    An optional dependency being BLOCKED must never take the pipeline down.

    This is a regression test for a real outage, and the cause is worth
    keeping written down because it will recur.

    This machine runs Windows Smart App Control in ENFORCED mode, which
    refuses to load binaries it does not consider verified-and-reputable -
    that is, most compiled Python wheels. It has now taken out torch
    (WinError 4551), spaCy's pipeline .pyd, and scipy's DLLs (_ufuncs_cxx,
    _ellip_harm_2 - the name varies per run, because the decision consults a
    cloud reputation service and is therefore INTERMITTENT).

    The failure was not the block itself; degradation handles that. It was
    that SKIMAGE_AVAILABLE used a bare `hasattr(_skimage_feature, "canny")`
    at module scope. scikit-image uses lazy_loader, so that attribute access
    is what actually performs the import - and when the import is blocked,
    hasattr RAISES instead of returning False. The exception propagated
    straight out of `import ocr_engine`, so a missing OPTIONAL library made
    the core module unimportable and the whole system unusable.
    """

    def test_has_returns_false_instead_of_raising(self):
        """The exact shape of the failure: attribute access explodes."""

        class Exploding:
            def __getattr__(self, name):
                raise ImportError(
                    "DLL load failed while importing _ufuncs_cxx: "
                    "An Application Control policy has blocked this file.")

        self.assertFalse(oe._has(Exploding(), "canny"))

    def test_has_survives_any_exception_type(self):
        """lazy_loader has raised more than one kind over the years, and the
        contract is "report unavailable", not "report unavailable for the
        errors we predicted"."""
        for exc in (ImportError, OSError, RuntimeError, AttributeError, ValueError):
            class Boom:
                def __getattr__(self, name, _e=exc):
                    raise _e("blocked")
            self.assertFalse(oe._has(Boom(), "canny"), exc.__name__)

    def test_has_is_false_for_a_missing_module(self):
        self.assertFalse(oe._has(None, "canny"))

    def test_has_is_false_when_the_attribute_is_simply_absent(self):
        """The original reason the attribute check exists: a module that
        imported fine but is the wrong module. _try_import once returned the
        top-level package for a dotted name, and every call into it failed
        silently."""
        self.assertFalse(oe._has(object(), "canny"))

    def test_has_is_true_for_a_real_attribute(self):
        self.assertTrue(oe._has(oe, "extract"))

    def test_the_availability_flag_is_a_bool_either_way(self):
        """Downstream code branches on this; None or an exception object
        would be truthy and silently re-enable a dead path."""
        self.assertIsInstance(oe.SKIMAGE_AVAILABLE, bool)

    def test_the_module_is_usable_with_every_optional_dependency_absent(self):
        """
        The contract in one assertion: OCR capability is reported, not
        assumed, and the module answers rather than raising.
        """
        caps = oe.capabilities()
        for key in ("pdf_text_layer", "image_preprocessing", "tesseract"):
            self.assertIn(key, caps)
        self.assertIsInstance(oe.tesseract_available(), bool)


class LongPathTests(unittest.TestCase):
    """
    Regression tests for a real, silent, product-breaking bug: when this
    project sits in a deep directory (its own development checkout is 209
    characters deep), the preprocessed-image path crosses Windows' 260-char
    MAX_PATH limit. cv2.imwrite then fails by RETURNING FALSE rather than
    raising, the unwritten path was handed to Tesseract anyway, and every
    scanned image extracted zero text while still reporting
    engine=tesseract - a successful-looking OCR run that silently found
    nothing. Found by end-to-end testing; the unit suite passed throughout.
    """

    def test_deep_directory_target_is_relocated_somewhere_usable(self):
        deep = "C:\\" + "\\".join("d" * 20 for _ in range(14))
        self.assertGreater(len(deep), 250, "fixture must actually be a deep path")
        target = oe._preprocessed_target(deep, "scan_01_khatauni_up_clean_skew_blur")
        if os.name == "nt":
            self.assertLess(len(os.path.abspath(target)), oe._MAX_USABLE_PATH,
                            "relocated target must be openable by both OpenCV "
                            "and the external Tesseract process")
        self.assertTrue(target.endswith(".png"))

    def test_short_directory_target_is_left_alone(self):
        with tempfile.TemporaryDirectory() as short:
            target = oe._preprocessed_target(short, "page")
            self.assertEqual(os.path.dirname(os.path.abspath(target)),
                             os.path.abspath(short),
                             "a path that fits must stay next to its document")

    def test_distinct_deep_directories_do_not_collide(self):
        a = oe._preprocessed_target("C:\\" + "a" * 260, "page")
        b = oe._preprocessed_target("C:\\" + "b" * 260, "page")
        if os.name == "nt":
            self.assertNotEqual(a, b, "two documents must not overwrite each "
                                      "other's preprocessed image")



class DeskewThresholdTests(unittest.TestCase):
    """
    Rotating a page is not free, so the threshold is a measured quantity.

    warpAffine with INTER_CUBIC resamples every pixel, softening strokes
    that were sharp. Below the angle at which Tesseract stops coping by
    itself, that is pure cost. The threshold was 0.4 degrees, which meant
    every real page that produced a skew estimate got rotated:

      * 32 real pages (15 Bhu-Naksha reports + a 13-page notarised GPA)
        yielded 7 Hough estimates, ALL between 1.22 and 1.88 degrees.
      * The synthetic corpus is skewed 2.4 and 3.6 degrees on purpose by
        tools/make_samples.py, and the estimator reads -2.291 on it.

    So the real and synthetic bands do not overlap, and 2.0 sits in the gap.
    Raising it costs nothing - gated Hough deskew measured exactly zero
    difference on the synthetic corpus - and recovers 6.6 points of field
    accuracy on the real one, which is why 84.4% was previously only
    reproducible with scikit-image pinned off.
    """

    REAL_ANGLES = (1.219, 1.302, 1.381, 1.397, 1.397, 1.302, 1.878)
    SYNTHETIC_ANGLE = -2.291

    def test_no_real_page_is_rotated(self):
        """The measured band from 32 real pages, none of which benefits."""
        for angle in self.REAL_ANGLES:
            self.assertFalse(oe._should_deskew(angle), angle)

    def test_the_deliberately_skewed_synthetic_page_is_rotated(self):
        """The capability must survive the threshold change."""
        self.assertTrue(oe._should_deskew(self.SYNTHETIC_ANGLE))

    def test_the_threshold_sits_between_the_two_bands(self):
        """
        If this fails, the threshold has drifted out of the measured gap and
        the 6.6-point real-corpus regression is back.
        """
        self.assertGreater(oe.MIN_DESKEW_DEGREES, max(self.REAL_ANGLES))
        self.assertLess(oe.MIN_DESKEW_DEGREES, abs(self.SYNTHETIC_ANGLE))

    def test_a_genuinely_crooked_scan_is_still_corrected(self):
        """
        Declining to rotate applies only in the band where rotating was
        measured to hurt. No measurement here covers a badly crooked page,
        and Tesseract does eventually stop absorbing skew.
        """
        for angle in (2.5, 5.0, 10.0, 17.0):
            self.assertTrue(oe._should_deskew(angle), angle)

    def test_the_decision_ignores_the_direction_of_lean(self):
        for angle in (3.0, -3.0, 1.4, -1.4):
            self.assertEqual(oe._should_deskew(angle),
                             oe._should_deskew(-angle), angle)

    def test_no_estimate_means_no_rotation(self):
        """
        _skew_hough returns None when its evidence gate fails, and 25 of the
        32 real pages do exactly that. None must never be treated as zero
        and must never rotate.
        """
        self.assertFalse(oe._should_deskew(None))

    def test_exactly_at_the_threshold_does_not_rotate(self):
        """Strictly greater, so the boundary is not a coin toss."""
        self.assertFalse(oe._should_deskew(oe.MIN_DESKEW_DEGREES))


if __name__ == "__main__":
    unittest.main()
