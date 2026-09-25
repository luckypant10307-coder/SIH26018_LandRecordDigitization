#!/usr/bin/env python3
"""
Unit tests for backend/trocr_htr.py (TrOCR handwriting recognition via ONNX).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Most of these run WITHOUT the 600 MB weights, because the parts most likely
to be silently wrong are not the neural network - they are the byte-level BPE
detokenizer and the script guard, both of which are hand-written here and
both of which fail quietly rather than loudly when wrong.

The script guard is the one that matters. TrOCR's handwritten checkpoint is
IAM-trained English; shown Devanagari it does not refuse, it invents. Measured
on a real Rajasthan certificate it returned "MRB Protected Authority
Commission Decem" for a Hindi society name, at a confidence that would pass
most thresholds. The guard has to stop that before the model runs.

Run from anywhere with:
    python3 tests/test_trocr_htr.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import trocr_htr as tr  # noqa: E402

try:
    import cv2
    import numpy as np
    CV_OK = True
except Exception:                                    # pragma: no cover
    CV_OK = False

WEIGHTS = tr.available()
ABSENT = os.path.join(tempfile.gettempdir(), "trocr_definitely_absent")


class ByteDecoderTests(unittest.TestCase):
    """
    Byte-level BPE stores BYTES re-encoded as printable codepoints. Getting
    the inverse mapping wrong turns every non-ASCII character into mojibake,
    and - worse - still returns plausible-looking ASCII for English, so it
    would pass a casual glance.
    """

    def test_the_table_covers_every_byte_exactly_once(self):
        table = tr._bytes_to_unicode()
        self.assertEqual(len(table), 256)
        self.assertEqual(len(set(table.values())), 256)
        self.assertEqual(sorted(table.keys()), list(range(256)))

    def test_printable_ascii_maps_to_itself(self):
        table = tr._bytes_to_unicode()
        for ch in "Aa0/-.":
            self.assertEqual(table[ord(ch)], ch)

    def test_space_uses_the_G_with_stroke_convention(self):
        """A leading space is stored as U+0120, which is why tokens look like
        'Ġthe'. If this is not reversed, every word runs together."""
        self.assertEqual(tr._bytes_to_unicode()[ord(" ")], "\u0120")


@unittest.skipUnless(WEIGHTS, "TrOCR weights not downloaded")
class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = tr.Tokenizer(os.path.join(tr.MODEL_DIR, tr.TOKENIZER_FILE))

    def test_special_tokens_are_dropped_from_output(self):
        self.assertEqual(self.tok.decode([tr.DECODER_START_TOKEN]), "")
        self.assertEqual(self.tok.decode([tr.PAD_TOKEN]), "")

    def test_spaces_are_restored_between_words(self):
        """'Ġ' prefixes must become real spaces, or words concatenate."""
        ids = [i for i in (1344, 337, 8014, 1528) ]      # ins|al|ums|Ġtrue
        text = self.tok.decode(ids)
        self.assertIn(" ", text)
        self.assertFalse(text.startswith(" "))

    def test_unknown_ids_are_skipped_not_crashed_on(self):
        self.assertEqual(self.tok.decode([999999999]), "")


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class PreprocessTests(unittest.TestCase):
    def test_output_shape_matches_the_vit(self):
        crop = np.full((40, 300), 200, dtype=np.uint8)
        tensor = tr.TrOCR.preprocess(crop)
        self.assertEqual(tensor.shape, (1, 3, tr.IMAGE_SIZE, tr.IMAGE_SIZE))

    def test_values_are_normalised_to_the_trained_range(self):
        """(x/255 - 0.5)/0.5, so pure black is -1 and pure white is +1."""
        black = tr.TrOCR.preprocess(np.zeros((40, 300), np.uint8))
        white = tr.TrOCR.preprocess(np.full((40, 300), 255, np.uint8))
        self.assertAlmostEqual(float(black.min()), -1.0, places=5)
        self.assertAlmostEqual(float(white.max()), 1.0, places=5)

    def test_aspect_ratio_is_deliberately_not_preserved(self):
        """
        TrOCR was trained on line crops squashed to a square. Letterboxing
        looks more respectful of the image and gives the model geometry it
        never saw, so a wide crop and a tall one must both come out square.
        """
        wide = tr.TrOCR.preprocess(np.full((20, 600), 128, np.uint8))
        tall = tr.TrOCR.preprocess(np.full((600, 20), 128, np.uint8))
        self.assertEqual(wide.shape, tall.shape)

    def test_an_empty_crop_is_refused(self):
        with self.assertRaises(Exception):
            tr.TrOCR.preprocess(np.zeros((0, 0), np.uint8))


@unittest.skipUnless(CV_OK, "OpenCV/numpy not installed")
class ScriptGuardTests(unittest.TestCase):
    """
    The guard runs BEFORE the model, so these need no weights - which is the
    point: refusing an unsupported script must not depend on a 600 MB
    download being present.
    """

    CROP = None

    def setUp(self):
        self.CROP = np.full((40, 300), 200, dtype=np.uint8)

    def test_devanagari_is_refused_without_running_the_model(self):
        result = tr.transcribe_line(self.CROP, "नरहरपुर सुनीता देवी",
                                    model_dir=ABSENT)
        self.assertIsNotNone(result)
        self.assertFalse(result["script_supported"])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)

    def test_the_refusal_explains_itself(self):
        result = tr.transcribe_line(self.CROP, "सुनीता देवी", model_dir=ABSENT)
        self.assertIn("Latin", result["reason"])
        self.assertIn("devanagari", result["reason"])

    def test_a_refusal_never_returns_invented_text(self):
        """
        The failure this guard exists for: on real Devanagari the model
        returned fluent English ("MRB Protected Authority Commission Decem").
        An empty string is the only safe answer.
        """
        for text in ("नरहरपुर", "সুনীতা", "சுனிதா", "ಸುನೀತಾ"):
            result = tr.transcribe_line(self.CROP, text, model_dir=ABSENT)
            self.assertEqual(result["text"], "", text)
            self.assertFalse(result["script_supported"], text)

    def test_missing_weights_return_none_for_a_supported_script(self):
        """None means 'not attempted' and must never read as an empty
        transcription of a line that was never looked at."""
        self.assertIsNone(tr.transcribe_line(self.CROP, "industrie",
                                             model_dir=ABSENT))


class AvailabilityTests(unittest.TestCase):
    def test_describe_reports_missing_weights_with_a_hint(self):
        report = tr.describe(ABSENT)
        self.assertFalse(report["available"])
        self.assertIn("fetch_trocr", report.get("hint", ""))

    def test_transcribe_without_weights_returns_none(self):
        self.assertIsNone(tr.transcribe(None, ABSENT))

    @unittest.skipUnless(WEIGHTS, "TrOCR weights not downloaded")
    def test_describe_states_the_script_limitation(self):
        report = tr.describe()
        self.assertTrue(report["available"])
        self.assertIn("Latin", report["script"])
        self.assertFalse(report["auto_accept"])

    def test_the_encoder_default_is_full_precision(self):
        """
        Measured, not assumed: int8 quantisation of the vision tower turned
        a cursive "industrie" into "insalums true", while int8 on the decoder
        cost nothing. A future tidy-up that makes both consistent would
        silently break the recogniser.
        """
        self.assertNotIn("quantized", tr.ENCODER_FILE)
        self.assertIn("quantized", tr.DECODER_FILE)


@unittest.skipUnless(WEIGHTS and CV_OK, "TrOCR weights not downloaded")
class EndToEndTests(unittest.TestCase):
    def test_a_latin_line_is_transcribed_and_marked_for_review(self):
        sample = os.path.join("C:/lrdv/iam", "iam_picture.jpeg")
        if not os.path.exists(sample):
            self.skipTest("IAM fixture not fetched")
        crop = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
        result = tr.transcribe_line(crop, "industrie")
        self.assertTrue(result["script_supported"])
        self.assertTrue(result["needs_review"],
                        "a transcription must never be auto-accepted")
        self.assertGreater(len(result["text"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
