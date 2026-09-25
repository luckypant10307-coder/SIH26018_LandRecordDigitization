#!/usr/bin/env python3
"""
Unit tests for backend/document_authenticity.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The detection thresholds and approach in document_authenticity.py were
tuned and verified against real reference material (real India Non-Judicial
stamp paper and a real notarised affidavit with a genuine notary seal,
supplied for this purpose) - see that module's docstring for what was tried
and measurably failed (a naive top-of-page saturation average, a naive
darkness threshold for seal ink, cv2.HoughCircles for seal shape) before
arriving at the current approach.

That real material contains a named individual's personal and financial
details and is deliberately NOT bundled into this repository. These tests
instead use small synthetic fixtures built to share the exact structural
properties that were verified against the real pages: a saturated colour
band away from the very top edge (real stamp paper's banner did not start
until ~15% down the page), a ring-shaped coloured seal mark (not a solid
disc - a real seal is made of small text characters, not solid ink), and a
sparse, non-printed-text ink pattern for a signature.

Run from anywhere with:
    python3 tests/test_document_authenticity.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import document_authenticity as da  # noqa: E402


def _blank_page(cv2, np, w=900, h=1200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


@unittest.skipUnless(da.CV_AVAILABLE, "OpenCV/numpy not installed")
class StampPaperDetectionTests(unittest.TestCase):
    def test_banner_offset_from_top_edge_is_still_found(self):
        # Regression: a naive average over the top region diluted the real
        # banner's saturation below threshold because the banner did not
        # start at row 0 on real pages - reproduce that offset here.
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        h, w = img.shape[:2]
        band_y0, band_y1 = int(h * 0.15), int(h * 0.30)
        img[band_y0:band_y1, :] = (200, 180, 60)  # a saturated BGR colour
        path = "test_stamp_banner.png"
        cv2.imwrite(path, img)
        try:
            detected, _ = da.detect_stamp_paper(path)
            self.assertTrue(detected)
        finally:
            os.remove(path)

    def test_blank_page_is_not_stamp_paper(self):
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        path = "test_blank.png"
        cv2.imwrite(path, img)
        try:
            detected, text = da.detect_stamp_paper(path)
            self.assertFalse(detected)
            self.assertIsNone(text)
        finally:
            os.remove(path)


@unittest.skipUnless(da.CV_AVAILABLE, "OpenCV/numpy not installed")
class SealDetectionTests(unittest.TestCase):
    def _ring_seal(self, cv2, np, img, center, radius, color):
        """A ring outline, not a solid filled disc - a real seal's ink is
        its circular border plus a ring of text, never a solid blob."""
        cv2.circle(img, center, radius, color, thickness=8)

    def test_ring_shaped_seal_is_detected(self):
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        self._ring_seal(cv2, np, img, center=(300, 900), radius=60, color=(200, 60, 180))
        path = "test_seal.png"
        cv2.imwrite(path, img)
        try:
            seals = da.detect_seals(path)
            self.assertEqual(len(seals), 1)
            x, y, bw, bh = seals[0].bbox
            aspect = bw / float(bh)
            self.assertAlmostEqual(aspect, 1.0, delta=0.3)
        finally:
            os.remove(path)

    def test_elongated_coloured_text_row_is_not_a_seal(self):
        # Regression: cv2.HoughCircles hallucinated circles from elongated
        # coloured text rows on a real page; the aspect-ratio filter must
        # reject a wide, short coloured band outright.
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        cv2.rectangle(img, (100, 900), (700, 940), (60, 60, 200), -1)
        path = "test_text_row.png"
        cv2.imwrite(path, img)
        try:
            seals = da.detect_seals(path)
            self.assertEqual(seals, [])
        finally:
            os.remove(path)

    def test_no_seals_on_blank_page(self):
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        path = "test_blank2.png"
        cv2.imwrite(path, img)
        try:
            self.assertEqual(da.detect_seals(path), [])
        finally:
            os.remove(path)


@unittest.skipUnless(da.CV_AVAILABLE, "OpenCV/numpy not installed")
class SignatureDetectionTests(unittest.TestCase):
    def test_sparse_ink_in_bottom_region_is_a_signature(self):
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        h, w = img.shape[:2]
        cv2.ellipse(img, (int(w * 0.5), int(h * 0.85)), (120, 30), 0, 0, 300, (30, 30, 30), 2)
        path = "test_signature.png"
        cv2.imwrite(path, img)
        try:
            detected, region = da.detect_signature(path)
            self.assertTrue(detected)
            self.assertIsNotNone(region)
        finally:
            os.remove(path)

    def test_blank_bottom_region_has_no_signature(self):
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        path = "test_blank3.png"
        cv2.imwrite(path, img)
        try:
            detected, _ = da.detect_signature(path)
            self.assertFalse(detected)
        finally:
            os.remove(path)

    def test_dense_printed_text_block_is_not_a_signature(self):
        # A full block of printed text is far denser than a signature -
        # must fall above max_ink_ratio, not be mistaken for one.
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        h, w = img.shape[:2]
        region_y0 = int(h * 0.65)
        for y in range(region_y0 + 10, h - 10, 8):
            cv2.line(img, (40, y), (w - 40, y), (20, 20, 20), 4)
        path = "test_dense_text.png"
        cv2.imwrite(path, img)
        try:
            detected, _ = da.detect_signature(path)
            self.assertFalse(detected)
        finally:
            os.remove(path)


@unittest.skipUnless(da.CV_AVAILABLE, "OpenCV/numpy not installed")
class AnalyzeDocumentTests(unittest.TestCase):
    def test_combined_result_shape(self):
        cv2, np = da._cv2, da._numpy
        img = _blank_page(cv2, np)
        path = "test_combined.png"
        cv2.imwrite(path, img)
        try:
            result = da.analyze_document(path)
            d = result.to_dict()
            self.assertIn("stamp_paper_detected", d)
            self.assertIn("seals", d)
            self.assertIn("signature_detected", d)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
