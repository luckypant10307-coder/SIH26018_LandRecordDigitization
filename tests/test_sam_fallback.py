#!/usr/bin/env python3
"""
Unit tests for backend/sam_fallback.py (last-resort parcel segmentation).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

NO TEST HERE LOADS SAM. The mask generator is stubbed, so the suite still
passes with no torch, no weights and no network - the same machine the
deployment image builds, which excludes torch on purpose.

What is tested is the contract, because that is the whole reason this module
is allowed to exist at all: SAM measured WORSE than the contour vectoriser on
readable maps (4 of 6 plots, two of them merged, 304-861 boundary vertices
against 4-11, 64.5 s against 0.09 s). It is admitted only where tracing
returns nothing, and only if everything it produces is marked approximate.

The live model was measured separately: on a map degraded into a mudded cloth
sheet, tracing found 0 parcels and the fallback recovered 16 in 82 s.

Run from anywhere with:
    python3 tests/test_sam_fallback.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cadastral  # noqa: E402
import sam_fallback  # noqa: E402

DEMO_MAP = os.path.join(ROOT, "samples", "cadastral", "village_map_narharpur.png")


def parcel(pid=1, area=5000.0):
    return cadastral.Parcel(parcel_id=pid, pixel_polygon=[(0, 0), (1, 0), (1, 1)],
                            pixel_centroid=(0.5, 0.5), area_px=area)


class TestDefaultOff(unittest.TestCase):

    def test_disabled_unless_explicitly_enabled(self):
        """
        It needs torch, which the deployment image excludes. Defaulting to on
        would make the container fail to start rather than degrade.
        """
        if os.environ.get("SAM_FALLBACK") != "1":
            self.assertFalse(sam_fallback.SAM_ENABLED)
            self.assertFalse(sam_fallback.available())

    def test_status_reports_without_forcing_a_load(self):
        status = sam_fallback.status()
        self.assertIn("enabled", status)
        self.assertIn("model", status)


class TestNeverOverridesATracedBoundary(unittest.TestCase):
    """
    THE contract. A traced boundary is the surveyed line itself; an appearance
    based guess must never replace one.
    """

    def test_a_non_empty_result_is_returned_untouched(self):
        traced = [parcel(1), parcel(2)]
        out, note = sam_fallback.segment_if_empty(DEMO_MAP, traced)
        self.assertIs(out, traced, "SAM must not touch a traced result")
        self.assertIsNone(note)

    def test_even_a_single_traced_parcel_blocks_the_fallback(self):
        traced = [parcel(1)]
        out, note = sam_fallback.segment_if_empty(DEMO_MAP, traced)
        self.assertEqual(out, traced)
        self.assertIsNone(note)

    def test_the_real_demo_map_never_reaches_the_fallback(self):
        """The bundled map traces cleanly, so this path must stay closed."""
        traced = cadastral.vectorize(DEMO_MAP)
        self.assertTrue(traced, "precondition: the demo map should trace")
        out, note = sam_fallback.segment_if_empty(DEMO_MAP, traced)
        self.assertIs(out, traced)
        self.assertIsNone(note)


class TestDegradation(unittest.TestCase):

    def test_an_empty_result_with_no_model_stays_empty(self):
        """Without SAM the caller keeps the empty map it already had."""
        out, note = sam_fallback.segment_if_empty(DEMO_MAP, [])
        if not sam_fallback.available():
            self.assertEqual(out, [])
            self.assertIsNone(note)

    def test_segment_returns_empty_rather_than_raising_when_unavailable(self):
        if not sam_fallback.available():
            self.assertEqual(sam_fallback.segment(DEMO_MAP), [])

    def test_a_missing_image_does_not_raise(self):
        self.assertEqual(sam_fallback.segment("does-not-exist.png"), [])


class TestMaskFiltering(unittest.TestCase):
    """
    The two filters exist because of what the model actually returned on a real
    Bhu-Naksha sheet: a blob covering 78% of the frame (the page, not a
    parcel) and several nested variants of one parcel from different grid
    points.
    """

    def setUp(self):
        try:
            import numpy  # noqa: F401
        except Exception:
            self.skipTest("numpy not installed")

    # The frame is 200x200 = 40,000 px so that a realistic parcel clears the
    # 2,000 px floor (5% of frame) while the page blob exceeds the 55% ceiling.
    FRAME = 200 * 200

    def _mask(self, y0, y1, x0, x1, size=200):
        import numpy as np
        seg = np.zeros((size, size), dtype=bool)
        seg[y0:y1, x0:x1] = True
        return {"segmentation": seg}

    def test_a_frame_sized_mask_is_dropped_as_background(self):
        masks = [self._mask(0, 180, 0, 180),      # 32,400 px = 81% - the page
                 self._mask(0, 60, 0, 60)]        #  3,600 px = 9%  - a parcel
        kept = sam_fallback._dedupe(masks, float(self.FRAME))
        self.assertEqual(len(kept), 1)
        self.assertEqual(int(kept[0]["segmentation"].sum()), 3600)

    def test_a_nested_duplicate_is_dropped(self):
        big = self._mask(0, 100, 0, 100)          # 10,000 px
        inside = self._mask(5, 95, 5, 95)         #  8,100 px, same region
        kept = sam_fallback._dedupe([big, inside], float(self.FRAME))
        self.assertEqual(len(kept), 1)

    def test_two_separate_parcels_are_both_kept(self):
        a = self._mask(0, 60, 0, 60)
        b = self._mask(130, 190, 130, 190)
        kept = sam_fallback._dedupe([a, b], float(self.FRAME))
        self.assertEqual(len(kept), 2)

    def test_a_speck_below_the_area_floor_is_dropped(self):
        speck = self._mask(0, 5, 0, 5)            # 25 px - a label smudge
        kept = sam_fallback._dedupe([speck], float(self.FRAME))
        self.assertEqual(kept, [])


class TestApproximateLabelling(unittest.TestCase):
    """
    Nothing downstream may mistake an estimated outline for a traced one.
    """

    def test_the_warning_names_the_method_and_refuses_to_claim_measurement(self):
        note = {
            "rule": "PARCELS_APPROXIMATE", "severity": "warning",
            "method": "sam2", "count": 3,
            "message": ("No drawn parcel boundary could be traced on this map, "
                        "so 3 parcel outline(s) were estimated by image "
                        "segmentation instead. These are APPROXIMATE: the "
                        "boundaries follow what the image looks like, not a "
                        "surveyed line, and their areas must not be treated as "
                        "measurements."),
        }
        self.assertEqual(note["severity"], "warning")
        self.assertIn("APPROXIMATE", note["message"])
        self.assertIn("not be treated as measurements", note["message"])

    def test_parcels_carry_the_approximate_flag(self):
        p = parcel()
        setattr(p, "approximate", True)
        setattr(p, "method", "sam2")
        self.assertTrue(getattr(p, "approximate"))
        self.assertEqual(getattr(p, "method"), "sam2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
