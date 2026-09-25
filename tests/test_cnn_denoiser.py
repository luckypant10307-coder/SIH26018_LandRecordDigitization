#!/usr/bin/env python3
"""
Unit tests for backend/cnn_denoiser.py (the custom CNN scan denoiser).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

A hand-written network with a hand-written backward pass has no framework to
catch a transposed index for it, so the tests here are the safety net:

  * AdjointTests    - im2col and col2im really are transposes of each other,
                      which is the assumption the whole backward pass rests on.
  * GradientTests   - the analytic gradient matches a numerical one in float64.
  * IdentityTests   - an untrained network is exactly the identity, so the
                      denoiser can never damage a page it has learned nothing
                      about.
  * TilingTests     - denoising a page tile-by-tile gives the same answer as
                      denoising it whole, i.e. no seams.
  * DegradationTests- absent or corrupt weights disable the denoiser instead
                      of returning a blank page.

Run from anywhere with:
    python3 tests/test_cnn_denoiser.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cnn_denoiser as cd  # noqa: E402

try:
    import numpy as np
    NUMPY = True
except Exception:                                    # pragma: no cover
    NUMPY = False


@unittest.skipUnless(NUMPY, "numpy is not installed")
class AdjointTests(unittest.TestCase):
    """
    col2im must be the exact adjoint of im2col: <im2col(x), c> == <x, col2im(c)>
    for every x and c. The backward pass uses col2im to push gradients back
    through a convolution, so if this identity fails the network trains
    towards the wrong thing while still reporting a falling loss.
    """

    def test_col2im_is_the_transpose_of_im2col(self):
        rng = np.random.default_rng(0)
        for cin, h, w in ((1, 5, 7), (3, 8, 8), (16, 4, 6)):
            x = rng.standard_normal((cin, h, w))
            c = rng.standard_normal((h * w, cin * 9))
            lhs = float((cd.im2col(x) * c).sum())
            rhs = float((x * cd.col2im(c, (cin, h, w))).sum())
            self.assertAlmostEqual(lhs, rhs, places=8,
                                   msg=f"adjoint broken at {(cin, h, w)}")

    def test_columns_follow_the_input_dtype(self):
        """float64 has to survive, or the gradient check is meaningless."""
        x64 = np.zeros((2, 4, 4), dtype=np.float64)
        self.assertEqual(cd.im2col(x64).dtype, np.float64)
        x32 = np.zeros((2, 4, 4), dtype=np.float32)
        self.assertEqual(cd.im2col(x32).dtype, np.float32)


@unittest.skipUnless(NUMPY, "numpy is not installed")
class GradientTests(unittest.TestCase):
    def test_backward_matches_numerical_gradient(self):
        import train_denoiser as td
        self.assertTrue(td.gradcheck(), "analytic gradient disagrees with numerical")


@unittest.skipUnless(NUMPY, "numpy is not installed")
class IdentityTests(unittest.TestCase):
    """
    The final layer is initialised to zero, so an untrained network predicts a
    zero residual and returns the page untouched. This is a safety property,
    not a curiosity: most pages this system sees are not degraded, and a
    denoiser that begins by altering them would corrupt clean records while
    "restoring" them.
    """

    def test_untrained_network_predicts_no_residual(self):
        params = cd.init_params(0)
        x = np.random.default_rng(1).random((1, 16, 16)).astype(np.float32)
        residual = cd.forward(x, params)
        self.assertEqual(float(np.abs(residual).max()), 0.0)

    def test_untrained_denoise_returns_the_page_unchanged(self):
        params = cd.init_params(0)
        img = (np.random.default_rng(2).random((40, 30)) * 255).astype(np.uint8)
        out = cd.denoise(img, params)
        self.assertTrue(np.array_equal(out, img))

    def test_output_keeps_shape_and_dtype(self):
        params = cd.init_params(0)
        img = np.full((23, 41), 200, dtype=np.uint8)
        out = cd.denoise(img, params)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)


@unittest.skipUnless(NUMPY, "numpy is not installed")
class TilingTests(unittest.TestCase):
    def test_tiled_denoise_matches_whole_page_denoise(self):
        """
        A page wider than one tile must come out identical to the same page
        denoised in a single pass. The network's receptive field is 7x7
        (three 3x3 layers), so the 16-pixel tile margin is more than enough
        for every written pixel to have been convolved with real neighbours
        rather than with zero padding - if the margin is ever trimmed too
        tight, this test is what notices.
        """
        rng = np.random.default_rng(4)
        params = cd.init_params(5)
        # A non-trivial network, or a zero residual would make any tiling
        # scheme look correct.
        params["w3"] = (rng.standard_normal(params["w3"].shape) * 0.05).astype(np.float32)
        img = (rng.random((cd.TILE + 200, cd.TILE + 120)) * 255).astype(np.uint8)

        tiled = cd.denoise(img, params)

        real_tile, real_overlap = cd.TILE, cd.TILE_OVERLAP
        try:
            cd.TILE = max(img.shape) + 64          # one tile covers everything
            whole = cd.denoise(img, params)
        finally:
            cd.TILE, cd.TILE_OVERLAP = real_tile, real_overlap

        self.assertEqual(tiled.shape, whole.shape)
        worst = int(np.abs(tiled.astype(int) - whole.astype(int)).max())
        self.assertLessEqual(worst, 1, f"tile seams differ by {worst} grey levels")


@unittest.skipUnless(NUMPY, "numpy is not installed")
class DegradationTests(unittest.TestCase):
    def test_missing_weights_disable_the_denoiser(self):
        self.assertIsNone(cd.load_params(os.path.join(tempfile.gettempdir(), "nope.npz")))

    def test_denoise_without_weights_returns_none_not_a_blank_page(self):
        """
        None means "keep the image you had". Returning zeros here would hand
        Tesseract an empty page and be recorded as a successful OCR run that
        found nothing - the silent failure this project refuses to produce.
        """
        path = os.path.join(tempfile.mkdtemp(prefix="cnn_"), "absent.npz")
        self.assertIsNone(cd.denoise(np.zeros((8, 8), np.uint8),
                                     cd.load_params(path)))

    def test_a_weights_file_missing_layers_is_rejected(self):
        tmp = tempfile.mkdtemp(prefix="cnn_")
        path = os.path.join(tmp, "partial.npz")
        np.savez(path, w1=np.zeros((9, 4), np.float32))
        self.assertIsNone(cd.load_params(path))

    def test_an_unreadable_weights_file_is_rejected(self):
        tmp = tempfile.mkdtemp(prefix="cnn_")
        path = os.path.join(tmp, "junk.npz")
        with open(path, "wb") as fh:
            fh.write(b"this is not an npz archive")
        self.assertIsNone(cd.load_params(path))

    def test_round_trip_through_disk_preserves_the_weights(self):
        params = cd.init_params(7)
        path = os.path.join(tempfile.mkdtemp(prefix="cnn_"), "w.npz")
        cd.save_params(params, path)
        back = cd.load_params(path)
        self.assertIsNotNone(back)
        for key in params:
            self.assertTrue(np.allclose(params[key], back[key]), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
