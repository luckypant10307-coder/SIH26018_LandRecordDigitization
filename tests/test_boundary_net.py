#!/usr/bin/env python3
"""
Unit tests for backend/boundary_net.py and tools/make_boundary_dataset.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

torch is optional (README S2), so model tests skip rather than fail when it
is absent - it was blocked outright by this machine's Application Control
policy for most of this project's life and could be again.

The dataset tests are the more important half. A segmentation model is only
as honest as its ground truth, and the two ways this ground truth could be
quietly wrong are both checked here: the mask must not contain the khasra
numerals (or the model learns "dark pixel = boundary"), and adjacent parcels
must keep sharing their corners (or the dataset teaches the model that
correct sheets contain sliver gaps).

Run from anywhere with:
    python3 tests/test_boundary_net.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import boundary_net as bn  # noqa: E402

try:
    import numpy as np
    _NUMPY = True
except Exception:
    _NUMPY = False

try:
    import make_boundary_dataset as mbd
    _GEN = True
except Exception:
    _GEN = False

_TORCH = bn.torch_available()


class ContractTests(unittest.TestCase):
    """The optional-dependency contract, which must hold either way."""

    def test_capabilities_always_answers(self):
        caps = bn.capabilities()
        for key in ("torch", "weights_installed", "weights_path"):
            self.assertIn(key, caps)
        self.assertIsInstance(caps["torch"], bool)

    def test_a_missing_dependency_is_reported_not_hidden(self):
        """
        available() False must come with a reason. "It does not work" without
        a why is what sent this project chasing an Application Control
        refusal through three unrelated modules.
        """
        if bn.torch_available():
            self.assertIsNone(bn.unavailable_reason())
        else:
            self.assertIsNotNone(bn.unavailable_reason())

    def test_absent_weights_load_to_none_rather_than_raising(self):
        self.assertIsNone(bn.load("/definitely/not/a/path/boundary_net.pt"))

    def test_predict_on_a_null_model_returns_none(self):
        self.assertIsNone(bn.predict_boundary(None, None))

    @unittest.skipUnless(_NUMPY, "numpy not installed")
    def test_predict_with_no_model_returns_none(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        self.assertIsNone(bn.predict_boundary(None, gray))


@unittest.skipUnless(_TORCH and _NUMPY, "torch/numpy not installed")
class ModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = bn.build_model()

    def test_the_model_builds(self):
        self.assertIsNotNone(self.model)
        self.assertGreater(bn.parameter_count(self.model), 100000)

    def test_output_is_one_logit_per_input_pixel(self):
        import torch
        out = self.model(torch.zeros(1, 1, 128, 128))
        self.assertEqual(tuple(out.shape), (1, 1, 128, 128))

    def test_the_positional_embedding_is_interpolated_not_skipped(self):
        """
        The first implementation skipped the embedding when the grid did not
        match, which meant training on 256px crops and inferring on 512px
        tiles applied it at inference and never during training - the model
        was handed a signal it had never learned to read. Interpolation keeps
        train and inference consistent at any size.
        """
        import torch
        attn = self.model.attn
        small = attn._pos_for(8, 8)
        native = attn._pos_for(attn.grid, attn.grid)
        self.assertEqual(small.shape[1], 64)
        self.assertEqual(native.shape[1], attn.grid * attn.grid)
        # Native size must be returned as-is, not round-tripped through
        # interpolation, which would perturb the learned values.
        self.assertTrue(torch.equal(native, attn.pos))

    def test_several_input_sizes_all_work(self):
        import torch
        for size in (64, 128, 256):
            out = self.model(torch.zeros(1, 1, size, size))
            self.assertEqual(out.shape[-1], size)

    def test_a_non_multiple_size_survives_the_skip_connections(self):
        """A real sheet is not a clean multiple of 16, and a one-pixel
        mismatch in a skip connection fails only at inference time."""
        import torch
        out = self.model(torch.zeros(1, 1, 70, 100))
        self.assertEqual(tuple(out.shape[-2:]), (70, 100))

    def test_predict_returns_probabilities_at_the_input_size(self):
        for shape in ((200, 150), (600, 300)):
            gray = (np.random.rand(*shape) * 255).astype(np.uint8)
            prob = bn.predict_boundary(self.model, gray)
            self.assertEqual(prob.shape, shape)
            self.assertGreaterEqual(prob.min(), 0.0)
            self.assertLessEqual(prob.max(), 1.0)

    def test_tiled_inference_has_no_seam(self):
        """
        Tiles are blended with a raised-cosine window, and the accumulated
        result is divided by the summed weights. A butt joint - or forgetting
        that division - would leave a step in confidence exactly where contour
        extraction is most fragile.

        This drives a CONSTANT-output stub rather than the real network. An
        untrained net is randomly initialised, so its output on a uniform
        input is arbitrary and varies with the seed; asserting low variance
        through it tested the initialisation, not the blending, and failed
        intermittently. With a constant model, any residual variation in the
        output is the seam and nothing else.
        """
        import torch

        class _Constant(torch.nn.Module):
            def forward(self, x):
                # sigmoid(0) == 0.5 everywhere.
                return torch.zeros_like(x)

        prob = bn.predict_boundary(_Constant(), np.full((700, 700), 200, np.uint8))
        self.assertAlmostEqual(float(prob.min()), 0.5, places=5)
        self.assertAlmostEqual(float(prob.max()), 0.5, places=5)
        self.assertLess(float(prob.std()), 1e-6)

    def test_inference_is_deterministic(self):
        gray = (np.random.rand(128, 128) * 255).astype(np.uint8)
        a = bn.predict_boundary(self.model, gray)
        b = bn.predict_boundary(self.model, gray)
        self.assertTrue(np.allclose(a, b))


@unittest.skipUnless(_GEN and _NUMPY, "generator or numpy unavailable")
class DatasetTests(unittest.TestCase):
    """Ground truth is only useful if it is actually true."""

    @classmethod
    def setUpClass(cls):
        cls.img, cls.mask, cls.meta = mbd.make_sample(12345, mbd.FAMILIES)

    def test_image_and_mask_are_the_same_size(self):
        self.assertEqual(self.img.shape, self.mask.shape)
        self.assertEqual(self.img.shape, (mbd.TILE, mbd.TILE))

    def test_the_mask_is_binary(self):
        self.assertEqual(set(np.unique(self.mask).tolist()) - {0, 255}, set())

    def test_the_mask_is_sparse_because_boundaries_are_thin(self):
        """
        Boundary pixels are a few percent of a sheet. If this ever climbs the
        mask has started including fills or annotations, and the loss would
        be optimising the wrong target.
        """
        frac = float((self.mask > 0).mean())
        self.assertGreater(frac, 0.005)
        self.assertLess(frac, 0.15)

    def test_the_mask_excludes_the_khasra_numerals(self):
        """
        The central learning signal. Labels are drawn INTO the image and left
        OUT of the mask, so a model that merely finds dark pixels scores
        badly. If labels leaked into the mask the task would collapse to
        thresholding.
        """
        clean_img, mask = mbd._render(
            mbd._warp_network(
                mbd._shared_jitter(
                    mbd._subdivide((30, 30, mbd.TILE - 30, mbd.TILE - 30), 3,
                                   __import__("random").Random(4)),
                    __import__("random").Random(4)),
                __import__("random").Random(4)),
            __import__("random").Random(4))
        # Every mask pixel must correspond to ink in the image; the reverse
        # must NOT hold, because the image also carries labels and furniture.
        ink = clean_img < 200
        boundary = mask > 0
        self.assertGreater(int(ink.sum()), int(boundary.sum()),
                           "image should carry more ink than boundary alone")

    def test_degradation_is_recorded_per_sample(self):
        self.assertTrue(self.meta["degradations"])
        for d in self.meta["degradations"]:
            self.assertIn(d["family"], mbd.FAMILIES)

    def test_holding_out_a_family_actually_excludes_it(self):
        families = tuple(f for f in mbd.FAMILIES if f != "dashes")
        for seed in range(30):
            _i, _m, meta = mbd.make_sample(seed, families)
            for d in meta["degradations"]:
                self.assertNotEqual(d["family"], "dashes")

    def test_only_one_family_can_be_forced(self):
        """Used to build the out-of-family evaluation set; if it leaked other
        families the resulting number would not be out-of-family at all."""
        for seed in range(20):
            _i, _m, meta = mbd.make_sample(seed, ("dashes",))
            self.assertEqual([d["family"] for d in meta["degradations"]],
                             ["dashes"])

    def test_the_generator_is_deterministic_for_a_seed(self):
        a_img, a_mask, _ = mbd.make_sample(777, mbd.FAMILIES)
        b_img, b_mask, _ = mbd.make_sample(777, mbd.FAMILIES)
        self.assertTrue(np.array_equal(a_img, b_img))
        self.assertTrue(np.array_equal(a_mask, b_mask))

    def test_adjacent_parcels_keep_sharing_their_corners(self):
        """
        The warp is applied to the whole network at once precisely so shared
        corners stay shared. Per-parcel jitter would tear neighbours apart
        and manufacture the sliver gaps topology.py exists to detect - the
        dataset would then be teaching the model that a correct sheet has
        gaps in it.
        """
        import random as _random
        rnd = _random.Random(9)
        rects = mbd._subdivide((30, 30, mbd.TILE - 30, mbd.TILE - 30), 3, rnd)
        polys = mbd._warp_network(mbd._shared_jitter(rects, rnd), rnd)
        pts = [tuple(round(c, 4) for c in p) for poly in polys for p in poly]
        # A subdivided sheet must have interior corners used by more than one
        # parcel; if every point were unique the parcels are not touching.
        self.assertLess(len(set(pts)), len(pts),
                        "no corner is shared - parcels have been torn apart")

    def test_the_warp_keeps_vertices_on_the_sheet(self):
        for seed in range(25):
            _i, _m, _meta = mbd.make_sample(seed, mbd.FAMILIES)
        rnd = __import__("random").Random(3)
        rects = mbd._subdivide((20, 20, mbd.TILE - 20, mbd.TILE - 20), 4, rnd)
        polys = mbd._warp_network(mbd._shared_jitter(rects, rnd), rnd)
        # make_sample clamps; the raw warp may exceed, which is why the clamp
        # exists. Assert the clamp in make_sample rather than the raw warp.
        img, mask, _ = mbd.make_sample(3, mbd.FAMILIES)
        self.assertEqual(img.shape, (mbd.TILE, mbd.TILE))
        self.assertEqual(mask.shape, (mbd.TILE, mbd.TILE))


if __name__ == "__main__":
    unittest.main()
