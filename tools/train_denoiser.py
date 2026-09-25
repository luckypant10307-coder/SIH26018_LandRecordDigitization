#!/usr/bin/env python3
"""
Train the custom CNN denoiser in backend/cnn_denoiser.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Supervised on paired (degraded, clean) page renders. The clean side is the
PDF rasterised directly; the degraded side is that same render put through
the blur / fade / noise recipe from tools/make_samples.py.

ROTATION IS DELIBERATELY EXCLUDED from the training degradation. A rotated
degraded page is no longer pixel-aligned with its clean target, so a
regression loss over the pair would be dominated by misalignment rather than
by noise, and the network would be punished for the one defect it cannot fix.
Skew is corrected upstream by the classical deskew in
ocr_engine.assess_and_preprocess; this network is trained for, and only for,
the two defects a convolution can actually undo.

TWO SPLITS, BECAUSE ONE IS NOT ENOUGH
-------------------------------------
  * DOCUMENT split - trains on samples 1-6 and reports on 7-11, so no page
    the network tuned on is scored.
  * DEGRADATION-STYLE holdout (--holdout) - trains on two of the three
    degradation families and reports on the third. This is the split that
    matters for the honest question, which is not "did it learn this corpus"
    but "did it learn noise, or did it learn OUR noise generator". A model
    that only wins in-family is a model that will do nothing on a real
    scanner.

Neither split can fully answer that, because every degradation here comes
from the same code. The number to trust is the one measured on a genuinely
real scan, and this corpus cannot supply one.

Usage:
    python3 tools/train_denoiser.py --gradcheck        # verify the backward pass
    python3 tools/train_denoiser.py --holdout heavy    # honest out-of-family read
    python3 tools/train_denoiser.py --install          # train on all, save weights
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cnn_denoiser as cd            # noqa: E402
import make_samples                  # noqa: E402

RENDER_DPI = 170                     # same as make_samples.make_scans
PATCH = 48
TRAIN_SAMPLES = {1, 2, 3, 4, 5, 6}   # same split as tools/train_learning_loop.py

# The degradation families from make_scans, with the rotation removed (see
# the module docstring). Ranges rather than fixed values so the network sees
# a spread of severities instead of three exact operating points.
STYLES: Dict[str, dict] = {
    "skew_blur": {"blur": (3, 3), "gain": (1.0, 1.0), "bias": (0.0, 0.0),
                  "noise": (7.0, 11.0)},
    "faded":     {"blur": (1, 1), "gain": (0.50, 0.60), "bias": (105.0, 119.0),
                  "noise": (10.0, 16.0)},
    "heavy":     {"blur": (5, 5), "gain": (0.57, 0.67), "bias": (88.0, 102.0),
                  "noise": (16.0, 24.0)},
}


def sample_index(name: str) -> int:
    return int(name.split("_")[1])


# --------------------------------------------------------------------------
# Paired data
# --------------------------------------------------------------------------

def render_clean(pdf_path: str) -> Optional[np.ndarray]:
    try:
        import pymupdf
    except Exception:
        try:
            import fitz as pymupdf          # noqa: N813
        except Exception:
            return None
    doc = pymupdf.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=RENDER_DPI)
    doc.close()
    buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2GRAY)
    return buf[:, :, 0].copy()


def degrade(clean: np.ndarray, style: str, rng: np.random.Generator) -> np.ndarray:
    """One draw from a degradation family. No rotation - see module docstring."""
    spec = STYLES[style]
    img = clean.astype(np.float32)
    k = spec["blur"][0]
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    gain = rng.uniform(*spec["gain"]) if spec["gain"][1] > spec["gain"][0] else spec["gain"][0]
    bias = rng.uniform(*spec["bias"]) if spec["bias"][1] > spec["bias"][0] else spec["bias"][0]
    img = img * gain + bias
    sigma = rng.uniform(*spec["noise"])
    img = img + rng.normal(0.0, sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def build_pairs(styles: List[str], samples: set, per_page: int,
                seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    (X, Y) patch stacks: X the degraded input, Y the residual to predict
    (degraded - clean), both float32 in [0,1] / [-1,1].

    Patches are drawn only where the CLEAN page has ink. A land record is
    mostly white paper, so uniform sampling would fill the batch with blank
    margins and train a network that is excellent at leaving nothing alone.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for spec in make_samples.SAMPLES:
        if sample_index(spec["name"]) not in samples:
            continue
        pdf = os.path.join(ROOT, "storage", "eval", spec["name"])
        if not os.path.exists(pdf):
            pdf = os.path.join(ROOT, "samples", spec["name"])
        clean = render_clean(pdf) if os.path.exists(pdf) else None
        if clean is None:
            continue
        h, w = clean.shape[:2]
        for style in styles:
            noisy = degrade(clean, style, rng)
            taken = 0
            attempts = 0
            while taken < per_page and attempts < per_page * 40:
                attempts += 1
                y0 = int(rng.integers(0, max(1, h - PATCH)))
                x0 = int(rng.integers(0, max(1, w - PATCH)))
                cpatch = clean[y0:y0 + PATCH, x0:x0 + PATCH]
                if cpatch.shape != (PATCH, PATCH) or cpatch.std() < 18.0:
                    continue                      # blank paper - skip
                npatch = noisy[y0:y0 + PATCH, x0:x0 + PATCH]
                xs.append(npatch.astype(np.float32) / 255.0)
                ys.append((npatch.astype(np.float32) - cpatch.astype(np.float32)) / 255.0)
                taken += 1
    if not xs:
        return np.zeros((0, PATCH, PATCH), np.float32), np.zeros((0, PATCH, PATCH), np.float32)
    return np.stack(xs), np.stack(ys)


# --------------------------------------------------------------------------
# Backward pass
# --------------------------------------------------------------------------

def backward(cache: list, dresidual, params: dict) -> dict:
    """
    Gradients for the three conv layers, given d(loss)/d(residual).

    Written against exactly the column layout cnn_denoiser.conv_forward
    produces, and checked numerically by --gradcheck, because a silently
    wrong gradient here would still "train" - to the wrong place.
    """
    grads: Dict[str, np.ndarray] = {}
    (x1, c1, h1, a1), (x2, c2, h2, a2), (x3, c3, h3) = cache

    # Layer 3 (linear)
    d3 = dresidual.reshape(dresidual.shape[0], -1).T          # (H*W, 1)
    grads["w3"] = c3.T @ d3
    grads["b3"] = d3.sum(axis=0)
    da2 = cd.col2im(d3 @ params["w3"].T, x3.shape)

    # Layer 2 (ReLU)
    dh2 = da2 * (h2 > 0)
    d2 = dh2.reshape(dh2.shape[0], -1).T                      # (H*W, C)
    grads["w2"] = c2.T @ d2
    grads["b2"] = d2.sum(axis=0)
    da1 = cd.col2im(d2 @ params["w2"].T, x2.shape)

    # Layer 1 (ReLU)
    dh1 = da1 * (h1 > 0)
    d1 = dh1.reshape(dh1.shape[0], -1).T
    grads["w1"] = c1.T @ d1
    grads["b1"] = d1.sum(axis=0)
    return grads


def loss_and_grads(x, y, params) -> Tuple[float, dict]:
    cache: list = []
    pred = cd.forward(x, params, cache)
    diff = pred - y
    n = diff.size
    loss = float((diff ** 2).sum() / n)
    return loss, backward(cache, (2.0 / n) * diff, params)


def gradcheck(seed: int = 3) -> bool:
    """
    Numerical vs analytic gradient on a tiny random patch, in float64.

    The dtype is the point. At float32 a central-difference estimate of a
    derivative carries around 1e-2 of relative noise, so the check can only
    pass with a tolerance loose enough to also pass a genuinely wrong
    gradient. Run in float64 it settles near 1e-9, which is tight enough that
    a sign error or a transposed index cannot slip through.
    """
    rng = np.random.default_rng(seed)
    params = {k: v.astype(np.float64) for k, v in cd.init_params(1).items()}
    # Zero-initialised last layer makes every upstream gradient identically
    # zero, which would let a broken backward pass pass this check by
    # accident. Perturb it so all three layers carry real signal.
    params["w3"] = rng.standard_normal(params["w3"].shape) * 0.1
    x = rng.random((1, 8, 8))
    y = rng.random((1, 8, 8)) * 0.1

    _, grads = loss_and_grads(x, y, params)
    worst = 0.0
    eps = 1e-6
    for key in ("w1", "b1", "w2", "b2", "w3", "b3"):
        flat = params[key].ravel()
        for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            original = float(flat[idx])
            flat[idx] = original + eps
            lp, _ = loss_and_grads(x, y, params)
            flat[idx] = original - eps
            lm, _ = loss_and_grads(x, y, params)
            flat[idx] = original
            numeric = (lp - lm) / (2 * eps)
            analytic = float(grads[key].ravel()[idx])
            denom = max(1e-6, abs(numeric) + abs(analytic))
            worst = max(worst, abs(numeric - analytic) / denom)
    print("  worst relative gradient error: %.3e" % worst)
    return worst < 1e-6


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(x, y, epochs: int, batch: int, lr: float, seed: int = 0,
          quiet: bool = False) -> dict:
    params = cd.init_params(seed)
    keys = list(params.keys())
    m = {k: np.zeros_like(params[k]) for k in keys}
    v = {k: np.zeros_like(params[k]) for k in keys}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    rng = np.random.default_rng(seed + 1)
    n = x.shape[0]

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total, count = 0.0, 0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            # Patches are accumulated one at a time: the conv primitives take
            # a single (C,H,W) volume, and a 48x48 patch is small enough that
            # the python-level loop is not what costs.
            acc = {k: np.zeros_like(params[k]) for k in keys}
            bloss = 0.0
            for i in idx:
                li, gi = loss_and_grads(x[i][np.newaxis, :, :],
                                        y[i][np.newaxis, :, :], params)
                bloss += li
                for k in keys:
                    acc[k] += gi[k].reshape(acc[k].shape)
            scale = 1.0 / len(idx)
            step += 1
            for k in keys:
                g = acc[k] * scale
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * (g * g)
                mhat = m[k] / (1 - b1 ** step)
                vhat = v[k] / (1 - b2 ** step)
                params[k] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
            total += bloss
            count += len(idx)
        if not quiet:
            print("    epoch %2d/%d  train mse %.6f" % (epoch, epochs, total / max(1, count)),
                  flush=True)
    return params


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    if mse <= 0:
        return 99.0
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def evaluate(params: dict, styles: List[str], samples: set, seed: int) -> dict:
    """Whole-page PSNR before and after, on pages the model never saw."""
    rng = np.random.default_rng(seed)
    rows = []
    for spec in make_samples.SAMPLES:
        if sample_index(spec["name"]) not in samples:
            continue
        pdf = os.path.join(ROOT, "storage", "eval", spec["name"])
        if not os.path.exists(pdf):
            pdf = os.path.join(ROOT, "samples", spec["name"])
        clean = render_clean(pdf) if os.path.exists(pdf) else None
        if clean is None:
            continue
        for style in styles:
            noisy = degrade(clean, style, rng)
            fixed = cd.denoise(noisy, params)
            rows.append((style, psnr(noisy, clean), psnr(fixed, clean)))
    out: Dict[str, dict] = {}
    for style in styles:
        vals = [(b, a) for s, b, a in rows if s == style]
        if vals:
            out[style] = {
                "pages": len(vals),
                "psnr_before": round(sum(b for b, _ in vals) / len(vals), 2),
                "psnr_after": round(sum(a for _, a in vals) / len(vals), 2),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gradcheck", action="store_true",
                    help="verify the backward pass numerically and exit")
    ap.add_argument("--holdout", choices=sorted(STYLES),
                    help="train without this degradation family and report on it")
    ap.add_argument("--install", action="store_true",
                    help="save the trained weights to storage/denoiser.npz")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patches", type=int, default=60,
                    help="patches drawn per page per degradation family")
    args = ap.parse_args()

    if args.gradcheck:
        print("gradient check:")
        ok = gradcheck()
        print("  " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1

    train_styles = [s for s in STYLES if s != args.holdout]
    print("1. building paired patches ...")
    print("   train styles: " + ", ".join(train_styles)
          + ("   holdout: " + args.holdout if args.holdout else ""))
    x, y = build_pairs(train_styles, TRAIN_SAMPLES, args.patches, seed=0)
    print("   %d patches of %dx%d from samples %s"
          % (x.shape[0], PATCH, PATCH, sorted(TRAIN_SAMPLES)))
    if x.shape[0] == 0:
        print("   ! no patches - is pymupdf installed and the corpus rendered?")
        return 1

    print("2. training ...")
    started = time.time()
    params = train(x, y, args.epochs, args.batch, args.lr)
    print("   done in %.1fs, %d parameters"
          % (time.time() - started, sum(int(v.size) for v in params.values())))

    held_docs = {sample_index(s["name"]) for s in make_samples.SAMPLES} - TRAIN_SAMPLES
    print("\n3. whole-page PSNR on HELD-OUT documents %s:" % sorted(held_docs))
    for style, r in evaluate(params, list(STYLES), held_docs, seed=7).items():
        tag = "  <-- HELD-OUT STYLE" if style == args.holdout else ""
        delta = r["psnr_after"] - r["psnr_before"]
        print("     %-10s %5.2f dB -> %5.2f dB  (%+.2f)%s"
              % (style, r["psnr_before"], r["psnr_after"], delta, tag))

    if args.install:
        cd.save_params(params)
        cd.reset_cache()
        print("\n  Weights saved to %s" % os.path.relpath(cd.MODEL_PATH, ROOT))
    else:
        print("\n  Weights NOT saved (pass --install to keep them).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
