"""
Custom CNN denoiser for degraded land-record scans.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

A small residual convolutional network that removes scanner noise and
restores faded ink before Tesseract sees the page. Trained on paired
(degraded, clean) renders by tools/train_denoiser.py.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It handles NOISE and FADE. It does not handle SKEW, and no convolutional
denoiser does: rotating a page back is a geometric transform that needs the
angle estimated explicitly, not something a stack of 3x3 kernels can express.
Deskew therefore stays where it belongs - the classical estimator in
ocr_engine.assess_and_preprocess, which runs BEFORE this network. Claiming
otherwise would be the kind of overstatement this project refuses to make.

WHY IT IS WRITTEN IN NUMPY
--------------------------
PyTorch installs on this platform but cannot load: Windows Application
Control blocks torch/lib/shm.dll (WinError 4551), and the same policy blocks
part of scipy.signal. Rather than depend on something that cannot start, the
network is implemented directly - im2col convolution with an explicit
backward pass (see tools/train_denoiser.py). numpy and OpenCV are already
required by the OCR path, so this adds no new runtime dependency, and a full
page denoises in well under a second on CPU.

ARCHITECTURE (DnCNN-lite, residual)
-----------------------------------
    conv 3x3  1 -> 16   + ReLU
    conv 3x3 16 -> 16   + ReLU
    conv 3x3 16 ->  1   (linear)
    output = input - predicted_residual

The network predicts the NOISE, not the clean page. That is the DnCNN result:
the residual is close to zero almost everywhere, so it is a much easier target
than the image itself, and the identity mapping comes for free when the input
is already clean - which matters here, because most pages fed to this system
are not degraded at all and must not be damaged by "restoring" them.

HONEST LIMITS ON THE TRAINING DATA
----------------------------------
The only paired data available is synthetic: tools/make_samples.py degrades
clean renders with a known recipe (Gaussian blur, Gaussian noise, linear
fade). A network trained on that learns to invert THAT degradation. Real
scanner noise is different in kind - JPEG ringing, halftone, paper texture,
ink bleed - so accuracy measured on synthetic scans is an upper bound on what
a real scan would gain, and a document-level train/test split does not fix
that (both halves share the same degradation code). tools/train_denoiser.py
therefore also reports a held-out DEGRADATION STYLE, which is the closest
thing to an out-of-family check that this corpus can offer.

Degradation is honest, as everywhere else: if the weights file is absent or
unreadable, denoise() returns None and the caller keeps the unfiltered image.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

try:
    import numpy as _np
except Exception:                                    # pragma: no cover
    _np = None

NUMPY_AVAILABLE = _np is not None

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "storage", "denoiser.npz")

# Channel width and depth. Deliberately tiny: the corpus is small, the target
# (additive noise) is simple, and inference has to stay well inside the
# per-page budget the OCR path already spends on preprocessing.
CHANNELS = 16
KERNEL = 3

# Pages are denoised in overlapping tiles so peak memory stays bounded on a
# large scan; the overlap is trimmed off each tile so seams do not appear in
# the output the way they would with abutting tiles.
TILE = 512
TILE_OVERLAP = 16


# --------------------------------------------------------------------------
# Convolution primitives (shared with the trainer, which needs the backward
# pass; both are written against the same column layout so a gradient check
# in tests/test_cnn_denoiser.py validates what inference actually runs).
# --------------------------------------------------------------------------

def im2col(x, pad: int = 1):
    """
    (Cin, H, W) -> (H*W, Cin*9), one row per output pixel.

    Built by sliding the padded input by each of the nine kernel offsets, so
    col2im below is its exact transpose and the two cannot drift apart.

    The column dtype follows the input rather than being pinned to float32.
    Inference runs in float32 (denoise() casts), but the gradient check in
    tools/train_denoiser.py needs float64 to mean anything: at float32 the
    numerical estimate of a derivative carries ~1e-2 relative noise, which is
    larger than most real backward-pass bugs and would hide them.
    """
    cin, h, w = x.shape
    xp = _np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    cols = _np.empty((h * w, cin * KERNEL * KERNEL), dtype=x.dtype)
    k = 0
    for dy in range(KERNEL):
        for dx in range(KERNEL):
            patch = xp[:, dy:dy + h, dx:dx + w]
            cols[:, k * cin:(k + 1) * cin] = patch.reshape(cin, -1).T
            k += 1
    return cols


def col2im(cols, shape: Tuple[int, int, int], pad: int = 1):
    """Transpose of im2col: scatter-add columns back onto (Cin, H, W)."""
    cin, h, w = shape
    xp = _np.zeros((cin, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    k = 0
    for dy in range(KERNEL):
        for dx in range(KERNEL):
            block = cols[:, k * cin:(k + 1) * cin].T.reshape(cin, h, w)
            xp[:, dy:dy + h, dx:dx + w] += block
            k += 1
    return xp[:, pad:pad + h, pad:pad + w]


def conv_forward(x, weight, bias):
    """(Cin,H,W) x (Cin*9,Cout) -> (Cout,H,W), plus the columns for backward."""
    cin, h, w = x.shape
    cols = im2col(x)
    out = cols @ weight + bias
    return out.T.reshape(-1, h, w), cols


def init_params(seed: int = 0) -> dict:
    """
    He-initialised weights for the three layers.

    The last layer starts at ZERO so the untrained network predicts a
    zero residual, i.e. it starts as the identity. A denoiser that begins by
    scribbling on the page would have to learn its way back to "do no harm";
    starting there means every training step is a departure it had to earn.
    """
    rng = _np.random.default_rng(seed)
    shapes = [(1, CHANNELS), (CHANNELS, CHANNELS), (CHANNELS, 1)]
    params: Dict[str, object] = {}
    for i, (cin, cout) in enumerate(shapes, start=1):
        fan_in = cin * KERNEL * KERNEL
        if i == len(shapes):
            w = _np.zeros((fan_in, cout), dtype=_np.float32)
        else:
            w = (rng.standard_normal((fan_in, cout)) *
                 _np.sqrt(2.0 / fan_in)).astype(_np.float32)
        params[f"w{i}"] = w
        params[f"b{i}"] = _np.zeros((cout,), dtype=_np.float32)
    return params


def forward(x, params, cache: Optional[list] = None):
    """
    Predict the residual for one (1,H,W) float32 patch in [0,1].

    Returns the residual, NOT the cleaned image, so the trainer can regress
    it directly against (degraded - clean).
    """
    h1, c1 = conv_forward(x, params["w1"], params["b1"])
    a1 = _np.maximum(h1, 0.0)
    h2, c2 = conv_forward(a1, params["w2"], params["b2"])
    a2 = _np.maximum(h2, 0.0)
    h3, c3 = conv_forward(a2, params["w3"], params["b3"])
    if cache is not None:
        cache.extend([(x, c1, h1, a1), (a1, c2, h2, a2), (a2, c3, h3)])
    return h3


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_params(params: dict, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _np.savez_compressed(path, **params)


def load_params(path: str = MODEL_PATH) -> Optional[dict]:
    """The trained weights, or None if they are absent or malformed."""
    if _np is None or not os.path.exists(path):
        return None
    try:
        with _np.load(path) as data:
            params = {k: data[k].astype(_np.float32) for k in data.files}
    except Exception:
        return None
    expected = {f"{p}{i}" for i in (1, 2, 3) for p in ("w", "b")}
    if not expected.issubset(params.keys()):
        return None
    return params


_CACHED: Optional[dict] = None
_CACHE_TRIED = False


def available(path: str = MODEL_PATH) -> bool:
    return NUMPY_AVAILABLE and os.path.exists(path)


def _params(path: str = MODEL_PATH) -> Optional[dict]:
    """Load once and keep - a page must not pay for a disk read per call."""
    global _CACHED, _CACHE_TRIED
    if not _CACHE_TRIED:
        _CACHED = load_params(path)
        _CACHE_TRIED = True
    return _CACHED


def reset_cache() -> None:
    """Drop the cached weights (used by tests and after retraining)."""
    global _CACHED, _CACHE_TRIED
    _CACHED, _CACHE_TRIED = None, False


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

# Distinguishes "caller did not pass weights" from "caller passed weights and
# they were None". `params or _params()` conflated the two, so handing denoise
# the result of a FAILED load silently ran the installed model instead - the
# caller asked for one thing and got another, which is exactly the sort of
# quiet substitution that makes a measurement untrustworthy.
_UNSET = object()


def denoise(gray, params=_UNSET):
    """
    Denoise one uint8 grayscale page.

    Returns a uint8 array of the same shape, or None when the model is
    unavailable - which the caller must read as "keep the image you had",
    never as "the page is blank".

    Omitting `params` uses the installed weights; passing None explicitly
    means "no model", and returns None rather than falling back.
    """
    if _np is None or gray is None:
        return None
    if params is _UNSET:
        params = _params()
    if params is None:
        return None

    x = (gray.astype(_np.float32) / 255.0)
    h, w = x.shape[:2]
    out = _np.empty_like(x)

    step = TILE - 2 * TILE_OVERLAP
    if step <= 0:                                    # pragma: no cover
        step = TILE
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            # Read a tile with a margin, write back only its interior, so the
            # pixels near a seam were still convolved with real neighbours.
            ry0, rx0 = max(0, y0 - TILE_OVERLAP), max(0, x0 - TILE_OVERLAP)
            ry1 = min(h, y0 + step + TILE_OVERLAP)
            rx1 = min(w, x0 + step + TILE_OVERLAP)
            tile = x[ry0:ry1, rx0:rx1][_np.newaxis, :, :]
            residual = forward(_np.ascontiguousarray(tile), params)[0]
            cleaned = tile[0] - residual
            wy0, wx0 = y0 - ry0, x0 - rx0
            wy1 = min(wy0 + step, cleaned.shape[0])
            wx1 = min(wx0 + step, cleaned.shape[1])
            out[y0:y0 + (wy1 - wy0), x0:x0 + (wx1 - wx0)] = cleaned[wy0:wy1, wx0:wx1]

    return (_np.clip(out, 0.0, 1.0) * 255.0).astype(_np.uint8)


def describe(path: str = MODEL_PATH) -> dict:
    """Capability report, in the same shape the other optional modules use."""
    if not NUMPY_AVAILABLE:
        return {"available": False, "reason": "numpy is not installed"}
    if not os.path.exists(path):
        return {"available": False,
                "reason": "no trained weights - run tools/train_denoiser.py"}
    params = _params(path)
    if params is None:
        return {"available": False, "reason": "weights file could not be read"}
    return {
        "available": True,
        "channels": CHANNELS,
        "parameters": int(sum(int(v.size) for v in params.values())),
        "path": path,
    }
