"""
Learned parcel-boundary segmentation: U-Net with a transformer bottleneck.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Why a learned model at all
--------------------------
The classical vectoriser (backend/cadastral.py) is excellent on a clean sheet
and brittle in one specific, measurable way. Over the bundled Bhu-Naksha
plot report:

    undegraded              7 of 7 parcels found
    5% line breakage        0 of 7
    20% line breakage       1 of 7
    85% fade                7 of 7
    noise sigma 25          2 of 7

Fading is survivable - thresholding handles contrast. Broken linework is
fatal, because contour extraction needs a topologically CLOSED boundary and a
single gap lets the whole region leak away. A fifty-year-old shajra has
cracked linework everywhere, so the classical path fails exactly where this
project is aimed.

Why a transformer in the bottleneck
-----------------------------------
This is not decoration. The failure being fixed is a GAP in a long straight
boundary, and closing it requires evidence from both sides of the gap - often
a hundred pixels apart. A convolution stack reaches that distance only by
stacking depth, which blurs the boundary it is trying to localise. Self
attention at the coarsest level relates every position to every other in one
step, which is precisely the "is there a collinear boundary continuing
further along?" question. The U-Net's skip connections then restore the
sharp localisation the bottleneck cannot carry.

Why this is optional, and stays optional
----------------------------------------
torch is imported lazily and the module degrades the same way every other
optional layer in this project does. It was blocked outright by this
machine's Application Control policy for most of the project's life, which is
why backend/cnn_denoiser.py is hand-written NumPy. It loads now, but the
classical path remains the default until the learned one measurably beats it
ON REAL SHEETS - synthetic-to-real transfer is where the denoiser's gain fell
from +5.8 dB in-family to +1.6 dB out-of-family, and there is no reason to
expect this to behave better.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

_torch = None
_nn = None
_IMPORT_ERROR: Optional[str] = None


def _load_torch():
    """Import torch on demand, remembering why it failed if it did."""
    global _torch, _nn, _IMPORT_ERROR
    if _torch is not None or _IMPORT_ERROR is not None:
        return _torch
    try:
        import torch
        import torch.nn as nn
        _torch, _nn = torch, nn
    except Exception as exc:                          # pragma: no cover
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    return _torch


def torch_available() -> bool:
    return _load_torch() is not None


def unavailable_reason() -> Optional[str]:
    _load_torch()
    return _IMPORT_ERROR


WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "storage", "models", "boundary_net.pt")

# Architecture. Deliberately small: this runs on CPU on a demo laptop, and a
# 512x512 sheet must segment in seconds rather than minutes.
BASE_CHANNELS = 24
DEPTH = 4                # 512 -> 256 -> 128 -> 64 -> 32
ATTN_HEADS = 4
ATTN_LAYERS = 2
TILE = 512


def build_model():
    """
    Construct the network. Returns None when torch is unavailable, matching
    the project's optional-dependency contract rather than raising.
    """
    torch = _load_torch()
    if torch is None:
        return None
    nn = _nn

    class ConvBlock(nn.Module):
        """Two 3x3 convolutions with GroupNorm.

        GroupNorm rather than BatchNorm because inference here is on ONE
        sheet at a time: BatchNorm's running statistics are estimated from
        training batches and a batch of one at inference is exactly the case
        it handles worst.
        """

        def __init__(self, cin, cout):
            super().__init__()
            groups = max(1, min(8, cout // 4))
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(groups, cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.GroupNorm(groups, cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.body(x)

    class AttentionBottleneck(nn.Module):
        """
        Self attention over the coarsest feature map.

        The positional embedding is learned and sized to the bottleneck grid,
        because the question being asked is geometric - "is a boundary
        continuing along this line?" - and attention without position cannot
        tell collinear from merely similar.

        It is INTERPOLATED to whatever grid it is handed rather than skipped
        when the size does not match. Skipping was the first implementation
        and it was quietly wrong in the worst way: training on 256px crops
        and inferring on 512px tiles meant the embedding applied at inference
        and not during training, so the model never learned to use the signal
        it was later given. Bilinear interpolation on the 2D grid - the same
        thing vision transformers do to change input resolution - keeps train
        and inference consistent at any size.
        """

        def __init__(self, channels, grid, heads, layers):
            super().__init__()
            self.grid = grid
            self.pos = nn.Parameter(_torch.randn(1, grid * grid, channels) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=channels, nhead=heads,
                dim_feedforward=channels * 2,
                dropout=0.0, batch_first=True, norm_first=True,
                activation="gelu")
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False)

        def _pos_for(self, h, w):
            if h * w == self.pos.shape[1]:
                return self.pos
            g = self.grid
            c = self.pos.shape[2]
            grid = self.pos.reshape(1, g, g, c).permute(0, 3, 1, 2)
            grid = _nn.functional.interpolate(
                grid, size=(h, w), mode="bilinear", align_corners=False)
            return grid.permute(0, 2, 3, 1).reshape(1, h * w, c)

        def forward(self, x):
            b, c, h, w = x.shape
            seq = x.flatten(2).transpose(1, 2)          # B, HW, C
            seq = seq + self._pos_for(h, w)
            seq = self.encoder(seq)
            return seq.transpose(1, 2).reshape(b, c, h, w)

    class BoundaryNet(nn.Module):
        def __init__(self, base=BASE_CHANNELS, depth=DEPTH):
            super().__init__()
            self.depth = depth
            chans = [base * (2 ** i) for i in range(depth)]

            self.downs = nn.ModuleList()
            cin = 1
            for c in chans:
                self.downs.append(ConvBlock(cin, c))
                cin = c
            self.pool = nn.MaxPool2d(2)

            bott = chans[-1] * 2
            self.mid = ConvBlock(chans[-1], bott)
            grid = TILE // (2 ** depth)
            self.attn = AttentionBottleneck(bott, grid, ATTN_HEADS, ATTN_LAYERS)

            self.ups = nn.ModuleList()
            self.up_convs = nn.ModuleList()
            cin = bott
            for c in reversed(chans):
                self.ups.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
                self.up_convs.append(ConvBlock(c * 2, c))
                cin = c
            # One logit per pixel: P(this pixel is on a parcel boundary).
            self.head = nn.Conv2d(chans[0], 1, 1)

        def forward(self, x):
            skips = []
            for block in self.downs:
                x = block(x)
                skips.append(x)
                x = self.pool(x)
            x = self.mid(x)
            x = self.attn(x)
            for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
                x = up(x)
                # Guard against an odd input size leaving a one-pixel
                # mismatch, which otherwise fails only at inference time on a
                # real sheet that is not a clean multiple of 16.
                if x.shape[-2:] != skip.shape[-2:]:
                    x = _nn.functional.interpolate(
                        x, size=skip.shape[-2:], mode="nearest")
                x = conv(_torch.cat([x, skip], dim=1))
            return self.head(x)

    return BoundaryNet()


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def load(path: str = WEIGHTS_PATH):
    """
    Load trained weights. Returns None when torch or the weights are absent -
    the caller falls back to the classical vectoriser.
    """
    torch = _load_torch()
    if torch is None or not os.path.exists(path):
        return None
    model = build_model()
    if model is None:
        return None
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


# Inference is tiled at the training tile size with an overlap, the same way
# backend/cnn_denoiser.py handles a full page.
#
# It is not only about memory here. The attention bottleneck carries a LEARNED
# positional embedding sized to the 32x32 bottleneck grid of a 512x512 tile.
# Fed a whole 1230x1438 sheet the grid becomes 77x90, the embedding no longer
# matches, and it is skipped - silently discarding the long-range reasoning
# the transformer exists to provide, which is the one thing this model is for.
# Tiling guarantees every forward pass sees the geometry it was trained on.
#
# The overlap is blended rather than butt-joined because a boundary crossing a
# tile seam would otherwise show a step in confidence exactly where the
# downstream contour extraction is most fragile.
INFER_TILE = TILE
INFER_OVERLAP = 64


def predict_boundary(model, gray) -> Optional["object"]:
    """
    Boundary probability map for one greyscale sheet, same size as the input.

    Tiled at the training size with a blended overlap. A sheet smaller than
    one tile is padded up and cropped back rather than resized - resizing
    would change the boundary stroke width the model was trained to expect.
    """
    torch = _load_torch()
    if torch is None or model is None or gray is None:
        return None
    import numpy as np

    h, w = gray.shape[:2]
    step = INFER_TILE - INFER_OVERLAP
    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    # A raised-cosine window, so overlapping tiles cross-fade instead of
    # stepping at the seam.
    #
    # The ramp is FLATTENED on any side of a tile that lies against the edge
    # of the image. The cosine starts at exactly zero, so a tile whose ramp
    # runs off the sheet gives the outermost row and column zero total
    # weight - and those pixels then come back as 0.0 no matter what the
    # model predicted. That is not a cosmetic seam: a cadastral sheet's frame
    # and its outermost parcel boundaries sit precisely there.
    _edge = None
    if INFER_OVERLAP > 0:
        _edge = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, INFER_OVERLAP,
                                                dtype=np.float32)))

    def _ramp(fade_low: bool, fade_high: bool):
        r = np.ones(INFER_TILE, dtype=np.float32)
        if _edge is None:
            return r
        if fade_low:
            r[:INFER_OVERLAP] = _edge
        if fade_high:
            r[-INFER_OVERLAP:] = _edge[::-1]
        return r

    ys = list(range(0, max(1, h - INFER_OVERLAP), step)) or [0]
    xs = list(range(0, max(1, w - INFER_OVERLAP), step)) or [0]

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + INFER_TILE, h)
                x1 = min(x0 + INFER_TILE, w)
                tile = gray[y0:y1, x0:x1]
                th, tw = tile.shape[:2]
                if th < INFER_TILE or tw < INFER_TILE:
                    tile = np.pad(tile, ((0, INFER_TILE - th),
                                         (0, INFER_TILE - tw)), mode="edge")
                x = torch.from_numpy(tile.astype("float32") / 255.0)[None, None]
                prob = torch.sigmoid(model(x))[0, 0].numpy()
                # Fade only towards a neighbouring tile, never towards the
                # edge of the sheet.
                window = np.outer(
                    _ramp(y0 > 0, y1 < h),
                    _ramp(x0 > 0, x1 < w))
                acc[y0:y1, x0:x1] += (prob * window)[:th, :tw]
                weight[y0:y1, x0:x1] += window[:th, :tw]

    # Every pixel is covered by at least one tile whose window is 1.0 at that
    # position, so the divisor is never the floor in practice; the floor is
    # kept only so a degenerate input cannot divide by zero.
    return acc / np.maximum(weight, 1e-6)


def capabilities() -> dict:
    """What this layer can do right now, for run.py --check."""
    return {
        "torch": torch_available(),
        "reason": unavailable_reason(),
        "weights_installed": os.path.exists(WEIGHTS_PATH),
        "weights_path": WEIGHTS_PATH,
    }
