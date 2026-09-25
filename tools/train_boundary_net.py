#!/usr/bin/env python3
"""
Train the parcel-boundary segmenter.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The metric that matters
-----------------------
Boundary IoU is reported because it is what the loss optimises, but it is NOT
the headline. The headline is PARCEL RECALL through the existing classical
extractor: predict a boundary mask, hand it to backend/cadastral.vectorize,
and count how many parcels come back against the number the generator drew.

That is the real job. A model can score a respectable IoU with a two-pixel
gap left in every boundary and still recover zero parcels, because contour
extraction needs closure - which is the exact failure this model exists to
fix. Optimising IoU and reporting IoU would hide that completely.

Class balance
-------------
Boundary pixels are roughly 3% of a sheet, so plain BCE has a trivial
minimum: predict "background" everywhere and be 97% right. Dice is added to
force overlap with the thin positive structure, and BCE keeps the
probabilities calibrated rather than collapsing to a hard mask.

Honesty about the numbers
-------------------------
Two evaluation sets, and the second is the one to believe:

  val  - same damage families as training (bleed excluded from both).
         In-family. Measures whether the model learned the task.
  oof  - bleed ONLY, a family never seen in training.
         Out-of-family. Measures whether it learned anything transferable.

The precedent is tools/train_denoiser.py, where the gain fell from +5.8 dB
in-family to +1.6 dB out-of-family. Expect the same shape of result here,
and treat a real scanned sheet as harder still than either.

Usage
-----
    python3 tools/train_boundary_net.py --epochs 10
    python3 tools/train_boundary_net.py --epochs 1 --limit 40 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
except Exception as exc:
    print(f"torch, OpenCV and numpy are required to train: {exc}")
    sys.exit(1)

import boundary_net as bn
import cadastral

DATA = os.path.join(ROOT, "storage", "boundary_dataset")
CROP = 256          # training crop; inference tiles at 512 (interpolated pos emb)
BIN_THRESHOLD = 0.5


# --------------------------------------------------------------- data

def load_split(name, limit=None):
    base = os.path.join(DATA, name)
    with open(os.path.join(base, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    samples = manifest["samples"]
    if limit:
        samples = samples[:limit]
    return base, samples


def read_pair(base, meta):
    img = cv2.imread(os.path.join(base, "images", meta["file"]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(base, "masks", meta["file"]), cv2.IMREAD_GRAYSCALE)
    return img, mask


def random_crop(img, mask, size, rnd):
    h, w = img.shape[:2]
    if h <= size or w <= size:
        return img, mask
    y = rnd.randint(0, h - size)
    x = rnd.randint(0, w - size)
    return img[y:y + size, x:x + size], mask[y:y + size, x:x + size]


def batches(base, samples, batch_size, rnd, crop=CROP):
    order = list(range(len(samples)))
    rnd.shuffle(order)
    for i in range(0, len(order) - batch_size + 1, batch_size):
        xs, ys = [], []
        for j in order[i:i + batch_size]:
            img, mask = read_pair(base, samples[j])
            if img is None or mask is None:
                continue
            img, mask = random_crop(img, mask, crop, rnd)
            if rnd.random() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if rnd.random() < 0.5:
                img, mask = img[::-1].copy(), mask[::-1].copy()
            xs.append(img.astype(np.float32) / 255.0)
            ys.append((mask > 127).astype(np.float32))
        if not xs:
            continue
        yield (torch.from_numpy(np.stack(xs))[:, None],
               torch.from_numpy(np.stack(ys))[:, None])


# --------------------------------------------------------------- loss

def dice_bce(logits, target, eps=1.0):
    """
    Dice + BCE.

    Dice is computed on probabilities rather than a hard mask so it stays
    differentiable, and the eps is on both numerator and denominator so an
    empty crop - which happens, a crop can land inside one large parcel -
    gives a finite loss instead of a division by zero that poisons the run.
    """
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = 1.0 - (num / den).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice + bce, dice.item(), bce.item()


# --------------------------------------------------------------- metrics

def boundary_scores(prob, target, threshold=BIN_THRESHOLD):
    pred = prob >= threshold
    truth = target > 0.5
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    iou = inter / union if union else 1.0
    denom = pred.sum() + truth.sum()
    f1 = (2.0 * inter / denom) if denom else 1.0
    return float(iou), float(f1)


def parcels_from_mask(prob, tmp_path, threshold=BIN_THRESHOLD):
    """
    Run the real classical extractor over a predicted boundary mask.

    The mask is written as dark lines on white paper because that is what
    cadastral.vectorize expects to see; handing it a probability map directly
    would measure a different function than the one that runs in production.
    """
    ink = np.where(prob >= threshold, 0, 255).astype(np.uint8)
    if not cv2.imwrite(tmp_path, ink):
        return -1
    try:
        return len(cadastral.vectorize(tmp_path))
    except Exception:
        return -1


def evaluate(model, base, samples, tag, tmp_path, max_parcel_checks=60):
    """IoU/F1 on every sample; parcel recall on a capped subset (it is slow)."""
    model.eval()
    ious, f1s = [], []
    got_total = 0
    want_total = 0
    checked = 0
    with torch.no_grad():
        for idx, meta in enumerate(samples):
            img, mask = read_pair(base, meta)
            if img is None or mask is None:
                continue
            prob = bn.predict_boundary(model, img)
            truth = (mask > 127).astype(np.float32)
            iou, f1 = boundary_scores(prob, truth)
            ious.append(iou)
            f1s.append(f1)
            if checked < max_parcel_checks:
                got = parcels_from_mask(prob, tmp_path)
                if got >= 0:
                    got_total += min(got, meta["parcels"])
                    want_total += meta["parcels"]
                    checked += 1
    recall = (got_total / want_total) if want_total else 0.0
    return {
        "split": tag,
        "n": len(ious),
        "iou": float(np.mean(ious)) if ious else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "parcel_recall": float(recall),
        "parcel_checked": checked,
    }


def classical_baseline(base, samples, cap=60):
    """
    What the classical path scores on the SAME degraded sheets.

    Without this the model's numbers are unanchored - the whole claim is that
    it beats contour extraction on damaged linework, and that requires the
    comparison to run on identical inputs.
    """
    got = want = checked = 0
    for meta in samples[:cap]:
        path = os.path.join(base, "images", meta["file"])
        try:
            n = len(cadastral.vectorize(path))
        except Exception:
            continue
        got += min(n, meta["parcels"])
        want += meta["parcels"]
        checked += 1
    return {"parcel_recall": (got / want) if want else 0.0, "checked": checked}


# --------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N training samples.")
    ap.add_argument("--eval-limit", type=int, default=60)
    ap.add_argument("--out", default=bn.WEIGHTS_PATH)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to prove the loop learns at all.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rnd = random.Random(args.seed)
    tmp_path = os.path.join(DATA, "_eval_tmp.png")

    train_base, train_samples = load_split("train", args.limit)
    val_base, val_samples = load_split("val", args.eval_limit)
    oof_base, oof_samples = load_split("oof", args.eval_limit)

    model = bn.build_model()
    if model is None:
        print("torch unavailable:", bn.unavailable_reason())
        return 1
    print(f"model: {bn.parameter_count(model):,} parameters")
    print(f"train {len(train_samples)}  val {len(val_samples)}  oof {len(oof_samples)}")

    print("\nclassical baseline on the SAME sheets (parcel recall):")
    base_val = classical_baseline(val_base, val_samples, args.eval_limit)
    base_oof = classical_baseline(oof_base, oof_samples, args.eval_limit)
    print(f"  val  (in-family)      {base_val['parcel_recall']*100:5.1f}%  "
          f"n={base_val['checked']}")
    oof_only = json.load(open(os.path.join(DATA, "oof", "manifest.json"),
                              encoding="utf-8")).get("only") or "unknown"
    print(f"  oof  ({oof_only} only){' ' * max(0, 13 - len(oof_only))}"
          f"{base_oof['parcel_recall']*100:5.1f}%  n={base_oof['checked']}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses, dices = [], []
        steps = 0
        for x, y in batches(train_base, train_samples, args.batch, rnd):
            opt.zero_grad()
            logits = model(x)
            loss, dice, _bce = dice_bce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            dices.append(dice)
            steps += 1
            if args.smoke and steps >= 6:
                break
        sched.step()
        took = time.time() - t0

        val = evaluate(model, val_base, val_samples, "val", tmp_path,
                       args.eval_limit)
        print(f"\nepoch {epoch}/{args.epochs}  loss {np.mean(losses):.4f}  "
              f"dice {np.mean(dices):.4f}  steps {steps}  {took:.0f}s")
        print(f"  val  IoU {val['iou']:.4f}  F1 {val['f1']:.4f}  "
              f"parcel recall {val['parcel_recall']*100:5.1f}%")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val": val})

        if val["parcel_recall"] > best:
            best = val["parcel_recall"]
            torch.save({"model": model.state_dict(),
                        "arch": {"base": bn.BASE_CHANNELS, "depth": bn.DEPTH,
                                 "heads": bn.ATTN_HEADS, "layers": bn.ATTN_LAYERS},
                        "val": val}, args.out)
            print(f"  saved -> {args.out}")

    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)
    oof = evaluate(model, oof_base, oof_samples, "oof", tmp_path, args.eval_limit)
    print(f"  in-family  (val)  IoU {history[-1]['val']['iou']:.4f}  "
          f"parcel recall {history[-1]['val']['parcel_recall']*100:5.1f}%  "
          f"(classical {base_val['parcel_recall']*100:.1f}%)")
    print(f"  out-of-family     IoU {oof['iou']:.4f}  "
          f"parcel recall {oof['parcel_recall']*100:5.1f}%  "
          f"(classical {base_oof['parcel_recall']*100:.1f}%)")
    print("\n  The out-of-family row is the one to believe, and a real scanned")
    print("  sheet is harder than either. Nothing here licenses replacing the")
    print("  classical path until it is measured on real sheets.")

    with open(os.path.join(DATA, "training_log.json"), "w", encoding="utf-8") as fh:
        json.dump({"history": history, "oof": oof,
                   "classical": {"val": base_val, "oof": base_oof}}, fh, indent=1)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
