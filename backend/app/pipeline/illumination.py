"""Stage 6 — illumination flattening (FR-19..FR-21). THE headline design goal.

Operates on the L (lightness) channel in LAB and recombines with the original a/b
chroma, so blue ink and colored stamps SURVIVE. We never divide on grayscale,
which would throw color away.

Method:
  1. Estimate background illumination on L at LOW resolution (a morphological close
     on a downscaled copy), then upscale. This captures *large soft shadows* cheaply
     — a full-res kernel big enough to span a shadow blob would be very slow and a
     small one wouldn't remove it. Kernel scales with the estimate size (FR-20).
  2. Divide L by the background and rescale → flat lighting.
  3. White-point so the paper reads white, not muddy gray (FR-21).
  4. Gentle CLAHE finish.
  5. Median-blur the a/b chroma channels to suppress color moiré (screen photos)
     and speckle while keeping large ink/stamp regions intact.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings


def _estimate_background(L: np.ndarray) -> np.ndarray:
    """Low-res morphological-close background estimate, upscaled to full size."""
    cfg = settings.illumination
    h, w = L.shape
    long_edge = max(h, w)
    scale = min(1.0, cfg.bg_estimate_long_edge / long_edge)
    small = cv2.resize(L, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else L

    k = int(round(max(small.shape[:2]) * cfg.kernel_frac))
    k = max(cfg.kernel_min, k)
    if k % 2 == 0:
        k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=k / 4.0)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def flatten(bgr: np.ndarray, clahe: bool = True) -> np.ndarray:
    cfg = settings.illumination
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    bg = _estimate_background(L)

    # divide L by background and rescale to flatten shadows/uneven lighting
    L_flat = cv2.divide(L, bg, scale=255)

    # white-point: map the paper (high percentile) toward white so the background
    # is clean instead of muddy gray, without crushing darker ink (FR-21).
    hi = float(np.percentile(L_flat, cfg.white_point_pct))
    if hi > 1:
        gain = min(cfg.white_point_target / hi, cfg.white_point_max_gain)
        L_flat = np.clip(L_flat.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if clahe:
        cl = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                             tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        L_flat = cl.apply(L_flat)

    # chroma denoise: kill color moiré/speckle, keep large ink/stamp regions.
    if cfg.chroma_median and cfg.chroma_median >= 3:
        a = cv2.medianBlur(a, cfg.chroma_median)
        b = cv2.medianBlur(b, cfg.chroma_median)
    if cfg.chroma_blur_sigma and cfg.chroma_blur_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=cfg.chroma_blur_sigma)
        b = cv2.GaussianBlur(b, (0, 0), sigmaX=cfg.chroma_blur_sigma)

    merged = cv2.merge([L_flat, a, b])                       # keep original chroma
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
