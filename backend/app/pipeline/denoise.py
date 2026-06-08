"""Stage 7 — denoise (FR-22). Edge-preserving bilateral default.

Bilateral preserves thin strokes and signature lines (unlike Gaussian blur). NLM
is the optional higher-quality/slower mode (FR-23) — not the default given the
sub-second target. Applied per-channel-safe via cv2.bilateralFilter on color.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings


def denoise(bgr: np.ndarray, mode: str = "bilateral") -> np.ndarray:
    cfg = settings.denoise
    if mode == "nlm":
        # FR-23: optional, slower. Color-aware NLM.
        return cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
    return cv2.bilateralFilter(
        bgr, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space
    )


def sharpen(bgr: np.ndarray, amount: float | None = None) -> np.ndarray:
    """Gentle unsharp mask on the LAB *L* channel only.

    Sharpening luminance crisps text/handwriting edges without amplifying color
    noise or screen moiré (the a/b chroma channels are left untouched). `amount`
    overrides the default (used to sharpen harder after ML de-moiré).
    """
    cfg = settings.denoise
    amt = cfg.sharpen_amount if amount is None else amount
    if not amt or amt <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=cfg.sharpen_sigma)
    sharp = cv2.addWeighted(L, 1.0 + amt, blur, -amt, 0)
    lab = cv2.merge([sharp, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
