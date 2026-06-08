"""Per-image adaptive parameter selection — the core of "handles all images".

Instead of fixed denoise/sharpen strengths, MEASURE the warped page and choose
processing to match it. A clean shot is left sharp; a grainy one is denoised;
a screen photo is auto-routed to de-moiré. Thresholds are starting points and
must be calibrated on a real corpus (see adaptive-pipeline-upgrade.md §5/§6).
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast Gaussian-noise estimate. ~1 on a clean capture, high on a
    grainy low-light photo. Robust enough on mostly-flat document pages."""
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float64), -1, M, borderType=cv2.BORDER_REPLICATE)
    return float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def moire_score(bgr: np.ndarray, grid: int = 10) -> float:
    """Screen/moiré proxy: per-TILE chroma OSCILLATION (std) over paper tiles.

    Screen moiré is a spatial *oscillation* of color (the rainbow ripple), whereas a
    warm paper colour-cast is a near-uniform chroma. Measuring the per-tile chroma
    standard deviation (not the mean) separates the two — a uniform cast has low
    std, moiré has high std. We take the busiest paper tile (per-tile max)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128
    b = lab[:, :, 2].astype(np.float32) - 128
    chroma = np.sqrt(a * a + b * b)
    lowsat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1] < 40
    h, w = chroma.shape
    th, tw = max(1, h // grid), max(1, w // grid)
    best = 0.0
    for y in range(0, h - th + 1, th):
        for x in range(0, w - tw + 1, tw):
            m = lowsat[y:y + th, x:x + tw]
            if m.mean() > 0.5:                       # a mostly-paper/background tile
                vals = chroma[y:y + th, x:x + tw][m]
                if vals.size > 20:
                    best = max(best, float(vals.std()))
    return best


def select_params(bgr: np.ndarray, cfg) -> dict:
    """Decide denoise + sharpen + routing for THIS page."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sigma = estimate_noise_sigma(gray)
    mscore = moire_score(bgr)

    # denoise strength from measured noise (clean -> none; grainy -> NLM)
    if sigma < cfg.noise_skip:
        denoise = ("none", 0)
    elif sigma < cfg.noise_light:
        denoise = ("bilateral", cfg.bilat_light_sigma_color)
    elif sigma < cfg.noise_heavy:
        denoise = ("bilateral", cfg.bilat_strong_sigma_color)
    else:
        denoise = ("nlm", 0)

    # sharpen: SMALL radius always (no halos); amount tracks how much we softened
    amount = {"none": 0.35, "bilateral": 0.45, "nlm": 0.60}[denoise[0]]
    route_demoire = mscore >= cfg.moire_threshold
    if route_demoire:
        amount = min(amount, 0.40)                   # screen photos: don't over-sharpen here

    return {
        "noise_sigma": round(sigma, 2),
        "moire_score": round(mscore, 2),
        "denoise": denoise,                          # (mode, sigma_color)
        "sharpen": (amount, cfg.sharpen_sigma_small),  # (amount, radius)
        "route_demoire": route_demoire,
    }
