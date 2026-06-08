"""Stage 3 — content-aware quality gate (FR-6..FR-10).

Two deliberately content-aware metrics so sparse, mostly-white consent forms are
not falsely rejected:
  - Sharpness via Tenengrad (mean squared Sobel gradient) restricted to the
    busiest tiles (the actual content), NOT a naive global variance-of-Laplacian
    that collapses on white pages. (FR-7)
  - Glare via per-tile saturation hotspots, NOT a single global near-white
    fraction that would flag every well-lit white page. (FR-8)

Default policy is accept-and-flag (FR-9): we never silently block; we attach
flags and a confidence penalty.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings


def _tenengrad_on_content(gray: np.ndarray) -> float:
    cfg = settings.gate
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = gx * gx + gy * gy
    # tile the magnitude map; keep the busiest tiles as "content"
    h, w = gray.shape
    grid = 8
    th, tw = max(1, h // grid), max(1, w // grid)
    tile_means = []
    for ty in range(0, h - th + 1, th):
        for tx in range(0, w - tw + 1, tw):
            tile_means.append(float(mag[ty:ty + th, tx:tx + tw].mean()))
    if not tile_means:
        return float(mag.mean())
    tile_means.sort(reverse=True)
    keep = max(1, int(len(tile_means) * cfg.content_tile_frac))
    return float(np.mean(tile_means[:keep]))


def _glare_hotspots(gray: np.ndarray) -> int:
    cfg = settings.gate
    h, w = gray.shape
    g = cfg.glare_tile_grid
    th, tw = max(1, h // g), max(1, w // g)
    hot = 0
    for ty in range(0, h - th + 1, th):
        for tx in range(0, w - tw + 1, tw):
            tile = gray[ty:ty + th, tx:tx + tw]
            sat_frac = float(np.mean(tile >= cfg.glare_pixel_value))
            if sat_frac >= cfg.glare_tile_frac:
                hot += 1
    return hot


def assess(bgr: np.ndarray) -> dict:
    """Return gate metrics + flags. Does not block; the runner decides policy."""
    cfg = settings.gate
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = _tenengrad_on_content(gray)
    hot_tiles = _glare_hotspots(gray)
    long_edge = max(bgr.shape[:2])

    flags: list[str] = []
    if sharp < cfg.tenengrad_min:
        flags.append("blur")
    if hot_tiles > cfg.glare_max_hot_tiles:
        flags.append("glare")
    if long_edge < settings.ingest.min_long_edge:
        flags.append("low_resolution")

    return {
        "sharpness": sharp,
        "glare_hot_tiles": hot_tiles,
        "long_edge": long_edge,
        "flags": flags,
        "passed": len(flags) == 0,
    }
