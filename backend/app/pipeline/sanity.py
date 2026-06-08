"""Stage 9 — post-warp sanity check (FR-30).

A bad warp (wrong corners) often produces an implausible aspect ratio or a mostly
uniform/black image. Catch it here and flag for review instead of shipping it.
"""
from __future__ import annotations

import numpy as np

from app.config import settings


def check(bgr: np.ndarray) -> dict:
    cfg = settings.sanity
    h, w = bgr.shape[:2]
    flags: list[str] = []

    if h == 0 or w == 0:
        return {"passed": False, "flags": ["empty"], "aspect": 0.0, "uniform_frac": 1.0}

    aspect = max(h, w) / float(min(h, w))
    if not (cfg.min_aspect <= (w / h) <= cfg.max_aspect) and \
       not (cfg.min_aspect <= (h / w) <= cfg.max_aspect):
        flags.append("implausible_aspect")

    # uniformity: fraction of pixels at the modal gray value (cheap blank/black test)
    gray = bgr.mean(axis=2).astype("uint8")
    hist = np.bincount(gray.ravel(), minlength=256)
    uniform_frac = float(hist.max() / gray.size)
    if uniform_frac > cfg.max_uniform_frac:
        flags.append("mostly_uniform")

    return {
        "passed": len(flags) == 0,
        "flags": flags,
        "aspect": aspect,
        "uniform_frac": uniform_frac,
    }
