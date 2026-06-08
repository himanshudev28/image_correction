"""Stage 2 — non-photo detection (FR-5).

An already-clean scan/screenshot/rasterized-PDF page should NOT be put through the
full camera pipeline (boundary-find + warp would only degrade it). Heuristic: a
clean source already has a near-uniform bright background and low geometric skew.
We don't have the source filetype here (PDF pages look like images), so we judge
from pixels: high fraction of bright, low-saturation background => treat as clean.

Returns True if the page looks already-clean (route around crop/warp; still do a
gentle illumination touch-up downstream).
"""
from __future__ import annotations

import cv2
import numpy as np


def looks_already_clean(bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    # "paper-like" pixel: bright and low-saturation.
    bright = v > 200
    low_sat = s < 40
    paper_frac = float(np.mean(bright & low_sat))
    # A photo of a document on a desk has a substantial non-paper background;
    # a clean scan is almost all paper.
    return paper_frac > 0.75
