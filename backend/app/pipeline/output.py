"""Stage 8 — output mode (FR-24, FR-25).

Default is COLOR (or flattened grayscale) so signatures/stamps stay legible —
never binarized by default. B&W "scan look" via Sauvola is explicit opt-in.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def apply_mode(bgr: np.ndarray, mode: str = "color") -> np.ndarray:
    """Return a BGR image in the requested mode (always 3-channel for uniform
    downstream encoding)."""
    if mode == "color":
        return bgr
    if mode == "gray":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if mode == "bw":
        # FR-25: Sauvola adaptive threshold, opt-in only.
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        win = 25
        t = threshold_sauvola(g, window_size=win)
        bw = (g > t).astype("uint8") * 255
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unknown output mode: {mode}")
