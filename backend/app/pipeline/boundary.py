"""Stage 4 — boundary detection (FR-11..FR-14).

Robust, multi-strategy document boundary finder. Real phone photos (document held
in hand, cluttered floor, low-contrast bottom edge sitting on another white
surface) defeat the naive "single Canny + exactly-4-point contour" recipe, so we:

  1. Generate candidate quads from TWO independent strategies:
       - EDGE: auto-Canny + dilation + contours (good with visible margins).
       - BRIGHT: segment the bright, low-saturation paper region (good when the
         page contrasts with a darker background, even if its edges are soft).
  2. For each large contour, fit a quad via adaptive-epsilon approxPolyDP on the
     CONVEX HULL (tolerant of torn/curled/noisy edges); if that fails, fall back
     to the contour's minimum-area rotated RECTANGLE so we still crop.
  3. Score candidates by area + rectangularity, reject near-full-frame (that's the
     image border, not the page) and sub-min-area regions (FR-12), and return the
     best one — or None (caller then deskews the whole frame, FR-14).

Returns (quad[4,2] full-res, method) where method is "approx" | "rect", or
(None, None).
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings


def find_quad(bgr: np.ndarray, scale: float = 1.0):
    """`bgr` is the (downscaled) detection image; `scale` maps coords back to
    full-res (full = detection / scale). Returns (quad, method) or (None, None)."""
    cfg = settings.boundary
    h, w = bgr.shape[:2]
    frame_area = float(h * w)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)   # smooth texture, keep edges

    candidates = []
    candidates += _edge_candidates(gray, frame_area, cfg)
    candidates += _bright_candidates(bgr, frame_area, cfg)

    if not candidates:
        return None, None

    best = max(candidates, key=lambda c: c["score"])
    quad = best["quad"].astype("float32") / scale
    return quad, best["method"]


def _auto_canny(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(img))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lo, hi)


def _edge_candidates(gray: np.ndarray, frame_area: float, cfg) -> list[dict]:
    edges = _auto_canny(gray)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_candidates(cnts, frame_area, cfg)


def _bright_candidates(bgr: np.ndarray, frame_area: float, cfg) -> list[dict]:
    """Segment the paper as the near-white region: high value AND LOW saturation.

    Low saturation is the key discriminator — white paper sits near 0, while warm
    surfaces (wood/tile/beige floor) carry visible saturation even when bright.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    paper = ((s < cfg.paper_max_sat) & (v > cfg.paper_min_val)).astype(np.uint8) * 255
    k = np.ones((9, 9), np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, k, iterations=1)   # drop specks first
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k, iterations=3)  # fill text/holes
    cnts, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_candidates(cnts, frame_area, cfg)


def _contours_to_candidates(cnts, frame_area: float, cfg) -> list[dict]:
    out: list[dict] = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[: cfg.top_contours]:
        area = cv2.contourArea(c)
        if area < cfg.min_area_frac * frame_area:
            continue
        quad, method = _fit_quad(c)
        if quad is None:
            continue
        qarea = cv2.contourArea(quad.astype("float32"))
        coverage = qarea / frame_area
        if coverage < cfg.min_area_frac or coverage > cfg.max_coverage_frac:
            continue  # reject sub-regions and the full image border
        if not _convex_enough(quad):
            continue
        # rectangularity: how well the contour fills its own min-area rect (1.0 = perfect)
        rect = cv2.minAreaRect(c)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0.0
        # prefer larger, more rectangular, exact-approx over rect-fallback
        score = coverage * (0.5 + 0.5 * rectangularity) * (1.0 if method == "approx" else 0.85)
        out.append({"quad": quad, "method": method, "score": score})
    return out


def _fit_quad(contour) -> tuple[np.ndarray | None, str | None]:
    """Adaptive-epsilon quad fit on the convex hull; minAreaRect fallback."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32"), "approx"
    # fallback: rotated bounding rectangle (still a clean crop, just less tight)
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return box.astype("float32"), "rect"


def _convex_enough(quad: np.ndarray) -> bool:
    q = quad.reshape(-1, 1, 2).astype(np.int32)
    return cv2.isContourConvex(q)
