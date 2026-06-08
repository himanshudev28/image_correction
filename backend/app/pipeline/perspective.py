"""Stage 5 — perspective correction (FR-15..FR-17).

  - Rotation-robust corner ordering: order by angle about the centroid, then pick
    the top-left as the point nearest the origin. Naive sum/diff ordering breaks on
    significantly rotated captures; angle-based ordering does not. (FR-15)
  - Full-resolution 4-point warp (FR-16).
  - Skip-if-flat: if the quad already fills the frame and skew is tiny, skip the
    warp to avoid resampling artifacts (FR-17).
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered [top-left, top-right, bottom-right, bottom-left],
    robust to rotation."""
    pts = np.asarray(pts, dtype="float32")
    c = pts.mean(axis=0)
    # In image coordinates (y grows downward) increasing arctan2 sweeps clockwise,
    # so sorting by angle yields clockwise order: tl, tr, br, bl.
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]
    # rotate so the corner nearest the origin (top-left) comes first — robust to rotation
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype("float32")


def _max_edge_skew_deg(rect: np.ndarray, shape: tuple[int, int]) -> float:
    """Largest deviation of the quad's edges from horizontal/vertical, in degrees."""
    tl, tr, br, bl = rect
    edges = [tr - tl, br - tr, bl - br, tl - bl]
    devs = []
    for dx, dy in edges:
        a = np.degrees(np.arctan2(dy, dx)) % 90.0
        devs.append(min(a, 90.0 - a))
    return float(max(devs))


def should_skip_warp(quad: np.ndarray, shape: tuple[int, int]) -> bool:
    cfg = settings.perspective
    h, w = shape[:2]
    rect = order_points(quad)
    coverage = cv2.contourArea(rect.astype("float32")) / float(h * w)
    skew = _max_edge_skew_deg(rect, shape)
    return coverage >= cfg.flat_coverage_frac and skew < cfg.flat_skew_deg


def quad_is_plausible(quad: np.ndarray) -> bool:
    """Reject quads that can't plausibly be a document under reasonable perspective
    (implausible corner angles or lopsided opposite sides). Such a quad almost
    always means the corners are wrong — better to deskew than ship a sheared warp."""
    cfg = settings.perspective
    rect = order_points(quad)
    # interior angles at each corner
    for i in range(4):
        p0, p1, p2 = rect[(i - 1) % 4], rect[i], rect[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))
        if ang < cfg.min_corner_angle or ang > cfg.max_corner_angle:
            return False
    tl, tr, br, bl = rect
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    if max(top, bottom) / (min(top, bottom) + 1e-6) > cfg.max_side_ratio:
        return False
    if max(left, right) / (min(left, right) + 1e-6) > cfg.max_side_ratio:
        return False
    return True


def expand_quad(quad: np.ndarray, frac: float, shape: tuple[int, int]) -> np.ndarray:
    """Push corners outward from the centroid by `frac` (a safety margin so the
    warp doesn't clip content at the page edge), clamped to the image bounds."""
    if frac <= 0:
        return quad
    c = quad.mean(axis=0)
    expanded = c + (quad - c) * (1.0 + frac)
    h, w = shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype("float32")


def warp(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = order_points(quad)
    tl, tr, br, bl = rect
    out_w = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    out_h = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    out_w, out_h = max(out_w, 1), max(out_h, 1)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h))


def deskew(bgr: np.ndarray) -> np.ndarray:
    """FR-14/17 fallback: estimate dominant text skew and rotate to level, no crop."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return bgr
    angles = []
    for rho_theta in lines[:100]:
        theta = rho_theta[0][1]
        deg = np.degrees(theta) - 90.0   # relative to horizontal
        if -45 < deg < 45:
            angles.append(deg)
    if not angles:
        return bgr
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
