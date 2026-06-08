"""Synthetic consent-form generator for development and tests.

Renders a form-like page (header, text lines, a signature line with a blue-ink
squiggle, and a red circular "stamp"), then optionally places it on a desk-like
background with perspective skew, a shadow gradient, blur, and noise — producing
a realistic "bad phone photo" with known ground truth.

No external assets; pure OpenCV/numpy so it runs anywhere.
"""
from __future__ import annotations

import cv2
import numpy as np

# Distinctive colors we assert survive the pipeline.
INK_BLUE_BGR = (200, 60, 30)     # a blue pen
STAMP_RED_BGR = (40, 40, 200)    # a red stamp


def render_form(w: int = 1000, h: int = 1400) -> np.ndarray:
    """A clean, white, frontal consent form."""
    page = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(page, "CONSENT FORM", (60, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (20, 20, 20), 3, cv2.LINE_AA)
    y = 180
    for _ in range(18):
        x2 = w - 60 - int(120 * abs(np.sin(y)))   # ragged line ends, deterministic
        cv2.line(page, (60, y), (x2, y), (60, 60, 60), 2)
        y += 45

    # signature line + blue-ink signature
    cv2.line(page, (60, h - 220), (520, h - 220), (0, 0, 0), 2)
    cv2.putText(page, "Signature", (60, h - 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 1, cv2.LINE_AA)
    pts = np.array([[80, h - 240], [160, h - 280], [240, h - 230],
                    [320, h - 285], [420, h - 235]], np.int32)
    cv2.polylines(page, [pts], False, INK_BLUE_BGR, 4, cv2.LINE_AA)

    # red circular stamp (bottom-right)
    cv2.circle(page, (w - 160, h - 200), 70, STAMP_RED_BGR, 4)
    cv2.putText(page, "APPROVED", (w - 220, h - 195), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, STAMP_RED_BGR, 2, cv2.LINE_AA)
    return page


def _add_shadow(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    grad = np.tile(np.linspace(0.55, 1.0, w, dtype=np.float32), (h, 1))
    grad = grad[:, :, None]
    return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)


def _draw_clutter(desk: np.ndarray, rng) -> None:
    """Add tile grid lines, a dark object, and a finger to mimic a real photo."""
    h, w = desk.shape[:2]
    # warm-toned floor
    desk[:] = (150, 170, 185)
    desk += (rng.integers(-15, 15, desk.shape)).astype(np.int16).clip(-15, 15).astype(np.uint8)
    # tile grid lines
    for x in range(0, w, 260):
        cv2.line(desk, (x + 40, 0), (x - 40, h), (120, 140, 155), 6)
    for y in range(0, h, 320):
        cv2.line(desk, (0, y), (w, y + 30), (120, 140, 155), 6)
    # a dark object near the top (like furniture)
    cv2.rectangle(desk, (int(w * 0.2), 0), (int(w * 0.8), int(h * 0.14)),
                  (60, 60, 65), -1)
    # a finger/thumb at the left edge (skin tone, higher saturation)
    cv2.ellipse(desk, (int(w * 0.04), int(h * 0.55)), (70, 150), 10, 0, 360,
                (120, 150, 200), -1)


def photo_of_form(canvas_w: int = 1500, canvas_h: int = 2000,
                  skew: float = 0.18, shadow: bool = True,
                  blur: bool = True, noise: bool = True,
                  cluttered: bool = False,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Place a form on a textured desk, apply perspective skew + degradations.

    Returns (photo_bgr, ground_truth_quad) where the quad is the 4 corners of the
    page in the photo, ordered tl, tr, br, bl. `cluttered=True` mimics a real
    in-hand phone capture (tiled floor, finger, dark object).
    """
    rng = np.random.default_rng(seed)
    form = render_form()
    fh, fw = form.shape[:2]

    # desk background: mid-gray with mild texture so there's contrast at the edges
    desk = np.full((canvas_h, canvas_w, 3), 120, np.uint8)
    desk += (rng.integers(-12, 12, desk.shape)).astype(np.int16).clip(-12, 12).astype(np.uint8)
    if cluttered:
        _draw_clutter(desk, rng)

    # source corners of the form, destination corners skewed inside the canvas
    margin_x = (canvas_w - fw) // 2
    margin_y = (canvas_h - fh) // 2
    dx = int(fw * skew)
    dy = int(fh * skew * 0.5)
    src = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
    dst = np.float32([
        [margin_x + dx, margin_y + dy],
        [margin_x + fw - dx // 2, margin_y],
        [margin_x + fw, margin_y + fh - dy],
        [margin_x + dx // 2, margin_y + fh],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(form, M, (canvas_w, canvas_h),
                                 borderValue=(255, 255, 255))
    mask = cv2.warpPerspective(np.full((fh, fw), 255, np.uint8), M,
                               (canvas_w, canvas_h))
    photo = desk.copy()
    photo[mask > 0] = warped[mask > 0]

    if shadow:
        photo = _add_shadow(photo)
    if blur:
        photo = cv2.GaussianBlur(photo, (3, 3), 0)
    if noise:
        n = rng.normal(0, 6, photo.shape).astype(np.int16)
        photo = np.clip(photo.astype(np.int16) + n, 0, 255).astype(np.uint8)

    return photo, dst
