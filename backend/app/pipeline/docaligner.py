"""Optional ML corner detector — DocAligner (Apache-2.0), CPU via ONNX Runtime.

This runs DocAligner's heatmap-regression model directly through onnxruntime — we
do NOT depend on the full `docaligner_docsaid`/`capybara` toolkit (which pulls in
a non-headless OpenCV, Flask, matplotlib, poppler, etc.). Just onnxruntime + the
single Apache-2.0 ONNX file, keeping the image path lean and commercially clean.

Model I/O (verified against the official package):
  input  'img'     : (1, 3, 256, 256) float32, BGR / 255, CHW
  output 'heatmap' : (1, 4, 128, 128) float32, one corner heatmap per channel

Decoding: resize each heatmap to the original frame, threshold, take the centroid
of the largest connected blob → that corner. Returns a 4x2 quad in full-res
coordinates, or None (caller falls back to classical / deskew).

Graceful by design (FR-13): if disabled, onnxruntime is unavailable, or the model
file is missing, every call returns None and the pipeline stays fully classical.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger("scanner.docaligner")

_session = None
_load_failed = False
_lock = threading.Lock()


def available() -> bool:
    return _get_session() is not None


def _get_session():
    global _session, _load_failed
    if _session is not None:
        return _session
    if _load_failed:
        return None
    with _lock:
        if _session is not None:
            return _session
        cfg = settings.docaligner
        if not cfg.enabled:
            _load_failed = True
            return None
        from pathlib import Path

        if not Path(cfg.model_path).exists():
            logger.warning("DocAligner model not found at %s — staying classical. "
                           "Run scripts/fetch_docaligner_model.py to enable.", cfg.model_path)
            _load_failed = True
            return None
        try:
            import onnxruntime as ort

            _session = ort.InferenceSession(cfg.model_path, providers=["CPUExecutionProvider"])
            logger.info("DocAligner model loaded: %s", cfg.model_path)
        except Exception:  # noqa: BLE001 — any load error → classical fallback
            logger.exception("DocAligner failed to load — staying classical")
            _load_failed = True
            return None
    return _session


def detect_corners(bgr: np.ndarray) -> np.ndarray | None:
    """Return a 4x2 quad (full-res) or None.

    The image is padded with a border before inference so the model sees a margin
    around the document — essential when the page fills the frame. Detected corners
    are mapped back and clamped to the real image bounds (a true corner can sit on
    or just past the frame edge)."""
    session = _get_session()
    if session is None:
        return None
    cfg = settings.docaligner
    h, w = bgr.shape[:2]

    pad = int(round(max(h, w) * cfg.pad_frac))
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    ph, pw = padded.shape[:2]

    x = cv2.resize(padded, (cfg.input_size, cfg.input_size))
    x = x.transpose(2, 0, 1).astype("float32")[None] / 255.0
    try:
        heatmaps = session.run(None, {"img": x})[0]  # (1, 4, 128, 128)
    except Exception:  # noqa: BLE001
        logger.exception("DocAligner inference failed")
        return None

    pts = []
    for ch in heatmaps[0][:4]:
        hm = cv2.resize(ch, (pw, ph))
        mask = (hm >= cfg.heatmap_threshold).astype("uint8")
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            return None  # a corner heatmap was empty → unreliable, bail to classical
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pts.append(centroids[largest])

    quad = np.array(pts, dtype="float32") - pad           # back to original coords
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)            # clamp to real bounds
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    if quad.shape != (4, 2) or not _plausible(quad, w, h):
        return None
    return quad


def _plausible(quad: np.ndarray, w: int, h: int) -> bool:
    bcfg = settings.boundary
    hull = cv2.convexHull(quad.astype("float32"))
    coverage = cv2.contourArea(hull) / float(w * h)
    if coverage < bcfg.min_area_frac or coverage > bcfg.max_coverage_frac:
        return False
    return len(hull) == 4  # all four points are hull vertices → proper convex quad
