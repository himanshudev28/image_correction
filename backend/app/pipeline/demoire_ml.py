"""Optional ML de-moiré — ESDNet (Apache-2.0), CPU via ONNX Runtime.

Removes the rainbow/ripple moiré you get when photographing a screen. We run the
ESDNet model (from CVMI-Lab/UHDM, ECCV 2022, FHDMi-trained) directly through
onnxruntime — converted to ONNX from the released Apache-2.0 weights. It is heavy
(~1.5s at 768px on CPU), so it is OPT-IN per page, never part of the default flow.

Pipeline: downscale the page to a manageable long edge, RGB/255, pad to a multiple
of 32 (the net's requirement, padded with the FHDMi mean color), run, unpad, then
upscale back to the page size. Graceful: returns the input unchanged if disabled
or the model/onnxruntime is unavailable.
"""
from __future__ import annotations

import logging
import math
import threading

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger("scanner.demoire")

_session = None
_load_failed = False
_lock = threading.Lock()

# FHDMi per-channel mean used for padding (from the UHDM repo), RGB order.
_PAD_RGB = (0.3827, 0.4141, 0.3912)


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
        cfg = settings.demoire
        from pathlib import Path

        if not cfg.enabled or not Path(cfg.model_path).exists():
            _load_failed = True
            if cfg.enabled:
                logger.warning("De-moiré model not found at %s — feature disabled.", cfg.model_path)
            return None
        try:
            import onnxruntime as ort

            _session = ort.InferenceSession(cfg.model_path, providers=["CPUExecutionProvider"])
            logger.info("De-moiré model loaded: %s", cfg.model_path)
        except Exception:  # noqa: BLE001
            logger.exception("De-moiré model failed to load")
            _load_failed = True
            return None
    return _session


def demoire(bgr: np.ndarray) -> np.ndarray:
    """Return a de-moiréd copy of the page (or the input unchanged if unavailable)."""
    session = _get_session()
    if session is None:
        return bgr
    cfg = settings.demoire
    H, W = bgr.shape[:2]

    # downscale for tractable CPU inference
    scale = min(1.0, cfg.max_edge / max(H, W))
    work = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    h, w = work.shape[:2]

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(rgb, (2, 0, 1))[None]                      # 1x3xhxw

    # pad to a multiple of 32 (per-channel mean pad)
    ph = math.ceil(h / 32) * 32 - h
    pw = math.ceil(w / 32) * 32 - w
    top, bot = ph // 2, ph - ph // 2
    left, right = pw // 2, pw - pw // 2
    xp = np.empty((1, 3, h + ph, w + pw), np.float32)
    for c in range(3):
        xp[0, c] = np.pad(x[0, c], ((top, bot), (left, right)), mode="constant",
                          constant_values=_PAD_RGB[c])
    try:
        out = session.run(None, {"input": xp})[0]
    except Exception:  # noqa: BLE001
        logger.exception("De-moiré inference failed")
        return bgr

    out = out[0, :, top:top + h, left:left + w]                # unpad
    out = np.clip(out, 0, 1).transpose(1, 2, 0)
    cleaned = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if scale < 1:
        cleaned = cv2.resize(cleaned, (W, H), interpolation=cv2.INTER_CUBIC)
    return cleaned
