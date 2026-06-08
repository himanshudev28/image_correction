"""Stage 6 — illumination flattening (FR-19..FR-21). THE headline design goal.

Operates on the L (lightness) channel in LAB and recombines with the original a/b
chroma, so blue ink and colored stamps SURVIVE. We never divide on grayscale,
which would throw color away.

Method:
  1. Estimate background illumination on L at LOW resolution (a morphological close
     on a downscaled copy), then upscale. This captures *large soft shadows* cheaply
     — a full-res kernel big enough to span a shadow blob would be very slow and a
     small one wouldn't remove it. Kernel scales with the estimate size (FR-20).
  2. Divide L by the background and rescale → flat lighting.
  3. Gentle CLAHE for local contrast.
  4. Content-adaptive "scan look" level stretch (Adobe Magic-Color style): map a low
     L percentile → near-black and a high percentile → pure white so paper reads
     clean white with punchy text. Aggressive preset for document-like pages, gentle
     for colored photos/cards (so they aren't washed out). L channel only — chroma
     untouched (FR-19).
  5. Median-blur the a/b chroma channels to suppress color moiré (screen photos)
     and speckle while keeping large ink/stamp regions intact.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings


def _estimate_background(L: np.ndarray) -> np.ndarray:
    """Low-res morphological-close background (local paper/bright level), upscaled.

    The CLOSE takes the local bright level, so it bridges over dark bands/text
    (keeping them dark when divided — white-on-dark text stays crisp) while still
    following the smooth shadow gradient so it gets divided out. Kernel must be
    large enough to span the widest dark feature (e.g. a header band)."""
    cfg = settings.illumination
    h, w = L.shape
    long_edge = max(h, w)
    scale = min(1.0, cfg.bg_estimate_long_edge / long_edge)
    small = cv2.resize(L, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else L

    k = int(round(max(small.shape[:2]) * cfg.bg_close_frac)) | 1   # odd
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=k / 3.0)   # smooth the estimate
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def _is_document(bgr: np.ndarray) -> bool:
    """True if the page is mostly bright, low-saturation paper (vs a colored photo/card)."""
    cfg = settings.illumination
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    paper_frac = float(np.mean((v > cfg.paper_bright_v) & (s < cfg.paper_low_sat)))
    return paper_frac >= cfg.paper_frac_threshold


def _scan_levels(L: np.ndarray, strong: bool) -> np.ndarray:
    """White-point/black-point level stretch on L (the 'scan look'). Maps a low
    percentile → black_target and a high percentile → 255, with a mild gamma."""
    cfg = settings.illumination
    if strong:
        wp, bp, bt, g = cfg.doc_white_pct, cfg.doc_black_pct, cfg.doc_black_target, cfg.doc_gamma
    else:
        wp, bp, bt, g = cfg.photo_white_pct, cfg.photo_black_pct, cfg.photo_black_target, cfg.photo_gamma
    lo = float(np.percentile(L, bp))
    hi = float(np.percentile(L, wp))
    if hi - lo < 1.0:
        return L
    out = (L.astype(np.float32) - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if g != 1.0:
        out = np.power(out, g)
    out = out * (255 - bt) + bt
    return np.clip(out, 0, 255).astype(np.uint8)


def flatten(bgr: np.ndarray, clahe: bool = True) -> np.ndarray:
    cfg = settings.illumination
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # classify once: document-like (mostly white paper) vs colored photo/card.
    strong = _is_document(bgr)

    bg = _estimate_background(L)

    # divide L by background and rescale to flatten shadows/uneven lighting
    L_flat = cv2.divide(L, bg, scale=255)

    if clahe:
        clip = cfg.doc_clahe_clip if strong else cfg.photo_clahe_clip
        cl = cv2.createCLAHE(clipLimit=clip,
                             tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        L_flat = cl.apply(L_flat)

    # "scan look": adaptive white/black-point stretch — strong for documents,
    # gentle for colored photos/cards.
    L_flat = _scan_levels(L_flat, strong=strong)

    # chroma denoise: kill color moiré/speckle, keep large ink/stamp regions.
    if cfg.chroma_median and cfg.chroma_median >= 3:
        a = cv2.medianBlur(a, cfg.chroma_median)
        b = cv2.medianBlur(b, cfg.chroma_median)
    if cfg.chroma_blur_sigma and cfg.chroma_blur_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=cfg.chroma_blur_sigma)
        b = cv2.GaussianBlur(b, (0, 0), sigmaX=cfg.chroma_blur_sigma)

    merged = cv2.merge([L_flat, a, b])                       # keep original chroma
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
