"""Pipeline orchestrator — the fully-automatic "upload and it's done" path.

Given one BGR page, runs: gate -> (nonphoto?) -> boundary -> perspective ->
illumination -> denoise -> output mode -> post-warp sanity, and returns the
cleaned page plus metadata (confidence, flags, the quad used, per-stage timings).

The same function serves both image and PDF-page inputs (ingest already produced
BGR arrays), and is reused for manual re-crop (caller passes an explicit quad).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.pipeline import (
    boundary,
    demoire_ml,
    denoise as denoise_stage,
    docaligner,
    gate as gate_stage,
    illumination,
    nonphoto,
    output as output_stage,
    perspective,
    sanity,
)
from app.pipeline.ingest import downscale_for_detection


@dataclass
class PageResult:
    image: np.ndarray                       # cleaned BGR page
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)
    quad: list | None = None                # normalized 0..1 corners used (or None)
    timings_ms: dict = field(default_factory=dict)
    mode: str = "color"
    demoire: bool = False


def _normalize_quad(quad: np.ndarray, shape) -> list:
    h, w = shape[:2]
    return [[float(x) / w, float(y) / h] for x, y in quad]


def _denormalize_quad(norm_quad: list, shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[x * w, y * h] for x, y in norm_quad], dtype="float32")


def process_page(bgr: np.ndarray, mode: str = "color",
                 forced_quad: list | None = None,
                 denoise_mode: str = "bilateral",
                 demoire: bool = False) -> PageResult:
    """Run the auto pipeline on a single page.

    forced_quad: normalized [[x,y]*4] from a manual re-crop; bypasses auto boundary.
    demoire: opt-in ML de-moiré (ESDNet) on the warped page — for screen photos.
    """
    t = {}
    flags: list[str] = []
    confidence = 1.0

    # --- quality gate (accept-and-flag default) ---
    t0 = time.perf_counter()
    g = gate_stage.assess(bgr)
    t["gate"] = (time.perf_counter() - t0) * 1000
    flags.extend(g["flags"])
    confidence -= 0.15 * len(g["flags"])

    # --- boundary + perspective ---
    t0 = time.perf_counter()
    used_quad = None
    if forced_quad is not None:
        quad = _denormalize_quad(forced_quad, bgr.shape)
        warped = perspective.warp(bgr, quad)
        used_quad = forced_quad
    else:
        already_clean = nonphoto.looks_already_clean(bgr)
        small, scale = downscale_for_detection(bgr)
        quad, method = None, None
        if not already_clean:
            # DocAligner is PRIMARY when available — on real photos it is far more
            # reliable than classical, which often returns a confident-but-wrong quad.
            dq = docaligner.detect_corners(bgr)
            if dq is not None:
                quad, method = dq, "docaligner"
            else:
                # fallback: classical, but only if it produces a plausible shape
                cq, cm = boundary.find_quad(small, scale)
                if cq is not None and perspective.quad_is_plausible(cq):
                    quad, method = cq, cm
        if quad is None:
            # FR-14 fallback: deskew-only on the whole frame
            warped = perspective.deskew(bgr)
            if not already_clean:
                flags.append("boundary_not_found")
                confidence -= 0.25
        elif perspective.should_skip_warp(quad, bgr.shape):
            warped = perspective.deskew(bgr)         # FR-17 skip-if-flat
            used_quad = _normalize_quad(perspective.order_points(quad), bgr.shape)
        elif not perspective.quad_is_plausible(quad):
            # corners look wrong (sheared/lopsided) -> don't ship a bad warp
            warped = perspective.deskew(bgr)
            flags.append("geometry_uncertain")
            confidence -= 0.3
            used_quad = _normalize_quad(perspective.order_points(quad), bgr.shape)
        else:
            # outward safety margin so we don't clip content at the edge — tight for
            # the precise ML path, larger for the less-precise classical fallback.
            margin = (settings.perspective.docaligner_crop_margin_frac
                      if method == "docaligner"
                      else settings.perspective.crop_margin_frac)
            quad = perspective.expand_quad(quad, margin, bgr.shape)
            warped = perspective.warp(bgr, quad)
            used_quad = _normalize_quad(perspective.order_points(quad), bgr.shape)
            # a rotated-rect fallback is a looser crop than a true 4-corner fit
            if method == "rect":
                flags.append("approx_crop")
                confidence -= 0.1
    t["boundary_perspective"] = (time.perf_counter() - t0) * 1000

    # --- optional ML de-moiré (opt-in; for screen photos) ---
    if demoire:
        t0 = time.perf_counter()
        warped = demoire_ml.demoire(warped)
        t["demoire"] = (time.perf_counter() - t0) * 1000

    # --- illumination (color-preserving) ---
    t0 = time.perf_counter()
    scan_white = settings.illumination.demoire_white_pct if demoire else None
    flat = illumination.flatten(warped, scan_white_pct=scan_white)
    t["illumination"] = (time.perf_counter() - t0) * 1000

    # --- denoise + luminance sharpen (sharper after de-moiré, which softens) ---
    t0 = time.perf_counter()
    clean = denoise_stage.denoise(flat, mode=denoise_mode)
    sharpen_amt = settings.denoise.demoire_sharpen_amount if demoire else None
    clean = denoise_stage.sharpen(clean, amount=sharpen_amt)
    t["denoise"] = (time.perf_counter() - t0) * 1000

    # --- output mode ---
    t0 = time.perf_counter()
    final = output_stage.apply_mode(clean, mode=mode)
    t["output"] = (time.perf_counter() - t0) * 1000

    # --- post-warp sanity ---
    s = sanity.check(final)
    if not s["passed"]:
        flags.extend(f"sanity_{f}" for f in s["flags"])
        confidence -= 0.3

    confidence = max(0.0, min(1.0, confidence))
    return PageResult(
        image=final,
        confidence=round(confidence, 3),
        flags=flags,
        quad=used_quad,
        timings_ms={k: round(v, 1) for k, v in t.items()},
        mode=mode,
        demoire=demoire,
    )
