"""Central configuration — all pipeline thresholds and policy live here.

FR-10: thresholds shall be configurable and calibrated to capture resolution;
defaults documented, never hard-coded magic numbers scattered through the code.
Every tunable the pipeline uses is named here with a one-line rationale.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- storage / runtime layout (PoC: local filesystem) ---
BASE_DIR = Path(os.environ.get("SCANNER_DATA_DIR", Path(__file__).resolve().parent.parent / "var"))
DB_PATH = BASE_DIR / "scanner.db"
STORAGE_DIR = BASE_DIR / "storage"   # originals/, processed/, pdfs/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# --- dev wiring ---
# Vite dev server origin; CORS allows this in dev (prod serves the built SPA same-origin).
DEV_FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@dataclass(frozen=True)
class UploadLimits:
    """FR-4 — upload hardening (decompression-bomb + abuse guards)."""
    max_bytes: int = 40 * 1024 * 1024          # 40 MB per file
    max_pixels: int = 60_000_000               # Image.MAX_IMAGE_PIXELS guard (~60 MP)
    max_pages_per_pdf: int = 50                 # reject absurd PDFs
    # allow-listed input types; HEIC is transcoded to a normal raster on ingest.
    allowed_mime: tuple[str, ...] = (
        "image/jpeg", "image/png", "image/heic", "image/heif", "application/pdf",
    )


@dataclass(frozen=True)
class IngestConfig:
    detect_long_edge: int = 1500   # downscale copy used for DETECTION only (speed)
    pdf_render_dpi: int = 200      # rasterization DPI for input-PDF pages
    min_long_edge: int = 800       # below this, flag low-resolution (FR-6)
    preview_long_edge: int = 1200  # rasterized preview size for born-digital pages
    # born-digital detection: a PDF page with this many extractable text chars is
    # treated as already-clean and passed through losslessly (no rasterize/re-encode).
    born_digital_min_chars: int = 30


@dataclass(frozen=True)
class GateConfig:
    """Content-aware quality gate (FR-6..FR-10). Defaults are starting points and
    MUST be calibrated on representative captures (see research report caveats)."""
    # FR-7: content-aware sharpness. We use Tenengrad (mean squared Sobel gradient)
    # restricted to high-content regions so sparse white forms aren't false-rejected.
    tenengrad_min: float = 500.0
    content_tile_frac: float = 0.10   # top X% busiest tiles define the "content" region
    # FR-8: local glare. Tile the image; flag if any tile is mostly blown-out, rather
    # than relying on a single global near-white fraction (white pages are fine).
    glare_tile_grid: int = 8          # 8x8 tiles
    glare_pixel_value: int = 250      # >= this = saturated
    glare_tile_frac: float = 0.60     # a tile this saturated counts as a glare hotspot
    glare_max_hot_tiles: int = 2      # more than this many hot tiles -> glare flag


@dataclass(frozen=True)
class BoundaryConfig:
    """FR-11..FR-14 — bordered Canny+contour, convex + min-area quad."""
    border_pad: int = 8               # pad so edge-touching docs still close (FR-11)
    canny_low: int = 50
    canny_high: int = 150
    gaussian_ksize: int = 5
    approx_eps_frac: float = 0.02     # approxPolyDP epsilon as frac of perimeter
    min_area_frac: float = 0.18       # quad must cover >= this frac of frame (FR-12)
    max_coverage_frac: float = 0.985  # above this it's the image border, not the page
    top_contours: int = 8             # inspect the N largest contours
    # brightness-segmentation strategy: near-white = low saturation AND high value
    paper_max_sat: int = 40           # paper saturation is near 0; warm floors are higher
    paper_min_val: int = 150          # paper is bright


@dataclass(frozen=True)
class DocAlignerConfig:
    """FR-13 — optional ML corner detector (DocAligner, Apache-2.0), CPU/ONNX.

    Triggered only when classical detection is weak (no quad / loose rect fallback).
    Graceful: if disabled, onnxruntime is missing, or the model file is absent, the
    pipeline silently falls back to classical. Model is heatmap regression: input
    1x3x256x256 (BGR/255), output 1x4x128x128 (one heatmap per corner)."""
    enabled: bool = os.environ.get("SCANNER_DOCALIGNER", "1") != "0"
    model_path: str = os.environ.get(
        "SCANNER_DOCALIGNER_MODEL", str(MODELS_DIR / "docaligner_fastvit_sa24.onnx"))
    # auto-download the model on first run if missing (clone-and-run convenience).
    # Set SCANNER_DOCALIGNER_AUTODOWNLOAD=0 for air-gapped deploys (pre-place the file).
    auto_download: bool = os.environ.get("SCANNER_DOCALIGNER_AUTODOWNLOAD", "1") != "0"
    model_file_id: str = "14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl"  # DocAligner fastvit_sa24 (Google Drive)
    input_size: int = 256
    heatmap_threshold: float = 0.2
    # Pad the image with a border before inference. The model expects a margin
    # around the document; without this it finds nothing when the page fills the
    # frame (and weak corners near the edge get missed). ~30% recovers both.
    pad_frac: float = 0.30


@dataclass(frozen=True)
class PerspectiveConfig:
    """FR-15..FR-17 — robust ordering, full-res warp, skip-if-flat."""
    # skip-if-flat: if the detected quad already fills the frame and is barely
    # skewed, skip warping to avoid resampling artifacts.
    flat_coverage_frac: float = 0.92  # quad covers >= this frac of frame
    flat_skew_deg: float = 2.0        # and max edge angle deviation < this
    # outward safety margin on AUTO-detected corners before warping, so we don't
    # shave off content right at the page edge ("crops too much"). Expressed as a
    # fraction of the quad size; clamped to the image. Manual re-crop is exact.
    crop_margin_frac: float = float(os.environ.get("SCANNER_CROP_MARGIN", "0.05"))
    # DocAligner corners are precise → tight crop (Adobe-like); the larger margin
    # above is the safety net for the less-precise classical fallback.
    docaligner_crop_margin_frac: float = 0.02
    # plausibility guard: a detected quad whose corner angles or side ratios are
    # implausible for a document is treated as a BAD detection -> deskew instead of
    # shipping a sheared/clipped warp, and flag for manual Adjust.
    min_corner_angle: float = 55.0
    max_corner_angle: float = 125.0
    max_side_ratio: float = 2.6       # opposite sides shouldn't differ beyond this


@dataclass(frozen=True)
class IlluminationConfig:
    """FR-19..FR-21 — color-preserving (LAB L-channel) flatten, resolution-scaled kernel."""
    # Background illumination is estimated at LOW resolution then upscaled — this
    # captures large soft shadows cheaply (a huge full-res kernel would be slow and
    # miss big blobs). The kernel is relative to the small estimate (FR-20).
    bg_estimate_long_edge: int = 256  # estimate background at this size
    bg_blur_sigma_frac: float = 0.10  # gaussian-blur background ~10% of long edge:
                                      # captures smooth shadow gradients to divide out
    clahe_clip: float = 1.0           # gentle local-contrast finish, won't crush ink
    clahe_grid: int = 8
    # "scan look" level stretch (Adobe Magic-Color style) on the L channel only —
    # chroma (a/b) untouched so ink/stamp color survives (FR-19). Content-adaptive:
    # a document-like page (mostly bright paper) gets the aggressive preset; a
    # colored photo/card gets the gentle one so it isn't washed out.
    paper_frac_threshold: float = 0.45  # >= this frac of bright low-sat px → "document"
    paper_bright_v: int = 180           # HSV V above this = bright
    paper_low_sat: int = 45             # HSV S below this = low-saturation (paper-like)
    # document preset: whiten paper but keep darks SOFT so dark header bands and
    # small text stay readable (not crushed to black). High white-point cleans the
    # paper; a raised black floor preserves legibility (Adobe's clean-but-readable look).
    doc_white_pct: float = 78.0         # this L percentile and above → pure white
    doc_black_pct: float = 4.0          # only the darkest few % map to the floor
    doc_black_target: int = 32          # floor — keeps bands/text dark-gray, readable
    doc_gamma: float = 1.0              # neutral midtones (no extra crush/lift)
    # photo preset (gentle — protects colored cards/photos)
    photo_white_pct: float = 98.0
    photo_black_pct: float = 2.0
    photo_black_target: int = 0
    photo_gamma: float = 1.0
    # chroma denoise: median-blur + gentle gaussian on the LAB a/b channels to
    # suppress color moiré (photographing a screen) and speckle, while keeping large
    # ink/stamp regions. Real document color is large-area, so this is safe.
    chroma_median: int = 7            # odd; 0 disables
    chroma_blur_sigma: float = 1.5    # extra gaussian on a/b for large moiré bands; 0 disables


@dataclass(frozen=True)
class DenoiseConfig:
    """FR-22 — edge-preserving bilateral default (preserve thin strokes)."""
    bilateral_d: int = 5
    bilateral_sigma_color: int = 50
    bilateral_sigma_space: int = 50
    # gentle luminance-only unsharp finish (crisper text; no color/moiré boost).
    # Moderate so text is sharp without halos that hurt readability on dense forms.
    sharpen_amount: float = 0.45      # 0 disables
    sharpen_sigma: float = 1.1


@dataclass(frozen=True)
class SanityConfig:
    """FR-30 — post-warp sanity. Flag implausible output instead of shipping it."""
    min_aspect: float = 0.4           # H/W or W/H must be within [min, max]
    max_aspect: float = 3.5
    max_uniform_frac: float = 0.98    # if ~all pixels are one value -> blank/black warp


@dataclass(frozen=True)
class PdfConfig:
    """FR-26..FR-27 — honest tradeoff: JPEG default (small, lossy), PNG optional."""
    default_encoding: str = "jpeg"    # 'jpeg' | 'png'
    jpeg_quality: int = 90
    default_dpi: int = 200            # explicit DPI so downstream sees correct size


@dataclass(frozen=True)
class Settings:
    upload: UploadLimits = field(default_factory=UploadLimits)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    docaligner: DocAlignerConfig = field(default_factory=DocAlignerConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    sanity: SanityConfig = field(default_factory=SanityConfig)
    pdf: PdfConfig = field(default_factory=PdfConfig)

    # FR-9: gate policy. accept_and_flag (default, protect UX) | reject_and_retake.
    gate_policy: str = os.environ.get("SCANNER_GATE_POLICY", "accept_and_flag")
    # Confidence below this surfaces the manual "Adjust" affordance / review flag.
    low_confidence_threshold: float = 0.6


settings = Settings()
