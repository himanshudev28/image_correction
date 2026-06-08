# Adaptive Pipeline Upgrade — "handles all images"

**What this is.** A per-image adaptive layer so processing strength is decided by *measuring each page*, not by one static config. This is the only thing that generalizes across the full variety of inputs. It plugs into your existing `image_correction` pipeline; most edge cases you already handle — this adds the measured denoise/sharpen, automatic screen-photo routing, faint-ink protection, and an explicit tail → manual path.

**Honest scope.** There is no static config that is literally perfect on every possible image. This maximizes *automatic* coverage and routes the genuine tail to your manual **Adjust** UI + low-confidence review queue. Thresholds below are anchored to your real sample (input noise σ≈1.0) but **must be recalibrated on your own corpus** — see §5.

---

## 1. New module: `backend/app/pipeline/adaptive.py`

```python
"""Per-image adaptive parameter selection — the core of 'handles all images'.

Instead of fixed denoise/sharpen strengths, MEASURE the warped page and choose
processing to match it. A clean shot is left sharp; a grainy one is denoised;
a screen photo is routed to de-moiré. Thresholds are starting points — calibrate.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast Gaussian-noise estimate. ~1 on a clean capture, high on a
    grainy low-light photo. Robust enough on mostly-flat document pages."""
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float64), -1, M, borderType=cv2.BORDER_REPLICATE)
    return float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def moire_score(bgr: np.ndarray, grid: int = 8) -> float:
    """Screen/moire proxy: per-TILE max of low-saturation LAB-chroma. Moire is
    usually localized, so a per-tile MAX catches it where a global mean misses it
    (your screen sample read only ~1.8 as a global mean despite visible moire)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128
    b = lab[:, :, 2].astype(np.float32) - 128
    chroma = np.sqrt(a * a + b * b)
    lowsat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1] < 40
    h, w = chroma.shape
    th, tw = max(1, h // grid), max(1, w // grid)
    best = 0.0
    for y in range(0, h - th + 1, th):
        for x in range(0, w - tw + 1, tw):
            m = lowsat[y:y + th, x:x + tw]
            if m.mean() > 0.3:                       # a paper/background tile
                best = max(best, float(chroma[y:y + th, x:x + tw][m].mean()))
    return best


def select_params(bgr: np.ndarray, cfg) -> dict:
    """Decide denoise + sharpen + routing for THIS page."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sigma = estimate_noise_sigma(gray)
    mscore = moire_score(bgr)

    # denoise strength from measured noise (clean -> none; grainy -> NLM)
    if sigma < cfg.noise_skip:
        denoise = ("none", 0)
    elif sigma < cfg.noise_light:
        denoise = ("bilateral", cfg.bilat_light_sigma_color)
    elif sigma < cfg.noise_heavy:
        denoise = ("bilateral", cfg.bilat_strong_sigma_color)
    else:
        denoise = ("nlm", 0)

    # sharpen: SMALL radius always (no halos); amount tracks how much we softened
    amount = {"none": 0.35, "bilateral": 0.45, "nlm": 0.60}[denoise[0]]
    route_demoire = mscore >= cfg.moire_threshold
    if route_demoire:
        amount = min(amount, 0.40)                   # screen photos: don't over-sharpen

    return {
        "noise_sigma": round(sigma, 2),
        "moire_score": round(mscore, 2),
        "denoise": denoise,                          # (mode, sigma_color)
        "sharpen": (amount, cfg.sharpen_sigma_small),  # (amount, radius)
        "route_demoire": route_demoire,
    }
```

## 2. Small signature changes to existing stages

`denoise.py` — let `denoise()` take a per-call `sigma_color` and a `"none"` mode, and let `sharpen()` take a per-call radius:

```python
def denoise(bgr, mode="bilateral", sigma_color=None):
    cfg = settings.denoise
    if mode == "none":
        return bgr
    if mode == "nlm":
        return cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
    sc = cfg.bilateral_sigma_color if sigma_color is None else sigma_color
    return cv2.bilateralFilter(bgr, cfg.bilateral_d, sc, cfg.bilateral_sigma_space)

def sharpen(bgr, amount=None, sigma=None):
    cfg = settings.denoise
    amt = cfg.sharpen_amount if amount is None else amount
    if not amt or amt <= 0:
        return bgr
    sig = cfg.sharpen_sigma if sigma is None else sigma          # small radius from adaptive
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=sig)
    L = cv2.addWeighted(L, 1.0 + amt, blur, -amt, 0)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
```

## 3. Runner wiring (`runner.py`, the stages after warp)

Replace the fixed illumination → denoise → sharpen block with the adaptive one:

```python
from app.pipeline import adaptive

# ... after `warped` is produced ...

ap = adaptive.select_params(warped, settings.adaptive)
if ap["noise_sigma"] >= settings.adaptive.noise_heavy:
    flags.append("noisy")
if ap["route_demoire"]:
    flags.append("screen_moire")

# auto-route screen photos through de-moire (no manual flag needed)
if ap["route_demoire"] and settings.demoire.enabled:
    warped = demoire_ml.demoire(warped)

scan_white = settings.illumination.demoire_white_pct if ap["route_demoire"] else None
flat = illumination.flatten(warped, scan_white_pct=scan_white)

mode_d, sc = ap["denoise"]
clean = denoise_stage.denoise(flat, mode=mode_d, sigma_color=sc)
amt, sig = ap["sharpen"]
clean = denoise_stage.sharpen(clean, amount=amt, sigma=sig)

final = output_stage.apply_mode(clean, mode=mode)
```

The clean/born-digital and frame-filling cases are already routed earlier by `nonphoto.looks_already_clean` and the DocAligner fallback — leave those; they short-circuit the camera pipeline before this block.

## 4. Config additions (`config.py`)

```python
@dataclass(frozen=True)
class AdaptiveConfig:
    # noise (Immerkaer sigma): your clean screen sample read ~1.0
    noise_skip: float = 2.0           # below -> skip denoise (keep it sharp)
    noise_light: float = 5.0          # light bilateral
    noise_heavy: float = 9.0          # above -> NLM (grainy low-light)
    bilat_light_sigma_color: int = 12
    bilat_strong_sigma_color: int = 30
    sharpen_sigma_small: float = 0.6  # small radius => crisp edge, no ring
    moire_threshold: float = 6.0      # per-tile chroma; CALIBRATE (global mean ~1.8 on the sample)

# in Settings:
adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)

# PdfConfig: kill mosquito noise around text
jpeg_quality: int = 95
```

And in `pdf.encode_page`, the JPEG branch:

```python
params = [cv2.IMWRITE_JPEG_QUALITY, int(q)]
if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):           # OpenCV >= 4.3
    params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
ok, buf = cv2.imencode(".jpg", bgr, params)
```

## 5. Edge-case coverage

| Input | How it's handled |
|---|---|
| Clean paper photo | Full pipeline; adaptive **skips denoise** (σ low); small-radius sharpen. |
| Noisy / low-light photo | Adaptive bilateral or NLM by measured σ, *then* sharpen. |
| Screen photo / moiré | Auto-routed to ESDNet de-moiré (per-tile moiré detector); sharpen capped to avoid ring. |
| Screenshot / born-digital PDF | `nonphoto` + born-digital passthrough — camera pipeline skipped, no re-encode. |
| Faint pencil / pale-blue signature | Color-preserving LAB flatten + conservative black floor; **B&W stays opt-in**. (For auto-protection, detect thin mid-tone *connected components* — not a global mid-gray fraction, which your sample showed is dominated by gray backgrounds, 0.53.) |
| Frame-filling form (no margin) | DocAligner corner model, else accept full frame + deskew. |
| Curled / folded page | 4-point cannot dewarp it; geometry-plausibility guard flags it → manual Adjust. True dewarping is out of scope (and research-licensed). |
| Rotated capture | Rotation-robust corner ordering. |
| Low resolution | Gate flag (`low_resolution`). |
| Over/under-exposed, glare | Per-tile glare gate + illumination flatten; extreme → flag. |
| Bad warp | Post-warp sanity (aspect/uniformity) → flag, not shipped. |
| **Genuine tail** | Manual **Adjust** UI + low-confidence **review queue** — the deliberate catch-all. |

## 6. How to actually finalize (the real "done")

1. **Build a representative corpus** — 15–30 of your *real* paper consent forms across good/bad lighting, with and without signatures, plus a few odd ones (a screen shot, a low-res capture, a curled page).
2. **Run the pipeline on all of them**, eyeball outputs, and check the three failure modes that bit you: **no halos/rings**, **signatures legible**, **manual-Adjust rate acceptably low**.
3. **Calibrate** `AdaptiveConfig` (especially `moire_threshold` and the `noise_*` cutoffs) and the gate thresholds against that corpus — the numbers here are anchors, not gospel.
4. **Ship** when the corpus clears your bar. The manual Adjust + review queue absorb the rest.

That is the closest thing to "fixed for all images": adaptive logic for the body of the distribution, a calibrated corpus to prove it, and a manual path for the tail — which is exactly the shape Adobe ships, too.
