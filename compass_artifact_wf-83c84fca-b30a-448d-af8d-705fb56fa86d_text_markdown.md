# Server-Side Document Scanner & Image Cleanup Pipeline for Healthcare Consent Forms: Techniques Survey + Buildable Plan

## TL;DR
- **You can replicate the core of Adobe Scan / Microsoft Lens / CamScanner server-side with a classical OpenCV pipeline that needs no GPU and is commercially clean**: EXIF-normalize → quality gate (variance-of-Laplacian blur, overexposure, resolution) → largest-quadrilateral boundary detection (Canny + contour + `approxPolyDP`) → 4-point perspective warp (`getPerspectiveTransform`/`warpPerspective`) → illumination flattening (divide by morphologically-estimated background) → edge-preserving denoise → optional adaptive-threshold "scan look" → PDF via img2pdf. This is the canonical recipe and runs in well under a second per page on CPU.
- **License is the dominant constraint for a commercial healthcare product.** OpenCV (Apache-2.0 from v4.5.0+; BSD-3 on 4.4.0 and earlier), NumPy/scikit-image/Pillow (BSD/MIT-style), and img2pdf (LGPL-3.0) are all commercially safe. The high-accuracy dewarping models everyone cites — **DocTr, DocTr++, DocGeoNet, GeoTr — are research-only**; the official DocTr repo (fh2019ustc/DocTr) states verbatim: *"For commercial usage, please contact Professor Wengang Zhou (zhwg@ustc.edu.cn) and Hao Feng (haof@mail.ustc.edu.cn)"* — and so they must NOT ship in the product. The **HED** edge model's code is BSD-2-Clause but its pretrained weights were trained on Berkeley's **BSDS500 (non-commercial)** data, so the shipped weights are also tainted. The one DL upgrade that is commercially clean is **DocAligner (Apache-2.0)** for corner detection.
- **Recommendation: ship the classical CV baseline now** (fast, GPU-free, fully self-hosted, no PHI leaves the box), add **DocAligner (Apache-2.0)** as an optional corner-detection upgrade for forms that fill the frame or sit on low-contrast backgrounds, and treat full geometric dewarping as a "hard-case only" path you either skip (consent forms are usually flat) or build by retraining an open architecture on your own/synthetic data rather than shipping research weights. Avoid PyMuPDF (AGPL) for PDF assembly unless you buy the commercial license.

## Key Findings

1. **All the consumer scanner apps run the same conceptual pipeline**: detect document boundary → correct perspective → flatten lighting/remove shadow → enhance/threshold → export PDF. Classic apps did this with OpenCV-style heuristics; modern apps (CamScanner "Magic"/"Omnifix", Microsoft Lens, Adobe Scan) have moved boundary detection and enhancement to deep-learning models, but the *stages* are unchanged. The auto-capture logic (stability/glare/boundary-in-frame detection) is **on-device camera UX and is irrelevant server-side** — its server-side analogue is a **quality gate** that rejects or flags bad uploads.

2. **Boundary detection has two tiers.** Classical: grayscale → Gaussian blur → Canny → `findContours` → keep largest contour whose `approxPolyDP` yields 4 points. Fast, no training, but fails when the document has no visible margin (fills the frame), or against a cluttered/low-contrast background. Deep tier: semantic segmentation (U-Net/DeepLab), corner heatmap regression (stacked-hourglass), 

3. **Perspective correction ≠ dewarping.** A flat sheet photographed at an angle needs only a **homography / 4-point transform** (cheap, exact). A *curved, folded, or rolled* page needs full **geometric dewarping** (DewarpNet, DocUNet, DocTr/GeoTr, DocTr++, Marior) which predicts a per-pixel warp field. For signed consent forms — typically single flat sheets on a desk — **4-point perspective correction is almost always sufficient**; dewarping is a niche hard-case path, and the best models for it are research-licensed.

4. **Illumination/shadow removal is where "scan look" is won.** The classical, commercially-clean technique that mimics CamScanner's white background: estimate the background illumination by a large morphological close (or large-kernel blur/median), then **divide the original by the background** and rescale — this flattens shadows and uneven lighting. Deep options (BEDSR-Net, UDoc-GAN, DocTr's IllTr) exist but are heavier and mostly research-grade.

5. **For forms with signatures and stamps, do NOT hard-binarize by default.** Global/aggressive binarization can drop light pen strokes, blue ink, and stamp color. Prefer keeping a color or grayscale "flattened" output and reserve adaptive thresholding (Sauvola is best for documents) for an optional black-and-white mode, with signatures preserved by working on the illumination-corrected image.

## Details

### PART 1 — How the scanner apps do it (stage by stage)

**Stage 0 — Auto-capture (on-device only).** Apps decide *when* to shoot using: frame-to-frame **stability detection**, **boundary-in-frame** detection (the quadrilateral is fully visible), and **glare/blur** detection. None of this applies to a server that receives an already-taken photo. The server-side equivalent is a **quality gate** (Stage 1 below). This matches your DocumentAI V4 quality-assessment stage (sharpness, contrast, brightness, noise, completeness, skew).

**Stage 1 — Page/document boundary detection.**
- *Classical OpenCV recipe (the dominant approach):* convert to grayscale → Gaussian blur → **Canny** edge detection → `cv2.findContours` → sort contours by area → for each, `cv2.approxPolyDP(c, 0.02*perimeter, True)`; the first contour that approximates to **4 points** is taken as the page. A common robustness trick is adding a small (~5px) border so a document touching the image edge still yields a closed contour. Variants use `GrabCut` foreground segmentation or `morphologyEx` closing to suppress text/background before edge detection. Known failure modes (documented by LearnOpenCV and others): cluttered/low-contrast backgrounds produce spurious edges, and if a corner is outside the frame the quadrilateral assumption breaks.
- *Modern deep approaches:* (a) **Semantic segmentation** (U-Net / DeepLab-style) producing a document mask, then contour-fit; (b) **corner-point regression / heatmap regression** — a CNN (often a stacked-hourglass, as in Dropbox's and others' patents) predicts the 4 corners as heatmaps, robust to "in-the-wild" lighting/perspective/background; (c) **HED (Holistically-Nested Edge Detection)**, a VGG-based fully-convolutional net that yields cleaner object-boundary edge maps than Canny, available via OpenCV's `dnn` module. Relevant datasets/benchmarks: **SmartDoc 2015** (boundary/corner detection), **MIDV** (ID documents), **DocUNet / DIR300** (dewarping benchmarks).

**Stage 2 — Perspective correction & dewarping.**
- *Flat perspective correction:* order the 4 corners (top-left, top-right, bottom-right, bottom-left), compute target rectangle dimensions, then `cv2.getPerspectiveTransform` + `cv2.warpPerspective`. This is a homography and is exact for flat pages. The canonical `order_points` + `four_point_transform` is the PyImageSearch recipe (also in `imutils.perspective`).
- *Full geometric dewarping (curved/folded pages):* predicts a dense 2D displacement/forward-mapping field. Key models: **DocUNet** (stacked U-Net forward mapping), **DewarpNet** (stacked 3D+2D regression, ~86.9M params), **DocTr / GeoTr** (transformer geometric unwarping; ~26.9M params; SOTA on the DocUNet benchmark with reported LD≈8.38, with the IllTr companion for illumination), **DocTr++** (unrestricted/in-the-wild), **DocGeoNet** (~24.8M params), **Marior**, and recent hybrids (YOLOv8 mask + cubic-polynomial grid). **Difference:** perspective correction removes only the projective distortion of a *plane*; dewarping removes *non-planar* deformation (page curl, folds). Use dewarping only when the page is physically non-flat.

**Stage 3 — Illumination correction / shadow removal.**
- *Classical (recommended, clean):* estimate background illumination via large-kernel **morphological closing** (or large median/Gaussian blur), then **divide** the grayscale image by this background estimate and normalize — this is the standard "flatten shadows then threshold" pipeline. Variants: homomorphic filtering, multi-scale **Retinex** on the L channel (LAB), YUV/Y-channel histogram equalization, white-balancing. CamScanner's classic "Magic Color" is reproduced by **black-point/white-point selection** (push near-black to black, near-white to white) — i.e. contrast-stretch + level clipping.
- *Deep:* **BEDSR-Net** — by Yun-Hsuan Lin, Wen-Chin Chen, and Yung-Yu Chuang (National Taiwan University), CVPR 2020, pp. 12905–12914 — is described in-paper as *"the first deep network specifically designed for document image shadow removal"*; it pairs a background-estimation sub-net (BENet) with a shadow-removal sub-net (SRNet). Also **UDoc-GAN** (unpaired) and DocTr's **IllTr**. CamScanner's "Magic Pro"/"Omnifix" filters are described by the vendor as proprietary deep-learning that detects lighting/shadows/angles and removes blur, shadows, and moiré.

**Stage 4 — Noise reduction.** Options and trade-offs for documents: **median filter** (good for salt-and-pepper, can thin strokes), **bilateral filter** (`cv2.bilateralFilter`, edge-preserving — preserves text/signature edges), **Non-Local Means** (`cv2.fastNlMeansDenoising`, best quality, slower, preserves texture), morphological open/close for speck removal. For documents, the key constraint is **preserving thin strokes and signature lines** — bilateral or mild NLM are preferred over aggressive Gaussian blur.

**Stage 5 — Binarization / "scan look".** **Otsu** (global, fast, fails under uneven light), **adaptive threshold** (`cv2.adaptiveThreshold`, Gaussian/mean), **Sauvola** and **Niblack** (local, window-based; Sauvola is the document standard, `skimage.filters.threshold_sauvola`), **CLAHE** for contrast. The "black & white document" mode = illumination-flatten then Sauvola/adaptive threshold. **Signature/stamp preservation:** binarization can destroy light or colored ink — keep a color/gray flattened variant as the default for consent forms and treat B&W as opt-in.

**Stage 6 — Auto-capture → server-side quality gating.** Replace on-device capture logic with: **blur** via variance of the Laplacian (`cv2.Laplacian(gray, cv2.CV_64F).var()` below a threshold = blurry; threshold is application-specific and must be calibrated to your image resolution); **glare/overexposure** via fraction of saturated (near-255) pixels and histogram analysis; **resolution/DPI** minimums; and **boundary-found** check (reject if no 4-point contour and no DL fallback succeeds). These map directly onto DocumentAI V4's existing sharpness/contrast/brightness/noise/completeness/skew metrics.

### PART 2 — Buildable server-side plan

**Recommended libraries and licenses (commercial-safety flagged):**
- **OpenCV (`opencv-python-headless`)** — **Apache-2.0 for v4.5.0 and higher; 3-clause BSD for v4.4.0 and earlier** (per OpenCV's official license page; the change took effect ~August 2020 with the 4.5 series) — **commercially safe either way**. Use the `headless` wheel on servers (no GUI deps). pip-installable; bundles native libs.
- **NumPy** — BSD-3 — safe. Pure dependency.
- **scikit-image** — BSD-3 — safe (Sauvola/Niblack, restoration). pip-installable.
- **Pillow** — MIT-CMU/HPND (permissive) — safe. EXIF handling via `ImageOps.exif_transpose`.
- **img2pdf** — **LGPL-3.0** (PyPI classifier "GNU Lesser General Public License v3 (LGPLv3)"; latest release 0.6.3, Nov 2025) — safe for commercial use as an unmodified library dependency (weak copyleft). Lossless image→PDF: it embeds JPEG/PNG/JPEG2000/CCITT-G4 data without re-encoding, so resolution/color are preserved and the PDF adds only ~500–700 bytes of container overhead.
- **Tesseract / OCRmyPDF** — Tesseract Apache-2.0; OCRmyPDF MPL-2.0 — safe for an optional OCR text layer. Tesseract needs a system package. OCRmyPDF itself recommends img2pdf for image→PDF (it warns against ImageMagick/Ghostscript, which transcode/downsample).
- **DocAligner (DocsaidLab)** — **Apache-2.0, commercially safe** (confirmed in its repo LICENSE) — optional DL corner detector using heatmap regression; small model that runs on CPU (PyTorch/ONNX, input 128×128 or 256×256 RGB, mobile backbones like PP-LCNet).
- **AVOID for commercial product:** **PyMuPDF/fitz (AGPL-3.0)** — would force AGPL obligations unless you buy Artifex's commercial license; **DocTr/DocTr++/DocGeoNet/GeoTr** — research-only (DocTr README: *"For commercial usage, please contact Professor Wengang Zhou (zhwg@ustc.edu.cn) and Hao Feng (haof@mail.ustc.edu.cn)"*; DocTr-Plus: *"For commercial usage, please contact Hao Feng (haof@mail.ustc.edu.cn)"*); **HED `hed_pretrained_bsds.caffemodel`** — code is BSD-2-Clause but weights trained on **BSDS500**, whose Berkeley terms restrict use to *"non-commercial research and educational purposes"*, so don't ship the weights — retrain on a commercially-licensed dataset if you want HED.

**Recommended staged pipeline (primary classical + optional DL upgrade):**

| Stage | Primary (classical, GPU-free, clean) | Optional DL upgrade (license) |
|---|---|---|
| Ingest | Pillow `exif_transpose`, decode, downscale to ~1500–2000px long edge for processing (keep full-res original) | — |
| Quality gate | Laplacian variance (blur), saturated-pixel ratio (glare), min-resolution, boundary-found | — |
| Boundary detect | Canny + `findContours` + `approxPolyDP` largest quad | **DocAligner (Apache-2.0)** corner heatmaps; or U-Net mask (train your own) |
| Perspective | `getPerspectiveTransform` + `warpPerspective` on full-res | Full dewarping only for curved pages — retrain open arch; do NOT ship DocTr weights |
| Illumination | Background estimate (morph close / large blur) → divide → normalize; optional CLAHE/white-point | BEDSR-Net-style (research-grade; build/retrain if needed) |
| Denoise | Bilateral filter (default) or mild NLM | learned denoiser (optional) |
| Binarize (optional) | Sauvola adaptive threshold for B&W mode; keep color/gray as default | — |
| PDF | img2pdf (lossless) | OCRmyPDF for text layer |

**Classical-only baseline vs. hybrid:**
- **Classical-only baseline:** fully CPU, no GPU, no model downloads, fully self-contained, fully commercial-clean. Per-page latency is dominated by the warp and denoise on a downscaled image; the boundary/perspective/illumination steps are each on the order of tens of milliseconds on a downscaled image, comfortably **sub-second per page on a single CPU core** (variance-of-Laplacian and contour work are very cheap; NLM denoise is the slowest piece and can be swapped for bilateral). This is the right default for a PoC and for HIPAA, because **no image ever leaves the server** and there are no third-party API calls.
- **Hybrid (add DocAligner for hard cases):** add the Apache-2.0 corner model only when the classical contour finder fails (no 4-point quad), e.g. forms that fill the frame or low-contrast backgrounds. DocAligner is a small model that runs on CPU; a GPU is *not required* but would cut per-image inference latency.
- **GPU need:** none for the classical baseline or for occasional DocAligner CPU inference. A GPU only becomes worthwhile if you adopt heavy per-pixel dewarping/shadow nets at high throughput — which the licensing analysis says to avoid anyway.

**PDF conversion considerations:**
- Use **img2pdf** to assemble corrected page images losslessly (it embeds JPEG/PNG without transcoding, so resolution and color are preserved and file size stays near the source). Multi-page consent packets = pass a list of page images. Set DPI explicitly so downstream tools see correct physical size.
- For an optional searchable text layer, run **OCRmyPDF** (wraps Tesseract) — it can deskew/clean and produce PDF/A.
- **Do not use PyMuPDF** for assembly in the shipped commercial product due to AGPL, unless the commercial license is purchased; img2pdf + Pillow + (optionally) ReportLab cover the need with permissive/LGPL licensing.

**Reference pipeline sketch (adaptable to FastAPI):**

```python
import cv2, numpy as np, img2pdf
from PIL import Image, ImageOps

def load_normalized(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)          # fix mobile orientation
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def quality_gate(bgr, blur_thresh=100.0, glare_frac=0.05, min_long_edge=1000):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()          # blur
    glare = float((gray > 250).mean())                  # overexposure
    long_edge = max(bgr.shape[:2])
    ok = fm >= blur_thresh and glare <= glare_frac and long_edge >= min_long_edge
    return ok, {"laplacian_var": fm, "glare_frac": glare, "long_edge": long_edge}

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def find_document(bgr):
    ratio = 1000.0 / max(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=ratio, fy=ratio)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
    cnts,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return approx.reshape(4,2) / ratio          # back to full res
    return None   # -> fall back to DocAligner, or accept whole frame

def warp(bgr, quad):
    rect = order_points(quad)
    (tl,tr,br,bl) = rect
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W,H))

def flatten_illumination(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    norm = cv2.divide(gray, bg, scale=255)              # shadow/light flatten
    return norm

def denoise(gray):
    return cv2.bilateralFilter(gray, 5, 50, 50)         # preserve strokes

# FastAPI handler: load -> gate -> find (quad or DocAligner) -> warp
# -> flatten -> denoise -> (optional Sauvola) -> img2pdf.convert([...])
```

**Practical gotchas (consent-form specific):**
- **Already-flat photos:** if the boundary is near the full frame and skew is tiny, skip the warp (avoid resampling artifacts). Gate on detected skew angle.
- **EXIF orientation:** always run `ImageOps.exif_transpose` before any CV; iPhone/Android encode rotation in EXIF (orientation tag 6/3/8). Note that **img2pdf raises an error on an invalid EXIF Orientation value of 0** — per its docs, *"Android phones and Canon DSLR cameras produce JPEG images with the invalid value of zero"* — so sanitize/strip the orientation tag after transpose.
- **Very high-res phone photos:** detect on a downscaled copy (~1000px long edge) for speed, but **apply the final warp/output at full resolution** to preserve handwriting fidelity.
- **Forms that fill the whole frame (no margin/background):** the largest-quad heuristic fails — there's no background to contour against. Fallbacks: accept the full frame as-is (deskew only), or use **DocAligner** corner regression which doesn't require a visible margin.
- **Color vs. grayscale output:** default to **color or flattened grayscale** for consent forms so blue-ink signatures and colored stamps survive; offer B&W (Sauvola) as an option, never as the only output.
- **Signature/handwriting fidelity:** prefer bilateral/NLM over Gaussian blur; avoid global binarization; keep the illumination-corrected color image as the archival page.

## Recommendations

**Stage 1 (now, PoC):** Build the **classical OpenCV pipeline** exactly as sketched: EXIF-normalize → quality gate (Laplacian variance + glare + resolution) → Canny/contour boundary → 4-point warp → morphological-divide illumination flatten → bilateral denoise → img2pdf. Output **color/grayscale flattened PDF**, not binarized. This is GPU-free, sub-second/page, commercially clean, and keeps all PHI on your server — ideal for the FastAPI + SQLite synthetic-data PoC, and it slots in *before/as part of* DocumentAI V4's quality-assessment stage. Reuse V4's existing OpenCV sharpness/contrast/brightness/noise/skew metrics to drive the gate thresholds.

**Stage 2 (upgrade path, when classical boundary detection misses):** Add **DocAligner (Apache-2.0)** as a CPU fallback corner detector triggered only when no 4-point quad is found or detected skew is implausible. Keep it behind a feature flag and measure the failure rate before committing.

**Stage 3 (only if real curved/folded forms appear):** Add full dewarping — but **do not ship DocTr/DocTr++/DocGeoNet/GeoTr/DewarpNet pretrained weights** in the commercial product (research-only). Either retrain an open architecture on synthetic/owned data, or license a commercial SDK. For consent forms this is likely unnecessary.

**Decision thresholds that should change the plan:**
- If >~10–15% of uploads fail the classical 4-point detector → invest in DocAligner.
- If a meaningful fraction of forms are physically curved/folded (not just skewed) → evaluate dewarping (with the licensing caveat).
- If throughput requirements exceed what one CPU core delivers per page → scale horizontally (stateless workers) before reaching for a GPU; the classical pipeline parallelizes trivially.
- If a searchable archive is required → add OCRmyPDF (Tesseract) as a final, optional stage.

**Licensing rule of thumb to enforce in code review:** every model/library that touches a PHI image must be MIT/BSD/Apache (or LGPL for libraries). Flag AGPL (PyMuPDF) and any "contact us for commercial use" model (DocTr family) and any model whose *training data* is non-commercial (HED/BSDS500) as blockers.

## Caveats
- **Latency figures are engineering estimates**, not benchmarked on your hardware/images. Variance-of-Laplacian, contour detection, and perspective warp are individually cheap (tens of ms on downscaled images); Non-Local Means denoise is the main variable cost. Benchmark on representative consent-form photos before committing to SLAs.
- **Blur thresholds are not universal.** The variance-of-Laplacian cutoff must be calibrated to your capture resolution and downscaling; published examples range widely (e.g. ~50–1000). Tune on a labeled sample.
- **BEDSR-Net quantitative metrics not independently verified.** The paper reports strong PSNR/SSIM on document shadow removal, but specific figures should be read directly from the paper's results tables before being cited; treat BEDSR-Net as research-grade for the deep-shadow-removal path.
- **License statements should be verified by counsel before shipping.** DocTr/DocTr++/DocGeoNet repos explicitly direct commercial users to contact the authors; PyMuPDF is dual AGPL/commercial; HED weights derive from BSDS500 whose Berkeley terms are "non-commercial research and educational purposes." OpenCV's license changed across versions (BSD-3 through 4.4.0, Apache-2.0 from 4.5.0). DocAligner is Apache-2.0 per its LICENSE file. Confirm current LICENSE files at integration time.
- **CamScanner/Adobe/Microsoft internal algorithms are proprietary**; vendor descriptions ("Magic Pro", "Omnifix", deep-learning shadow/blur removal) are marketing-level, not implementation detail. The open-source equivalents described here reproduce the *results*, not the exact algorithms.
- **BSDS500 commercial status is not fully resolved** — Berkeley's primary terms say non-commercial, and the BIDS GitHub mirror carries no explicit license; treat HED-BSDS weights as non-commercial unless legal review establishes otherwise.