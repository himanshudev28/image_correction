# Image Correction — Document Scanner Webapp

An **Adobe Scan–class** document-cleanup web app. Upload a **photo or PDF** of a
document (built for healthcare consent forms) and the server **automatically**
detects the page, crops to its edges, corrects perspective/angle, flattens shadows
and lighting (preserving ink & stamp color), denoises, sharpens, and returns a
clean PDF — **entirely self-hosted, GPU-free, no data leaves the box.**

- **Auto, on upload** — crop → straighten → color-correct → clean → export. No manual steps in the common case.
- **Photos *and* PDFs** — images run the camera pipeline; born-digital PDF pages pass through verbatim (no rasterize/re-encode).
- **Robust corners** — an ML corner detector (DocAligner, ONNX/CPU) is primary, with a classical OpenCV fallback.
- **Commercially clean & private** — every component is MIT/BSD/Apache (or LGPL); no AGPL/research-only code; nothing is sent to third parties.
- **Manual override** — drag the four corners, rotate, or switch color/grayscale/B&W when needed.

> Built from the included *Server-Side Document Scanner Pipeline* research report
> and `consent-scanner-webapp-prd.md`. This is the **M1 MVP**: file upload, the
> processing pipeline, multi-page color PDF export, and an auto-first review UI.

---

## How it works

The moment a page arrives, the server runs the full pipeline:

| # | Stage | What happens |
|---|-------|--------------|
| 1 | **Ingest & sanitize** | Decode image *or* rasterize PDF (`pypdfium2`); fix EXIF orientation; **strip all metadata** (privacy). Born-digital PDF pages are kept verbatim. |
| 2 | **Quality gate** | Content-aware sharpness (Tenengrad on busy tiles) + local glare + resolution. Accept-and-flag — never silently blocks. |
| 3 | **Boundary** | **DocAligner** ML corner model (primary) → classical multi-strategy (auto-Canny + near-white segmentation, convex-hull quad, rotated-rect) as a validated fallback. |
| 4 | **Perspective** | Rotation-robust corner ordering, full-res 4-point warp, small safety margin, skip-if-flat, deskew fallback, plausibility guard against bad warps. |
| 5 | **Illumination** | Color-preserving flatten on the LAB **L** channel: low-res background estimate kills large shadows, white-point whitens paper, a/b chroma blur suppresses screen moiré. |
| 6 | **Denoise + sharpen** | Edge-preserving bilateral filter, then a gentle luminance-only unsharp mask for crisp text (no color/moiré boost). |
| 7 | **Output** | Color (default) / grayscale / B&W (Sauvola, opt-in). |
| 8 | **Post-warp sanity** | Implausible-aspect / blank-output check → flag for review. |
| 9 | **PDF** | `img2pdf` for image pages + lossless `pypdf` merge for born-digital pages; explicit DPI; metadata-free. |

Low-confidence pages surface a **manual Adjust** affordance (drag the 4 corners → re-crop) plus rotate and output-mode controls.

### Corner detection: ML-primary, classical-fallback

On real phone photos, classical edge detection often returns a confident-but-wrong
quad — so **DocAligner is the primary detector** (heatmap-regression corner model,
Apache-2.0). We run **only the ONNX model via `onnxruntime`** — *not* the full
`docaligner_docsaid`/`capybara` toolkit (which pulls a non-headless OpenCV, Flask,
matplotlib, poppler). The image is **padded before inference** so the model still
finds corners when the page fills the frame (its known blind spot). Classical
detection is used only when DocAligner returns nothing *and* its quad is
geometrically plausible.

The model isn't committed (it's ~79 MB) — it **auto-downloads on first server
start** (and lazily on first use), so cloning and running reproduces the demonstrated
quality with no manual step. Pre-fetch or air-gap it via
`python scripts/fetch_docaligner_model.py` / `SCANNER_DOCALIGNER_AUTODOWNLOAD=0`.

If the model can't be obtained (offline) or `SCANNER_DOCALIGNER=0`, the pipeline
stays fully classical — the fallback is graceful. ML inference adds ~170 ms only on
pages it runs.

---

## Quick start

Two terminals. The Vite dev server proxies `/api` → FastAPI, so it works on first run.

**Backend** (`:8000`)
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> The ~79 MB DocAligner model **auto-downloads on first start** (one-time, needs
> internet), so a fresh clone reproduces the demonstrated quality with no extra
> steps. For air-gapped installs, pre-fetch it with
> `python scripts/fetch_docaligner_model.py` and/or set
> `SCANNER_DOCALIGNER_AUTODOWNLOAD=0`. Without the model, the app still runs but
> corner detection drops to classical-only.

**Frontend** (`:5173`)
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**, drop a photo or PDF, review, and **Export PDF**.

> Requires Python 3.10+ and Node 18+.

---

## Project structure

```
backend/
  app/
    main.py            FastAPI app (CORS, static SPA, routes)
    config.py          all tunables/thresholds (no magic numbers)
    api/scans.py       REST endpoints
    service.py         orchestration + persistence
    pipeline/          one module per stage:
      ingest, nonphoto, gate, boundary, docaligner,
      perspective, illumination, denoise, output, sanity, pdf, runner
    models.py db.py storage.py security.py integration.py
  tests/               per-stage, end-to-end, real-photo regression
  scripts/             fetch_docaligner_model.py, gen_samples.py
  models/              DocAligner ONNX (gitignored; fetched)
frontend/              Vite + React + TS + Tailwind SPA
sample_data/           local test images (gitignored)
```

---

## Tech stack & licensing

Every library that touches an image is permissive or weak-copyleft.

| Area | Library | License |
|------|---------|---------|
| API | FastAPI / Uvicorn | MIT |
| CV core | OpenCV-headless | Apache-2.0 |
| Arrays / imaging | NumPy, scikit-image, Pillow, pillow-heif | BSD / MIT / LGPL |
| PDF in | pypdfium2 | Apache/BSD |
| PDF merge | pypdf | BSD |
| PDF out | img2pdf | LGPL |
| ML corners | onnxruntime + DocAligner weights | MIT / Apache-2.0 |
| Frontend | React, Tailwind, Vite | MIT |

**Deliberately avoided:** PyMuPDF (AGPL), DocTr/GeoTr/DocGeoNet (research-only),
HED/BSDS500 weights (non-commercial training data).

> Note: verify the DocAligner model's *training-data* license before any
> commercial deployment (the code/weights are Apache-2.0).

---

## Testing

```bash
cd backend && source .venv/bin/activate
pytest                          # per-stage + end-to-end + real-photo regression
python scripts/gen_samples.py   # synthetic inputs + cleaned outputs -> sample_data/
```

Real-photo regression tests (`test_real_photo_quality_regression`) assert that
known-good photos still crop, straighten, and clean correctly — they **skip** when
the images aren't present (they live in `sample_data/`, which is gitignored).

Latency: ~120 ms/page classical, ~140–580 ms/page with ML corners — well under the sub-second target.

---

## Privacy, scope & honest limitations

- **Privacy:** all processing is server-side; no third-party APIs/CDNs in the image path; metadata is stripped from every image and the output PDF.
- **PDF tradeoff:** processed pages are re-encoded — **JPEG** (small, lossy, default) or **PNG** (lossless, larger). "Lossless *and* small *and* processed" isn't achievable and isn't promised.
- **Limitations:** a photo *of a screen* keeps faint moiré (inherent); a borderless/edge-bleeding document may need the Adjust tool. These are input limits, not bugs.
- **PoC scope:** PHI controls (metadata stripping, no egress, upload hardening) are on from day one. Encryption-at-rest, a real audit-log service, retention jobs, auth/SSO, and the DocumentAI V4 integration are **stubbed** pending real interfaces.

## Roadmap

- **M1 (this repo):** upload, full pipeline, ML+classical corners, color/gray/B&W, multi-page PDF, born-digital passthrough, manual adjust. ✅
- **M2:** in-browser camera capture, page reorder UI, operator review queue.
- **M3:** OCR text layer (OCRmyPDF), horizontal scale-out.
- **M4:** geometric dewarping for curled pages (commercially-clean models only).
