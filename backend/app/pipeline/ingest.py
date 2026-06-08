"""Stage 1 — ingest & sanitize (FR-3, FR-4).

Decodes an uploaded image OR rasterizes an uploaded PDF into per-page BGR arrays.
Always:
  - applies EXIF orientation (`exif_transpose`) then STRIPS all metadata (PHI control),
  - sanitizes the invalid EXIF orientation value 0 (Android/Canon) before anything,
  - returns clean numpy BGR images with no embedded metadata.

Decoding happens here, in the worker (called via run_in_executor), not on the
event loop.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps

# register HEIC/HEIF opener so Image.open handles iPhone photos
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001 — HEIC support optional; JPEG/PNG still work
    pass

from app.config import settings


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    """Normalize a PIL image to a metadata-free BGR ndarray."""
    # Fix orientation. Guard against the invalid Orientation=0 some devices write,
    # which makes exif_transpose raise — strip it first.
    try:
        exif = im.getexif()
        if exif.get(0x0112) == 0:        # 0x0112 == Orientation
            del exif[0x0112]
            im.info.pop("exif", None)
    except Exception:  # noqa: BLE001
        pass
    im = ImageOps.exif_transpose(im)
    rgb = im.convert("RGB")              # convert() drops the metadata dict
    arr = np.asarray(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return _pil_to_bgr(im)


def load_pdf_pages(data: bytes) -> list[np.ndarray]:
    """Rasterize each PDF page to a BGR array via pypdfium2 (commercially clean)."""
    pdf = pdfium.PdfDocument(data)
    try:
        n = len(pdf)
        if n > settings.upload.max_pages_per_pdf:
            raise ValueError(
                f"PDF has {n} pages > limit {settings.upload.max_pages_per_pdf}"
            )
        scale = settings.ingest.pdf_render_dpi / 72.0  # PDF user space is 72 dpi
        pages: list[np.ndarray] = []
        for i in range(n):
            page = pdf[i]
            pil = page.render(scale=scale).to_pil()
            pages.append(_pil_to_bgr(pil))
        return pages
    finally:
        pdf.close()


def load_pages(kind: str, data: bytes) -> list[np.ndarray]:
    """Unified entry: returns a list of metadata-free BGR pages for image or PDF."""
    if kind == "pdf":
        return load_pdf_pages(data)
    return [load_image(data)]


# ---- born-digital passthrough ----------------------------------------------
#
# A clean/born-digital PDF page (vector text, not a scanned photo) should be kept
# verbatim — NOT rasterized and JPEG re-encoded, which would destroy crispness and
# add artifacts. We detect such pages by their extractable text and pass them
# through losslessly; only true photo/scan pages enter the camera pipeline.


def _extract_single_page_pdf(data: bytes, index: int) -> bytes:
    """Copy one page out to its own PDF, losslessly (pypdf, BSD)."""
    import io as _io

    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(_io.BytesIO(data))
    writer = PdfWriter()
    writer.add_page(reader.pages[index])
    buf = _io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def rasterize_pdf_bytes(pdf_bytes: bytes, long_edge: int | None = None) -> np.ndarray:
    """Render a (single-page) PDF to a BGR array — used for previews only."""
    target = long_edge or settings.ingest.preview_long_edge
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        page = pdf[0]
        w_pt, h_pt = page.get_size()
        scale = target / max(w_pt, h_pt)
        return _pil_to_bgr(page.render(scale=scale).to_pil())
    finally:
        pdf.close()


def load_inputs(kind: str, data: bytes) -> list[dict]:
    """Return a list of input items, one per page:
      {"type": "image", "bgr": ndarray}            -> run the camera pipeline
      {"type": "pdf_page", "pdf_bytes": bytes}     -> born-digital, pass through
    """
    if kind != "pdf":
        return [{"type": "image", "bgr": load_image(data)}]

    pdf = pdfium.PdfDocument(data)
    try:
        n = len(pdf)
        if n > settings.upload.max_pages_per_pdf:
            raise ValueError(
                f"PDF has {n} pages > limit {settings.upload.max_pages_per_pdf}"
            )
        scale = settings.ingest.pdf_render_dpi / 72.0
        items: list[dict] = []
        for i in range(n):
            page = pdf[i]
            text = page.get_textpage().get_text_range() or ""
            if len(text.strip()) >= settings.ingest.born_digital_min_chars:
                # born-digital: keep the original page bytes, no rasterize/re-encode
                items.append({"type": "pdf_page", "pdf_bytes": _extract_single_page_pdf(data, i)})
            else:
                # scanned/photo page: rasterize and run the camera pipeline
                items.append({"type": "image", "bgr": _pil_to_bgr(page.render(scale=scale).to_pil())})
        return items
    finally:
        pdf.close()


def downscale_for_detection(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (small_image, scale) where small = bgr * scale. Detection runs on the
    small copy for speed; warps apply to full-res (FR-16)."""
    long_edge = max(bgr.shape[:2])
    target = settings.ingest.detect_long_edge
    if long_edge <= target:
        return bgr, 1.0
    scale = target / long_edge
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, scale
