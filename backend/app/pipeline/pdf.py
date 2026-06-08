"""Stage 10 — PDF assembly (FR-26..FR-28).

Honest tradeoff (FR-26): processed pages are re-encoded before embedding.
  - JPEG (default): small, lossy.
  - PNG (optional): lossless, larger.
"Lossless AND small AND processed" is not achievable and is not promised.

img2pdf embeds the encoded image without transcoding; we set DPI explicitly so
downstream tools see correct physical size (FR-27), and we emit NO image metadata.
"""
from __future__ import annotations

import io

import cv2
import img2pdf
import numpy as np

from app.config import settings


def encode_page(bgr: np.ndarray, encoding: str, quality: int | None = None) -> bytes:
    """Encode one BGR page to JPEG or PNG bytes (no metadata)."""
    if encoding == "png":
        ok, buf = cv2.imencode(".png", bgr)
    else:
        q = quality if quality is not None else settings.pdf.jpeg_quality
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        raise RuntimeError(f"failed to encode page as {encoding}")
    return buf.tobytes()


def assemble(page_images: list[np.ndarray], encoding: str | None = None,
             dpi: int | None = None) -> bytes:
    """Assemble cleaned pages into a single PDF (multi-page, explicit DPI)."""
    encoding = encoding or settings.pdf.default_encoding
    dpi = dpi or settings.pdf.default_dpi

    encoded = [encode_page(p, encoding) for p in page_images]
    layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
    return img2pdf.convert(encoded, layout_fun=layout)


def image_to_pdf(bgr: np.ndarray, encoding: str | None = None,
                 dpi: int | None = None) -> bytes:
    """Single processed image -> single-page PDF bytes."""
    return assemble([bgr], encoding=encoding, dpi=dpi)


def merge_pdfs(pdfs: list[bytes]) -> bytes:
    """Concatenate PDFs losslessly in order (pypdf, BSD). Used to combine
    born-digital passthrough pages with processed image pages without re-encoding
    the born-digital ones."""
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for data in pdfs:
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
