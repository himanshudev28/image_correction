"""Upload hardening (FR-4). Validate BEFORE we hand bytes to any decoder.

Guards: allow-listed MIME, max file size, and a decompression-bomb pixel cap
(checked via Pillow without fully decoding). Decoding itself happens in the
ingest worker, off the event loop.
"""
from __future__ import annotations

import io

from PIL import Image

from app.config import settings

# Make Pillow refuse absurd pixel counts instead of allocating gigabytes.
Image.MAX_IMAGE_PIXELS = settings.upload.max_pixels


class UploadRejected(Exception):
    """Raised when an upload fails hardening; surfaced as HTTP 400."""


def validate_upload(filename: str, content_type: str | None, data: bytes) -> str:
    """Validate an upload and return a normalized kind: 'image' | 'pdf'."""
    limits = settings.upload

    if len(data) == 0:
        raise UploadRejected("empty file")
    if len(data) > limits.max_bytes:
        raise UploadRejected(
            f"file too large ({len(data)} bytes > {limits.max_bytes})"
        )

    kind = _sniff_kind(filename, content_type, data)
    if kind is None:
        raise UploadRejected(f"unsupported file type: {content_type or filename}")

    if kind == "pdf":
        _guard_pdf(data)
    else:
        _guard_image_pixels(data)
    return kind


def _sniff_kind(filename: str, content_type: str | None, data: bytes) -> str | None:
    # Trust magic bytes over the client-declared content type.
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":            # JPEG
        return "image"
    if data[:8] == b"\x89PNG\r\n\x1a\n":       # PNG
        return "image"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypheim", b"ftyphevc"):
        return "image"                          # HEIC/HEIF (transcoded on ingest)
    # fall back to declared type if magic bytes were inconclusive
    if content_type in settings.upload.allowed_mime:
        return "pdf" if content_type == "application/pdf" else "image"
    return None


def _guard_image_pixels(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Image.DecompressionBombError as exc:
        raise UploadRejected("image exceeds pixel limit") from exc
    except Exception as exc:  # noqa: BLE001 — any decode failure is a rejection
        raise UploadRejected("unreadable image") from exc
    if w * h > settings.upload.max_pixels:
        raise UploadRejected("image exceeds pixel limit")


def _guard_pdf(data: bytes) -> None:
    # Cheap structural guard; full page-count check happens in ingest with pypdfium2.
    if b"%%EOF" not in data[-2048:] and b"%%EOF" not in data:
        raise UploadRejected("malformed PDF")
