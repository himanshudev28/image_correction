"""Service layer: orchestrates ingest -> pipeline -> persistence.

Keeps the API thin. All CPU-bound work (decode + OpenCV pipeline) runs here and is
dispatched off the event loop by the API via run_in_executor.
"""
from __future__ import annotations

import cv2
import numpy as np

from app import storage
from app.db import get_session
from app.integration import audit, handoff_to_v4
from app.models import Page, Pdf, ScanSession
from app.pipeline import pdf as pdf_stage
from app.pipeline import ingest
from app.pipeline.ingest import load_inputs
from app.pipeline.runner import process_page
from app.security import validate_upload


class NotFound(Exception):
    pass


def _store_processed(page_id: str, bgr: np.ndarray) -> str:
    png = pdf_stage.encode_page(bgr, "png")
    return storage.put("processed", f"{page_id}.png", png)


def _store_original(page_id: str, bgr: np.ndarray) -> str:
    # store a sanitized (metadata-free) raster of the original for re-crop/re-mode
    png = pdf_stage.encode_page(bgr, "png")
    return storage.put("originals", f"{page_id}.png", png)


def _load(ref: str) -> np.ndarray:
    data = np.frombuffer(storage.get(ref), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---- create + process ----

def create_scan(uploads: list[tuple[str, str | None, bytes]], owner: str | None = None,
                consent_ref: str | None = None) -> str:
    """uploads: list of (filename, content_type, data). Returns scan_id.
    Processes every page (image pages or rasterized PDF pages) synchronously."""
    with get_session() as s:
        session = ScanSession(owner=owner, consent_ref=consent_ref, status="processing")
        s.add(session)
        s.commit()
        s.refresh(session)
        sid = session.id

    order = 0
    for filename, content_type, data in uploads:
        kind = validate_upload(filename, content_type, data)   # FR-4 hardening
        items = load_inputs(kind, data)                         # FR-3,5 sanitize/classify
        for item in items:
            if item["type"] == "pdf_page":
                _persist_passthrough_page(sid, order, item["pdf_bytes"])
            else:
                _persist_new_page(sid, order, item["bgr"])
            order += 1

    with get_session() as s:
        session = s.get(ScanSession, sid)
        session.status = "ready"
        s.add(session)
        s.commit()

    audit(owner, "scan.create", sid, {"pages": order})
    return sid


def _persist_new_page(session_id: str, order: int, bgr: np.ndarray) -> Page:
    page = Page(session_id=session_id, order=order)
    page.original_ref = _store_original(page.id, bgr)
    result = process_page(bgr, mode="color")          # demoire=None → auto-decide
    page.processed_ref = _store_processed(page.id, result.image)
    page.confidence = result.confidence
    page.gate_flags = result.flags
    page.quad = result.quad
    page.demoire = result.demoire                     # reflect the auto-routing decision
    page.transforms = [{"op": "auto", "timings_ms": result.timings_ms}]
    with get_session() as s:
        s.add(page)
        s.commit()
        s.refresh(page)
    return page


def _persist_passthrough_page(session_id: str, order: int, pdf_bytes: bytes) -> Page:
    """Born-digital PDF page: keep verbatim, only rasterize a preview."""
    page = Page(session_id=session_id, order=order, passthrough=True)
    page.pdf_ref = storage.put("processed", f"{page.id}.pdf", pdf_bytes)
    preview = ingest.rasterize_pdf_bytes(pdf_bytes)
    page.processed_ref = _store_processed(page.id, preview)
    page.confidence = 1.0
    page.gate_flags = ["born_digital"]
    page.transforms = [{"op": "passthrough"}]
    with get_session() as s:
        s.add(page)
        s.commit()
        s.refresh(page)
    return page


# ---- reads ----

def get_scan(scan_id: str) -> dict:
    with get_session() as s:
        session = s.get(ScanSession, scan_id)
        if not session:
            raise NotFound(scan_id)
        pages = _ordered_pages(s, scan_id)
        return {
            "scan_id": session.id,
            "status": session.status,
            "pages": [_page_dto(p) for p in pages],
        }


def _ordered_pages(s, scan_id: str) -> list[Page]:
    from sqlmodel import select

    rows = s.exec(
        select(Page).where(Page.session_id == scan_id, Page.deleted == False)  # noqa: E712
    ).all()
    return sorted(rows, key=lambda p: p.order)


def _page_dto(p: Page) -> dict:
    return {
        "page_id": p.id,
        "order": p.order,
        "mode": p.mode,
        "demoire": p.demoire,
        "rotation": p.rotation,
        "confidence": p.confidence,
        "gate_flags": p.gate_flags,
        "quad": p.quad,
        "passthrough": p.passthrough,   # born-digital page kept verbatim (no edits)
        "low_confidence": p.confidence < _low_threshold(),
        "preview_url": f"/api/scans/{p.session_id}/pages/{p.id}/preview",
    }


def _low_threshold() -> float:
    from app.config import settings

    return settings.low_confidence_threshold


def get_page_image(scan_id: str, page_id: str, original: bool = False) -> bytes:
    with get_session() as s:
        p = s.get(Page, page_id)
        if not p or p.session_id != scan_id or p.deleted:
            raise NotFound(page_id)
        ref = p.original_ref if original else p.processed_ref
    return storage.get(ref)


# ---- edits (manual override) ----

def recrop(scan_id: str, page_id: str, corners: list[list[float]]) -> dict:
    """corners: normalized [[x,y]*4]. Re-run warp + downstream from the original."""
    with get_session() as s:
        p = _require_page(s, scan_id, page_id)
        if p.passthrough:
            raise ValueError("born-digital pages are kept as-is; cropping not applicable")
        original = _load(p.original_ref)
        result = process_page(original, mode=p.mode, forced_quad=corners, demoire=p.demoire)
        p.processed_ref = _store_processed(p.id, result.image)
        p.confidence = result.confidence
        p.gate_flags = result.flags
        p.quad = result.quad
        p.rotation = 0
        p.transforms = (p.transforms or []) + [{"op": "recrop"}]
        s.add(p)
        s.commit()
        s.refresh(p)
        dto = _page_dto(p)
    audit(None, "page.recrop", page_id)
    return dto


def rotate(scan_id: str, page_id: str, degrees: int) -> dict:
    if degrees % 90 != 0:
        raise ValueError("rotation must be a multiple of 90")
    with get_session() as s:
        p = _require_page(s, scan_id, page_id)
        if p.passthrough:
            # rotate the verbatim PDF page losslessly, then refresh its preview
            pdf_bytes = _rotate_pdf(storage.get(p.pdf_ref), degrees)
            p.pdf_ref = storage.put("processed", f"{p.id}.pdf", pdf_bytes)
            p.processed_ref = _store_processed(p.id, ingest.rasterize_pdf_bytes(pdf_bytes))
        else:
            img = _rotate_image(_load(p.processed_ref), degrees)
            p.processed_ref = _store_processed(p.id, img)
        p.rotation = (p.rotation + degrees) % 360
        p.transforms = (p.transforms or []) + [{"op": "rotate", "deg": degrees}]
        s.add(p)
        s.commit()
        s.refresh(p)
        dto = _page_dto(p)
    audit(None, "page.rotate", page_id, {"deg": degrees})
    return dto


def set_mode(scan_id: str, page_id: str, mode: str) -> dict:
    if mode not in ("color", "gray", "bw"):
        raise ValueError("mode must be color|gray|bw")
    with get_session() as s:
        p = _require_page(s, scan_id, page_id)
        if p.passthrough:
            raise ValueError("born-digital pages are kept as-is; mode change not applicable")
        original = _load(p.original_ref)
        result = process_page(original, mode=mode, forced_quad=p.quad, demoire=p.demoire)
        img = _rotate_image(result.image, p.rotation) if p.rotation else result.image
        p.processed_ref = _store_processed(p.id, img)
        p.mode = mode
        p.confidence = result.confidence
        p.gate_flags = result.flags
        p.transforms = (p.transforms or []) + [{"op": "mode", "mode": mode}]
        s.add(p)
        s.commit()
        s.refresh(p)
        dto = _page_dto(p)
    audit(None, "page.mode", page_id, {"mode": mode})
    return dto


def set_demoire(scan_id: str, page_id: str, on: bool) -> dict:
    """Toggle opt-in ML de-moiré (for screen photos); re-runs from the original."""
    with get_session() as s:
        p = _require_page(s, scan_id, page_id)
        if p.passthrough:
            raise ValueError("born-digital pages are kept as-is; de-moiré not applicable")
        original = _load(p.original_ref)
        result = process_page(original, mode=p.mode, forced_quad=p.quad, demoire=on)
        img = _rotate_image(result.image, p.rotation) if p.rotation else result.image
        p.processed_ref = _store_processed(p.id, img)
        p.demoire = on
        p.confidence = result.confidence
        p.gate_flags = result.flags
        p.transforms = (p.transforms or []) + [{"op": "demoire", "on": on}]
        s.add(p)
        s.commit()
        s.refresh(p)
        dto = _page_dto(p)
    audit(None, "page.demoire", page_id, {"on": on})
    return dto


def reorder(scan_id: str, page_ids: list[str]) -> dict:
    with get_session() as s:
        for idx, pid in enumerate(page_ids):
            p = _require_page(s, scan_id, pid)
            p.order = idx
            s.add(p)
        s.commit()
    audit(None, "scan.reorder", scan_id)
    return get_scan(scan_id)


def delete_page(scan_id: str, page_id: str) -> dict:
    with get_session() as s:
        p = _require_page(s, scan_id, page_id)
        p.deleted = True
        s.add(p)
        s.commit()
    audit(None, "page.delete", page_id)
    return get_scan(scan_id)


# ---- export ----

def export_pdf(scan_id: str, encoding: str = "jpeg", dpi: int = 200,
               ocr: bool = False) -> str:
    with get_session() as s:
        session = s.get(ScanSession, scan_id)
        if not session:
            raise NotFound(scan_id)
        pages = _ordered_pages(s, scan_id)
        if not pages:
            raise ValueError("no pages to export")
        # build one single-page PDF per page: born-digital pages are copied verbatim
        # (no rasterize/re-encode); processed image pages go through img2pdf.
        per_page: list[bytes] = []
        for p in pages:
            if p.passthrough:
                per_page.append(storage.get(p.pdf_ref))
            else:
                per_page.append(pdf_stage.image_to_pdf(_load(p.processed_ref),
                                                       encoding=encoding, dpi=dpi))
        confidences = [p.confidence for p in pages]

    data = pdf_stage.merge_pdfs(per_page)
    ref = storage.put("pdfs", f"{scan_id}.pdf", data)

    with get_session() as s:
        rec = Pdf(session_id=scan_id, ref=ref, encoding=encoding, dpi=dpi, ocr=ocr)
        s.add(rec)
        s.commit()

    metrics = {"pages": len(per_page), "mean_confidence": round(sum(confidences) / len(confidences), 3)}
    handoff_to_v4(scan_id, ref, metrics)        # V4 handoff stub (A3/Q1)
    audit(None, "scan.export", scan_id, {"encoding": encoding, "dpi": dpi})
    return ref


def get_pdf(scan_id: str) -> bytes:
    from sqlmodel import select

    with get_session() as s:
        rows = s.exec(select(Pdf).where(Pdf.session_id == scan_id)).all()
        if not rows:
            raise NotFound(scan_id)
        ref = sorted(rows, key=lambda r: r.created_at)[-1].ref
    return storage.get(ref)


# ---- helpers ----

def _require_page(s, scan_id: str, page_id: str) -> Page:
    p = s.get(Page, page_id)
    if not p or p.session_id != scan_id or p.deleted:
        raise NotFound(page_id)
    return p


def _rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    deg = degrees % 360
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _rotate_pdf(pdf_bytes: bytes, degrees: int) -> bytes:
    """Rotate a single-page PDF losslessly (pypdf)."""
    import io as _io

    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(_io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    page = reader.pages[0]
    page.rotate(degrees % 360)
    writer.add_page(page)
    buf = _io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
