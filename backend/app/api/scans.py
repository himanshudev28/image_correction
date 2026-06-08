"""Scan API (PRD §11). Thin routes; CPU work runs in the threadpool via service.

The create/upload path runs the full auto pipeline synchronously (sub-second/page)
so the response already carries cleaned previews — the "upload and it's done" UX.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel

from app import service
from app.security import UploadRejected

router = APIRouter(prefix="/api/scans", tags=["scans"])


# ---- request bodies ----

class RecropBody(BaseModel):
    corners: list[list[float]]   # normalized [[x,y]*4]


class RotateBody(BaseModel):
    degrees: int                 # multiple of 90


class ModeBody(BaseModel):
    mode: str                    # color | gray | bw


class DemoireBody(BaseModel):
    on: bool                     # enable/disable ML de-moiré (screen photos)


class ReorderBody(BaseModel):
    page_ids: list[str]


class ExportBody(BaseModel):
    encoding: str = "jpeg"       # jpeg | png
    dpi: int = 200
    ocr: bool = False


# ---- routes ----

@router.post("")
async def create_scan(files: list[UploadFile] = File(...)):
    uploads = []
    for f in files:
        uploads.append((f.filename or "upload", f.content_type, await f.read()))
    try:
        scan_id = await run_in_threadpool(service.create_scan, uploads)
    except UploadRejected as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await run_in_threadpool(service.get_scan, scan_id)


@router.get("/{scan_id}")
async def get_scan(scan_id: str):
    try:
        return await run_in_threadpool(service.get_scan, scan_id)
    except service.NotFound:
        raise HTTPException(status_code=404, detail="scan not found")


@router.get("/{scan_id}/pages/{page_id}/preview")
async def page_preview(scan_id: str, page_id: str, original: bool = False):
    try:
        data = await run_in_threadpool(service.get_page_image, scan_id, page_id, original)
    except service.NotFound:
        raise HTTPException(status_code=404, detail="page not found")
    return Response(content=data, media_type="image/png")


@router.post("/{scan_id}/pages/{page_id}/recrop")
async def recrop(scan_id: str, page_id: str, body: RecropBody):
    if len(body.corners) != 4:
        raise HTTPException(status_code=400, detail="exactly 4 corners required")
    return await _edit(service.recrop, scan_id, page_id, body.corners)


@router.post("/{scan_id}/pages/{page_id}/rotate")
async def rotate(scan_id: str, page_id: str, body: RotateBody):
    return await _edit(service.rotate, scan_id, page_id, body.degrees)


@router.post("/{scan_id}/pages/{page_id}/mode")
async def set_mode(scan_id: str, page_id: str, body: ModeBody):
    return await _edit(service.set_mode, scan_id, page_id, body.mode)


@router.post("/{scan_id}/pages/{page_id}/demoire")
async def set_demoire(scan_id: str, page_id: str, body: DemoireBody):
    return await _edit(service.set_demoire, scan_id, page_id, body.on)


@router.post("/{scan_id}/pages/reorder")
async def reorder(scan_id: str, body: ReorderBody):
    try:
        return await run_in_threadpool(service.reorder, scan_id, body.page_ids)
    except service.NotFound:
        raise HTTPException(status_code=404, detail="page not found")


@router.delete("/{scan_id}/pages/{page_id}")
async def delete_page(scan_id: str, page_id: str):
    try:
        return await run_in_threadpool(service.delete_page, scan_id, page_id)
    except service.NotFound:
        raise HTTPException(status_code=404, detail="page not found")


@router.post("/{scan_id}/export")
async def export(scan_id: str, body: ExportBody):
    try:
        await run_in_threadpool(service.export_pdf, scan_id, body.encoding, body.dpi, body.ocr)
    except service.NotFound:
        raise HTTPException(status_code=404, detail="scan not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"pdf_url": f"/api/scans/{scan_id}/pdf"}


@router.get("/{scan_id}/pdf")
async def download_pdf(scan_id: str):
    try:
        data = await run_in_threadpool(service.get_pdf, scan_id)
    except service.NotFound:
        raise HTTPException(status_code=404, detail="pdf not found; export first")
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{scan_id}.pdf"'},
    )


async def _edit(fn, scan_id: str, page_id: str, arg):
    try:
        return await run_in_threadpool(fn, scan_id, page_id, arg)
    except service.NotFound:
        raise HTTPException(status_code=404, detail="page not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
