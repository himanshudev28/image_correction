"""Data model (PRD §12). PoC uses SQLite via SQLModel.

JSON-ish fields (gate_flags, transforms) are stored as JSON text columns since
SQLite has no native array type.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ScanSession(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    owner: Optional[str] = None
    consent_ref: Optional[str] = None
    status: str = "processing"          # processing | ready | error
    created_at: datetime = Field(default_factory=_now)


class Page(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    session_id: str = Field(index=True, foreign_key="scansession.id")
    order: int = 0
    original_ref: Optional[str] = None     # storage key of sanitized original
    processed_ref: Optional[str] = None    # storage key of cleaned page image (preview)
    # born-digital passthrough: page kept verbatim as a single-page PDF (no pipeline,
    # no rasterize/re-encode). processed_ref then holds only a rasterized PREVIEW.
    passthrough: bool = False
    pdf_ref: Optional[str] = None          # storage key of the verbatim page PDF
    mode: str = "color"                    # color | gray | bw
    demoire: bool = False                  # opt-in ML de-moiré applied (screen photos)
    confidence: float = 0.0
    rotation: int = 0                      # applied rotation in degrees (0/90/180/270)
    # corners the warp used, normalized 0..1 [[x,y]*4]; lets the UI re-seed Adjust.
    quad: list = Field(default=None, sa_column=Column(JSON))
    gate_flags: list = Field(default_factory=list, sa_column=Column(JSON))
    transforms: list = Field(default_factory=list, sa_column=Column(JSON))
    deleted: bool = False


class Pdf(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    session_id: str = Field(index=True, foreign_key="scansession.id")
    ref: str
    encoding: str = "jpeg"
    dpi: int = 200
    ocr: bool = False
    created_at: datetime = Field(default_factory=_now)


class AuditEvent(SQLModel, table=True):
    """Audit trail (NFR-5). PoC: table exists; writes go through a thin logger
    (audit.record). No PHI in the payload (NFR-8)."""
    id: str = Field(default_factory=_uuid, primary_key=True)
    actor: Optional[str] = None
    action: str = ""
    target: Optional[str] = None
    detail: dict = Field(default_factory=dict, sa_column=Column(JSON))
    timestamp: datetime = Field(default_factory=_now)
