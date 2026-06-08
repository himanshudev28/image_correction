"""DocumentAI V4 handoff (stub) + audit logging.

A3/Q1: V4's real quality-assessment interface is TBD. For the PoC we log the
payload we WOULD hand off (cleaned-PDF ref + quality metrics) so the contract is
visible and testable. No PHI in logs (NFR-8) — refs and numeric metrics only.
"""
from __future__ import annotations

import logging

from app.db import get_session
from app.models import AuditEvent

logger = logging.getLogger("scanner.integration")


def handoff_to_v4(session_id: str, pdf_ref: str, metrics: dict) -> None:
    """Stubbed V4 integration. Replace with the real client when the interface lands."""
    logger.info("V4 handoff session=%s pdf=%s metrics=%s", session_id, pdf_ref, metrics)


def audit(actor: str | None, action: str, target: str | None, detail: dict | None = None) -> None:
    """Thin audit logger (NFR-5). Persists an AuditEvent; keep detail PHI-free."""
    try:
        with get_session() as s:
            s.add(AuditEvent(actor=actor, action=action, target=target, detail=detail or {}))
            s.commit()
    except Exception:  # noqa: BLE001 — auditing must never break the request path
        logger.exception("audit write failed action=%s target=%s", action, target)
