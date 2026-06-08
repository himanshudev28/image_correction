"""Local-filesystem blob storage (PoC).

In production this becomes an encrypted object store (NFR-4). The interface is
deliberately key-based so the backend swaps without touching callers.
"""
from __future__ import annotations

from pathlib import Path

from app.config import STORAGE_DIR

_BUCKETS = ("originals", "processed", "pdfs")


def init_storage() -> None:
    for b in _BUCKETS:
        (STORAGE_DIR / b).mkdir(parents=True, exist_ok=True)


def _path(bucket: str, key: str) -> Path:
    assert bucket in _BUCKETS, f"unknown bucket {bucket}"
    return STORAGE_DIR / bucket / key


def put(bucket: str, key: str, data: bytes) -> str:
    p = _path(bucket, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return f"{bucket}/{key}"


def get(ref: str) -> bytes:
    bucket, key = ref.split("/", 1)
    return _path(bucket, key).read_bytes()


def path_of(ref: str) -> Path:
    bucket, key = ref.split("/", 1)
    return _path(bucket, key)
