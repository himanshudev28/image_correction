"""SQLite engine/session (PoC). Swap the URL for Postgres in production."""
from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine

from app.config import DB_PATH

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},  # FastAPI threadpool touches sessions
)


def init_db() -> None:
    # import models so SQLModel sees the table metadata before create_all
    import app.models  # noqa: F401

    SQLModel.metadata.create_all(_engine)


def get_session() -> Session:
    return Session(_engine)
