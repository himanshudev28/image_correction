"""FastAPI app entrypoint.

Dev: the Vite dev server (:5173) proxies /api to here (:8000); CORS for the Vite
origin is enabled as a backup. Prod: the built SPA is served same-origin from
./static, so no third-party origin is ever in the image path (FR-2/NFR-3).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.scans import router as scans_router
from app.config import DEV_FRONTEND_ORIGINS
from app.db import init_db
from app.storage import init_storage


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_storage()
    init_db()
    # Prefetch the ML corner model once on startup (best-effort, non-fatal) so the
    # first upload isn't slowed by the download. Falls back to classical if offline.
    from app.pipeline import docaligner

    docaligner.ensure_model_available()
    yield


app = FastAPI(title="Consent Document Scanner", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_FRONTEND_ORIGINS,   # dev backup; prod is same-origin
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(scans_router)

# Serve the built SPA in production if it exists (frontend/dist copied to ./static).
_static = Path(__file__).resolve().parent.parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="spa")
