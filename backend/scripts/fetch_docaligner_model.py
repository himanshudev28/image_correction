"""Download the DocAligner corner-detection model (Apache-2.0) ahead of time.

The server also auto-downloads this on first run, so this script is only needed
for offline/air-gapped prep or to fetch it explicitly before starting.

Run: python scripts/fetch_docaligner_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.pipeline import docaligner  # noqa: E402


def main() -> None:
    if docaligner.ensure_model_available():
        print(f"Model ready: {settings.docaligner.model_path}")
    else:
        print("Model could not be fetched (offline, or auto-download disabled).")
        sys.exit(1)


if __name__ == "__main__":
    main()
