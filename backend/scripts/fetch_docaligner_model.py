"""Download the DocAligner corner-detection model (Apache-2.0) for the ML fallback.

The model is the heatmap-regression ONNX published by DocsaidLab's DocAligner
(fastvit_sa24). We fetch only the single ONNX file — not the full toolkit — and
run it via onnxruntime. If the file is absent, the pipeline stays fully classical.

Run: python scripts/fetch_docaligner_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODELS_DIR  # noqa: E402

# Google Drive file id from the official DocAligner package (heatmap_reg, fastvit_sa24).
FILE_ID = "14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl"
DEST = MODELS_DIR / "docaligner_fastvit_sa24.onnx"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Model already present: {DEST}")
        return
    import gdown

    print(f"Downloading DocAligner model to {DEST} ...")
    gdown.download(id=FILE_ID, output=str(DEST), quiet=False)
    print("Done. The ML corner-detection fallback is now enabled.")


if __name__ == "__main__":
    main()
