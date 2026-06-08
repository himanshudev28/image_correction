"""Generate synthetic input photos and their cleaned outputs into ../sample_data.

Run: python scripts/gen_samples.py
Useful for eyeballing the pipeline and for manual upload testing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.pdf import assemble  # noqa: E402
from app.pipeline.runner import process_page  # noqa: E402
from tests.synthetic import photo_of_form  # noqa: E402

OUT = Path(__file__).resolve().parent.parent.parent / "sample_data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = []
    for seed, skew in [(1, 0.18), (2, 0.10), (3, 0.25)]:
        photo, _ = photo_of_form(seed=seed, skew=skew)
        cv2.imwrite(str(OUT / f"input_{seed}.jpg"), photo)
        result = process_page(photo)
        cv2.imwrite(str(OUT / f"cleaned_{seed}.png"), result.image)
        pages.append(result.image)
        print(f"seed {seed}: confidence={result.confidence} flags={result.flags} "
              f"timings={result.timings_ms}")
    pdf = assemble(pages, encoding="jpeg", dpi=200)
    (OUT / "cleaned_multipage.pdf").write_bytes(pdf)
    print(f"wrote samples to {OUT}")


if __name__ == "__main__":
    main()
