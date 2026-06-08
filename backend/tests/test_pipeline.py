"""Per-stage + end-to-end pipeline tests on synthetic forms."""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app.pipeline import boundary, illumination, pdf
from app.pipeline.ingest import (
    downscale_for_detection,
    load_image,
    load_inputs,
    load_pdf_pages,
)
from app.pipeline.perspective import order_points
from app.pipeline.runner import process_page
from tests.synthetic import INK_BLUE_BGR, STAMP_RED_BGR, photo_of_form, render_form


def build_text_pdf(text: str) -> bytes:
    """Minimal valid born-digital (vector text) PDF, no external deps."""
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>",
        None,  # filled below (content stream)
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    stream = b"BT /F1 24 Tf 72 700 Td (" + text.encode() + b") Tj ET"
    objs[3] = b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream"

    pdf_bytes = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(pdf_bytes))
        pdf_bytes += str(i).encode() + b" 0 obj" + body + b"endobj\n"
    xref_pos = len(pdf_bytes)
    n = len(objs) + 1
    pdf_bytes += b"xref\n0 " + str(n).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        pdf_bytes += ("%010d 00000 n \n" % off).encode()
    pdf_bytes += (b"trailer<</Size " + str(n).encode() + b"/Root 1 0 R>>\nstartxref\n"
                  + str(xref_pos).encode() + b"\n%%EOF")
    return pdf_bytes


def _saturation(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]


def test_boundary_recovers_quad():
    photo, gt = photo_of_form(seed=1)
    small, scale = downscale_for_detection(photo)
    quad, method = boundary.find_quad(small, scale)
    assert quad is not None, "boundary detector returned no quad"
    found = order_points(quad)
    gt_ordered = order_points(gt)
    # corners should be within ~3% of the image diagonal
    diag = np.hypot(*photo.shape[:2])
    err = np.linalg.norm(found - gt_ordered, axis=1).max()
    assert err < 0.03 * diag, f"corner error {err:.1f}px too large ({method})"


def test_boundary_on_cluttered_background():
    """Mimics a real phone photo: textured floor, a finger, soft bottom edge.
    The robust detector should still crop (approx or rect fallback)."""
    photo, gt = photo_of_form(seed=5, cluttered=True)
    small, scale = downscale_for_detection(photo)
    quad, method = boundary.find_quad(small, scale)
    assert quad is not None, "no crop on cluttered photo"
    diag = np.hypot(*photo.shape[:2])
    err = np.linalg.norm(order_points(quad) - order_points(gt), axis=1).max()
    # looser tolerance: a rect fallback won't be pixel-perfect, but must be close
    assert err < 0.10 * diag, f"cluttered corner error {err:.1f}px too large ({method})"


def test_order_points_rotation_robust():
    # square corners given in scrambled order should always come back tl,tr,br,bl
    base = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], np.float32)
    for roll in range(4):
        scrambled = np.roll(base, roll, axis=0)
        out = order_points(scrambled)
        np.testing.assert_allclose(out, base, atol=1.0)


def test_illumination_preserves_color():
    """FR-19: blue ink and red stamp must survive illumination flattening."""
    form = render_form()
    # darken one side so flattening has real work to do
    form = (form.astype(np.float32) * np.linspace(0.5, 1.0, form.shape[1])[None, :, None])
    form = form.clip(0, 255).astype(np.uint8)
    flat = illumination.flatten(form)

    # sample the stamp and ink regions; saturation must remain clearly chromatic
    h, w = flat.shape[:2]
    stamp_region = flat[h - 270:h - 130, w - 230:w - 90]
    ink_region = flat[h - 290:h - 220, 80:420]
    assert _saturation(stamp_region).max() > 80, "red stamp lost its color"
    assert _saturation(ink_region).max() > 60, "blue ink lost its color"


def test_end_to_end_process_page():
    photo, _ = photo_of_form(seed=2)
    result = process_page(photo, mode="color")
    h, w = result.image.shape[:2]
    # output should be a plausible portrait-ish page, not blank
    assert result.image.size > 0
    aspect = h / w
    assert 1.0 < aspect < 2.0, f"unexpected aspect {aspect:.2f}"
    assert result.confidence > 0.4
    # background should be flattened bright; color should still be present somewhere
    assert result.image.mean() > 150
    assert _saturation(result.image).max() > 60


def test_pdf_assemble_is_metadata_free():
    photo, _ = photo_of_form(seed=3)
    page = process_page(photo).image
    data = pdf.assemble([page, page], encoding="jpeg", dpi=200)
    assert data[:5] == b"%PDF-"
    # the embedded JPEG should carry no EXIF (we re-encode via cv2, which adds none)
    assert b"Exif" not in data


def test_ingest_strips_metadata():
    """FR-3: an image with EXIF should come out of ingest with none."""
    form = render_form()
    pil = Image.fromarray(cv2.cvtColor(form, cv2.COLOR_BGR2RGB))
    exif = pil.getexif()
    exif[0x0110] = "TestCamera"   # Model tag
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", exif=exif)

    out = load_image(buf.getvalue())
    # round-trip back through PIL to inspect metadata of what ingest produced
    chk = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    assert len(chk.getexif()) == 0, "metadata not stripped on ingest"


def test_pdf_input_rasterizes():
    """PDF uploads rasterize to per-page images and run the same pipeline."""
    photo, _ = photo_of_form(seed=4)
    # build a 2-page PDF from the synthetic photo, then read it back
    page = process_page(photo).image
    data = pdf.assemble([page, page], encoding="jpeg", dpi=150)
    pages = load_pdf_pages(data)
    assert len(pages) == 2
    assert all(p.ndim == 3 and p.shape[2] == 3 for p in pages)


from pathlib import Path

_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_data"
# Real-world regression fixtures (business card, billing screen photo, registration
# form). They live in sample_data/ (gitignored), so these tests skip when absent.
_REAL_IMAGES = [
    "IMG-20260501-WA0001.jpg",
    "IMG-20260604-WA0009.jpeg",
    "IMG-20260607-WA0008.jpg",
]


@pytest.mark.parametrize("fname", _REAL_IMAGES)
def test_real_photo_quality_regression(fname):
    """Lock in the crop/clean quality achieved on real photos: a real crop must
    happen, geometry must be sound, and the page must come out bright + crisp."""
    path = _SAMPLE_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not present in sample_data/")
    from app.pipeline.ingest import load_image

    bgr = load_image(path.read_bytes())
    result = process_page(bgr)

    bad = {"boundary_not_found", "geometry_uncertain",
           "sanity_implausible_aspect", "sanity_mostly_uniform"}
    assert not (bad & set(result.flags)), f"{fname}: bad flags {result.flags}"
    assert result.confidence >= 0.8, f"{fname}: low confidence {result.confidence}"
    assert result.quad is not None, f"{fname}: no crop quad (fell back to full frame)"

    h, w = result.image.shape[:2]
    aspect = max(h, w) / min(h, w)
    assert aspect <= 2.6, f"{fname}: implausible output aspect {aspect:.2f}"
    assert float(result.image.mean()) > 120, f"{fname}: output too dark/not flattened"
    # crisp content present (the sharpen + flatten should leave clear edges)
    sharp = cv2.Laplacian(cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var()
    assert sharp > 150, f"{fname}: output looks blurred/blank ({sharp:.0f})"


def test_docaligner_fallback_if_model_present():
    """If the DocAligner model is installed, it should detect corners on a hard
    cluttered photo. Skips cleanly when the model file isn't fetched."""
    from app.pipeline import docaligner

    if not docaligner.available():
        pytest.skip("DocAligner model not present (run scripts/fetch_docaligner_model.py)")
    photo, gt = photo_of_form(seed=5, cluttered=True)
    quad = docaligner.detect_corners(photo)
    assert quad is not None and quad.shape == (4, 2)
    diag = np.hypot(*photo.shape[:2])
    err = np.linalg.norm(order_points(quad) - order_points(gt), axis=1).max()
    assert err < 0.05 * diag, f"DocAligner corner error {err:.1f}px too large"


def test_docaligner_detect_is_total():
    """detect_corners must never raise and returns either None or a 4x2 quad,
    whether or not the model is installed (graceful fallback contract)."""
    from app.pipeline import docaligner

    photo, _ = photo_of_form(seed=1)
    result = docaligner.detect_corners(photo)
    assert result is None or result.shape == (4, 2)


def test_born_digital_pdf_passes_through():
    """A vector/text PDF page is kept verbatim, NOT rasterized."""
    data = build_text_pdf("This is a born-digital invoice page used for passthrough testing")
    items = load_inputs("pdf", data)
    assert len(items) == 1
    assert items[0]["type"] == "pdf_page"
    assert items[0]["pdf_bytes"][:5] == b"%PDF-"


def test_scanned_pdf_is_rasterized_not_passed_through():
    """An image-only PDF (no extractable text) goes through the camera pipeline."""
    photo, _ = photo_of_form(seed=6)
    img_pdf = pdf.assemble([photo], encoding="jpeg", dpi=150)
    items = load_inputs("pdf", img_pdf)
    assert items[0]["type"] == "image"


def test_export_preserves_born_digital_text():
    """End-to-end via merge: the born-digital page keeps real text in the output."""
    import pypdfium2 as pdfium

    text = "Born digital export preservation check invoice 12345"
    page_pdf = build_text_pdf(text)
    photo, _ = photo_of_form(seed=7)
    img_pdf = pdf.image_to_pdf(process_page(photo).image, encoding="jpeg", dpi=150)
    merged = pdf.merge_pdfs([page_pdf, img_pdf])

    doc = pdfium.PdfDocument(merged)
    try:
        assert len(doc) == 2
        extracted = doc[0].get_textpage().get_text_range()
    finally:
        doc.close()
    assert "Born digital export" in extracted
