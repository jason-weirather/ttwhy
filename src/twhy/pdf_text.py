"""Native PDF rectangle text extraction; no OCR, rasterization, or new dependency."""

from __future__ import annotations

import ctypes
from contextlib import closing
from math import isfinite
from pathlib import Path

from .image_input import PDF_LOCK, _index


def extract_pdf_text(path: str | Path, page: int, *,
                     bounds: tuple[float, float, float, float],
                     raster_shape: tuple[int, int]) -> str:
    """Extract a pixel-edge rectangle from the displayed page's embedded text.

    bounds is (left, top, right, bottom) on the actual displayed base raster.
    PDFium's own inverse device transform handles intrinsic rotation, CropBox,
    nonzero page origins and rounded raster dimensions. No high-DPI bitmap is
    allocated here. Text order and Unicode mappings are those supplied by PDFium;
    this is not layout analysis, OCR, or text from outlined vector lettering.
    """
    page = _index(page, "page", minimum=1)
    if len(raster_shape) != 2:
        raise ValueError("raster_shape must contain height and width")
    height, width = (_index(value, "raster dimension", minimum=1) for value in raster_shape)
    if max(height, width) >= 2**31:
        raise ValueError("Raster dimensions exceed PDFium's device-coordinate range")
    if len(bounds) != 4 or not all(isfinite(value) for value in bounds):
        raise ValueError("bounds must contain four finite pixel edges")
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        raise ValueError("bounds must be a nonempty rectangle")
    left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        return ""  # Letterboxing is not part of the PDF page.
    with PDF_LOCK:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw

        with pdfium.PdfDocument(Path(path)) as document:
            if page > len(document):
                raise IndexError(f"PDF page {page} does not exist")
            with closing(document[page - 1]) as selected:
                def point(x: int, y: int) -> tuple[float, float]:
                    px, py = ctypes.c_double(), ctypes.c_double()
                    if not raw.FPDF_DeviceToPage(selected, 0, 0, width, height, 0, x, y,
                                                 ctypes.byref(px), ctypes.byref(py)):
                        raise RuntimeError("PDFium could not map the selection to the page")
                    return px.value, py.value

                # Derive the affine transform using integer device coordinates,
                # then apply it to fractional edges without rounding the selection.
                origin, across, down = point(0, 0), point(width, 0), point(0, height)
                corners = [tuple(origin[i] + x / width * (across[i] - origin[i])
                                 + y / height * (down[i] - origin[i]) for i in (0, 1))
                           for x in (left, right) for y in (top, bottom)]
                xs, ys = zip(*corners, strict=True)
                with closing(selected.get_textpage()) as textpage:
                    text = textpage.get_text_bounded(min(xs), min(ys), max(xs), max(ys),
                                                     errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")
