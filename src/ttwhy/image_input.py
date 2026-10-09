"""File ingress into the existing Omeify Image -> ViewSession display pipeline.

OME-TIFF stays regional and uses only OMETiffReader. An ordinary raster or one
PDF page is decoded once, given a RAM-only display pyramid, and wrapped through
Omeify's public array API. No temporary TIFF, converter subprocess, backend
registry, or alternative sampling/rendering path is involved.
"""

from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from math import ceil, isfinite
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from .image_pyramid import image_pyramid

if TYPE_CHECKING:
    from omeify import Image

DEFAULT_PDF_DPI = 144.0
# Explicit allowlist: do not let Pillow's EPS/PS plugins launch Ghostscript.
RASTER_FORMATS = ("PNG", "JPEG", "WEBP", "BMP", "GIF", "ICO", "PPM", "TGA")
TIFF_HEADERS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
# PDFium is not thread-safe, even for different documents. All ttwhy PDF calls,
# including import, construction, rendering and cleanup, share this lock. Other
# code directly using PDFium in the same process must coordinate separately.
PDF_LOCK = RLock()


def image_kind(path: str | Path) -> Literal["tiff", "pdf", "raster"]:
    """Select ingress from content, not a filename extension or decoder failure."""
    with Path(path).open("rb") as stream:
        header = stream.read(1024)
    if header[:4] in TIFF_HEADERS:
        return "tiff"
    if header.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"%PDF-"):
        return "pdf"
    return "raster"


def _index(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _background(value: Sequence[int] | None) -> tuple[int, int, int]:
    if value is None:
        return (255, 255, 255)
    if len(value) != 3 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or not 0 <= v <= 255
        for v in value
    ):
        raise ValueError("background must contain three integer RGB values in 0..255")
    return tuple(int(v) for v in value)


def _raster_pixels(path: Path, frame: int, background: tuple[int, int, int]) -> NDArray:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError(
            "Raster viewing requires Pillow; reinstall ttwhy's dependencies"
        ) from error
    try:
        with Image.open(path, formats=list(RASTER_FORMATS)) as source:
            try:
                source.seek(frame)
            except EOFError as error:
                raise IndexError(
                    f"Frame {frame} does not exist; frame indices start at 0"
                ) from error
            source.load()
            # Loading first also makes PNG EXIF available. Work only on decoded
            # pixels; orientation edits never modify the source file.
            ImageOps.exif_transpose(source, in_place=True)
            if source.mode.startswith("I;16") or source.mode in ("I", "F"):
                pixels = np.array(source, copy=True)
                # Older Pillow versions expose 16-bit PNG as signed mode I.
                if source.format == "PNG" and source.mode == "I":
                    pixels = pixels.astype(np.uint16)
                transparent = source.info.get("transparency")
                if transparent is not None:
                    if pixels.dtype.kind != "u" or pixels.dtype.itemsize > 2:
                        raise ValueError("Transparency on this numeric raster mode is unsupported")
                    mask = pixels == transparent
                    pixels = np.repeat(pixels[..., None], 3, axis=-1)
                    maximum = np.iinfo(pixels.dtype).max
                    pixels[mask] = np.rint(np.asarray(background) * (maximum / 255))
                return pixels
            if "A" in source.getbands() or "transparency" in source.info:
                with source.convert("RGBA") as rgba, Image.new(
                    "RGBA", source.size, (*background, 255),
                ) as matte:
                    matte.alpha_composite(rgba)
                    with matte.convert("RGB") as rgb:
                        return np.array(rgb, copy=True)
            # Includes palette, bilevel, 8-bit gray and CMYK. Treat ordinary
            # photographs as RGB, not three independently normalized channels.
            with source.convert("RGB") as rgb:
                return np.array(rgb, copy=True)
    except UnidentifiedImageError as error:
        raise ValueError(
            "Unsupported or unrecognized image. Supported: OME-TIFF, PNG, JPEG, WebP, "
            "BMP, GIF, ICO, PPM/PGM/PBM, TGA, and PDF with ttwhy[pdf]. "
            "SVG, EPS/PS, HEIC/AVIF and raw camera formats are not enabled."
        ) from error
    except Image.DecompressionBombError as error:
        raise ValueError(f"Pillow rejected an excessively large raster: {error}") from error


def _pdf_pixels(path: Path, page: int, dpi: float,
                background: tuple[int, int, int]) -> NDArray:
    with PDF_LOCK:
        try:
            import pypdfium2 as pdfium
        except ImportError as error:
            raise RuntimeError(
                "PDF viewing requires the optional PDF renderer: "
                'python -m pip install "ttwhy[pdf]" '
                '(or python -m pip install -e ".[pdf]" in a checkout)'
            ) from error
        with pdfium.PdfDocument(path) as document:
            document.init_forms()
            count = len(document)
            if not 1 <= page <= count:
                raise IndexError(f"PDF page {page} does not exist; choose 1..{count}")
            with closing(document[page - 1]) as selected:
                scale = dpi / 72
                dimensions = (selected.get_width() * scale, selected.get_height() * scale)
                if any(not isfinite(v) or v <= 0 or ceil(v) >= 2**31 for v in dimensions):
                    raise ValueError(
                        "PDF raster dimensions are invalid or too large; lower --pdf-dpi"
                    )
                with closing(selected.render(
                    scale=scale, rev_byteorder=True, fill_color=(*background, 255),
                    draw_annots=True,
                )) as bitmap:
                    # to_numpy borrows native memory. Own RGB bytes before closing
                    # the bitmap/page/document, and before the UI worker starts.
                    return np.array(bitmap.to_numpy(), dtype=np.uint8, copy=True)


@contextmanager
def open_image(
    path: str | Path, *, series: int = 0, page: int | None = None,
    frame: int | None = None, pdf_dpi: float | None = None,
    background: Sequence[int] | None = None,
) -> Iterator["Image"]:
    """Open one OME series, ordinary image/frame, or rendered PDF page.

    series and frame are zero-based; PDF page is one-based (default 1).
    Ordinary images default to frame 0, PDFs to 144 DPI, transparency to white.
    Options for another input kind are rejected instead of silently ignored.
    OME-TIFF uses its original regional reader and stored pyramid unchanged.

    Other inputs are eagerly decoded into RAM, retaining the original decoded
    base resolution and area-mean overview levels. PDF resolution is fixed when
    opened: closer zoom magnifies that raster, not freshly rendered vectors.
    Unsigned RGB uses the usual full-range shared window. Higher-bit-depth
    scalar images keep raw intensities and the usual preview-based contrast.
    No ICC conversion, physical calibration inference, or disk caching occurs.

    Use the yielded open Omeify Image with ViewSession. As with Omeify's other
    array images, RAM remains owned while references to the image exist, even
    after close; drop those references to release it. TIFF errors never fall
    through to another decoder. Decoders are imported only when needed.
    """
    from omeify import ImageLevel, MultichannelImage, OMETiffReader, RGBImage

    path = Path(path)
    series = _index(series, "series")
    if page is not None:
        page = _index(page, "page", minimum=1)
    if frame is not None:
        frame = _index(frame, "frame")
    if pdf_dpi is not None and (
        isinstance(pdf_dpi, (bool, np.bool_)) or not isfinite(pdf_dpi) or pdf_dpi <= 0
    ):
        raise ValueError("pdf_dpi must be positive and finite")
    matte = _background(background)
    kind = image_kind(path)
    if kind == "tiff":
        if page is not None or frame is not None or pdf_dpi is not None or background is not None:
            raise ValueError(
                "TIFF uses --series; --page/--frame/--pdf-dpi/--background do not apply"
            )
        with OMETiffReader(path, series=series) as image:
            yield image
        return
    if series != 0:
        raise ValueError(
            "--series applies only to OME-TIFF; use --page for PDF or --frame for images"
        )
    if kind == "pdf":
        if frame is not None:
            raise ValueError("PDF uses --page, not --frame")
        dpi = DEFAULT_PDF_DPI if pdf_dpi is None else pdf_dpi
        pixels = _pdf_pixels(path, page or 1, dpi, matte)
    else:
        if page is not None or pdf_dpi is not None:
            raise ValueError("--page and --pdf-dpi apply only to PDF; use --frame for images")
        pixels = _raster_pixels(path, frame or 0, matte)
    arrays = image_pyramid(pixels)
    del pixels
    rgb = arrays[0].ndim == 3
    axes = "YXS" if rgb else "YX"
    levels = tuple(ImageLevel(i, axes, array.shape, downsample_yx=(2**i, 2**i))
                   for i, array in enumerate(arrays))
    factory = RGBImage if rgb else MultichannelImage
    with factory.from_array(
        arrays[0], axes=axes, level_arrays=arrays[1:], levels=levels, copy=False,
        channel_names=("RGB" if rgb else "Intensity",),
    ) as image:
        yield image
