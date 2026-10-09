"""Document selection and explicit ownership, above the unchanged Omeify Image API."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .image_input import _background, _index, array_image, image_kind, open_image, pdf_page_count
from .pdf_workers import DEFAULT_PDF_MAX_DPI, pdf_dpi_steps

if TYPE_CHECKING:
    from omeify import Image


class DocumentItem:
    """One immutable open image, retained independently by cache and active viewer.

    Reference accounting is event-loop-owned after load() returns. Releasing the
    final reference closes the image and drops the array-owning context. An old
    image can therefore be evicted without closing it underneath an active read.
    """

    def __init__(self, stack: ExitStack, image: Image, *, index: int,
                 dpi: float | None = None, nbytes: int = 0) -> None:
        self.image: Image | None = image
        self.index, self.dpi, self.nbytes = index, dpi, nbytes
        self._stack = stack
        self._references = 1

    def retain(self) -> DocumentItem:
        if self._references < 1:
            raise RuntimeError("Document image has been released")
        self._references += 1
        return self

    def release(self) -> None:
        if self._references < 1:
            raise RuntimeError("Document image released twice")
        self._references -= 1
        if not self._references:
            try:
                self._stack.close()
            finally:
                self.image = None


class ImageDocument:
    """Read one selected page/series; never scan pixels in the other OME series.

    load() may run in a source-loader thread. PDF refinement uses independent
    spawned processes; from_arrays() receives only completed owned RAM pyramids.
    The document does not own a renderer or introduce another image interface.
    """

    def __init__(self, path: str | Path, *, series: int = 0, page: int | None = None,
                 frame: int | None = None, pdf_max_dpi: float | None = None,
                 background=None) -> None:
        self.path = Path(path).resolve()
        self.kind = image_kind(self.path)
        self.series = _index(series, "series")
        self.page = None if page is None else _index(page, "page", minimum=1)
        self.frame = None if frame is None else _index(frame, "frame")
        self.background = _background(background)
        self._background_option = background
        if self.kind != "pdf" and pdf_max_dpi is not None:
            raise ValueError("--pdf-max-dpi applies only to PDF")
        self.dpi_steps = pdf_dpi_steps(
            DEFAULT_PDF_MAX_DPI if pdf_max_dpi is None else pdf_max_dpi,
        ) if self.kind == "pdf" else ()
        self.initial_index = (self.page or 1) if self.kind == "pdf" else series
        self.first_index = 1 if self.kind == "pdf" else 0
        self.count = 0

    @property
    def label(self) -> str:
        return "page" if self.kind == "pdf" else "series"

    @property
    def last_index(self) -> int:
        return self.first_index + self.count - 1

    def validate_index(self, index: int) -> int:
        index = _index(index, self.label, minimum=self.first_index)
        if not self.first_index <= index <= self.last_index:
            raise ValueError(f"Choose {self.label} {self.first_index}..{self.last_index}")
        return index

    def initialize(self) -> DocumentItem:
        item = self.load(self.initial_index)
        try:
            self.count = (pdf_page_count(self.path) if self.kind == "pdf" else
                          len(item.image.series_names) if self.kind == "tiff" else 1)
            self.validate_index(self.initial_index)
            return item
        except BaseException:
            item.release()
            raise

    def load(self, index: int) -> DocumentItem:
        """Open one quick preview or one region-readable OME series."""
        if self.count:
            self.validate_index(index)
        dpi = self.dpi_steps[0] if self.kind == "pdf" else None
        stack = ExitStack()
        try:
            image = stack.enter_context(open_image(
                self.path, series=index if self.kind == "tiff" else self.series,
                page=index if self.kind == "pdf" else self.page, frame=self.frame,
                pdf_dpi=dpi, background=self._background_option,
            ))
            size = (sum(int(np.prod(level.shape)) * image.dtype.itemsize for level in image.levels)
                    if self.kind == "pdf" else 0)
            return DocumentItem(stack, image, index=index, dpi=dpi, nbytes=size)
        except BaseException:
            stack.close()
            raise

    def from_arrays(self, index: int, dpi: float, arrays) -> DocumentItem:
        stack = ExitStack()
        try:
            image = stack.enter_context(array_image(arrays))
            return DocumentItem(stack, image, index=index, dpi=dpi,
                                nbytes=sum(array.nbytes for array in arrays))
        except BaseException:
            stack.close()
            raise

    def item_name(self, item: DocumentItem) -> str:
        selection = (f"page {item.index}/{self.count} | {item.dpi:g} DPI raster"
                     if self.kind == "pdf" else
                     f"series {item.index} of {self.count} (0-based)")
        return f"{self.path.name} | {selection}"
