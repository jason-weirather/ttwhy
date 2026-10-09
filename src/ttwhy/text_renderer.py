"""Spatial detail, expressive shape, legacy density, and half-block renderers."""

from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import image_array, positive_integer
from .resampling import resample_image
from .text_image import TextImage

DEFAULT_RAMP = " .:-=+*#%@"
RenderMode = Literal["detail", "shape", "simple", "ascii", "blocks"]
TEXT_MODES = ("detail", "shape", "simple")
RENDER_MODES = (*TEXT_MODES, "blocks")
MAX_STRUCTURED_SAMPLES = 4_194_304


class TextRenderer:
    """Render normalized grayscale or RGB arrays without importing the viewer.

    Inputs must be in [0, 1], with nonfinite values allowed as missing pixels.
    Detail (default) fits tone-balanced Braille dot patterns. Shape chooses
    directional/curved strokes and texture from canonical geometry descriptors.
    Both retain 8-by-4 display samples per cell and use foreground colors only.
    Simple is the unchanged density ramp; "ascii" is an alias for simple.
    ``invert`` reverses ink density, not RGB hue. Blocks preserve the original
    upper/lower two-color preview and ignore inversion. Output is real text in
    every mode, but only simple guarantees ASCII characters.
    """

    def __init__(
        self,
        *,
        mode: RenderMode = "detail",
        color: bool = True,
        invert: bool = False,
        ramp: str = DEFAULT_RAMP,
        cell_aspect: float = 2.0,
    ) -> None:
        mode = "simple" if mode == "ascii" else mode
        if mode not in RENDER_MODES:
            raise ValueError("mode must be detail, shape, simple (alias ascii), or blocks")
        if mode == "blocks" and not color:
            raise ValueError(
                "Half-block rendering requires color; use detail, shape, or simple for plain output"
            )
        if len(ramp) < 2 or any(ord(x) < 32 or ord(x) > 126 for x in ramp):
            raise ValueError("ramp must contain at least two printable ASCII characters")
        if mode in ("detail", "shape") and ramp != DEFAULT_RAMP:
            raise ValueError("Custom density ramps require mode='simple'")
        if not isfinite(cell_aspect) or cell_aspect <= 0:
            raise ValueError("cell_aspect must be positive and finite")
        self.mode, self.color, self.invert = mode, color, invert
        self.ramp, self.cell_aspect = ramp, cell_aspect

    def sample_shape(self, *, columns: int, rows: int) -> tuple[int, int]:
        """Return the normalized raster shape needed to fill a terminal cell grid."""
        columns = positive_integer(columns, "columns")
        rows = positive_integer(rows, "rows")
        if self.mode in ("detail", "shape"):
            from .glyph_atlas import SAMPLE_YX

            height, width = SAMPLE_YX
            if rows * columns * height * width > MAX_STRUCTURED_SAMPLES:
                raise ValueError("Structured preview exceeds 4,194,304 samples; use fewer cells")
            return rows * height, columns * width
        return rows * (2 if self.mode == "blocks" else 1), columns

    def render(
        self,
        image: ArrayLike,
        *,
        columns: int = 80,
        rows: int | None = None,
    ) -> TextImage:
        """Resample a normalized image and return independently owned text cells.

        Omitted rows preserve square source-pixel geometry using cell_aspect.
        Explicit rows intentionally select the output geometry. For calibrated
        whole-slide viewports, ViewSession handles physical pixel anisotropy.
        """
        value = image_array(image)
        columns = positive_integer(columns, "columns")
        finite = value[np.isfinite(value)]
        if finite.size and (finite.min() < 0 or finite.max() > 1):
            raise ValueError("Normalize raw pixels with IntensityWindow.apply before rendering")
        if rows is None:
            rows = max(1, round(value.shape[0] / value.shape[1] * columns / self.cell_aspect))
        shape = self.sample_shape(columns=columns, rows=rows)
        pixels = (value.astype(np.float32, copy=True) if value.shape[:2] == shape else
                  resample_image(value, rows=shape[0], columns=shape[1]))
        if self.mode in ("detail", "shape"):
            from .structured_rendering import render_structured

            return render_structured(pixels, rows=rows, columns=columns, mode=self.mode,
                                     color=self.color, invert=self.invert)
        valid = (np.isfinite(pixels) if pixels.ndim == 2 else
                 np.isfinite(pixels).all(axis=-1))
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
        rgb = (np.repeat(pixels[..., None], 3, axis=-1) if pixels.ndim == 2 else pixels)
        if self.mode == "blocks":
            rgb[~valid] = 0.0
            colors = np.rint(rgb * 255).astype(np.uint8)
            chars = np.full((rows, columns), "▀", dtype="U1")
            chars[~valid[::2] & ~valid[1::2]] = " "
            return TextImage(chars, foreground=colors[::2], background=colors[1::2])
        brightness = rgb.max(axis=-1)
        density = 1.0 - brightness if self.invert else brightness
        indices = np.rint(density * (len(self.ramp) - 1)).astype(np.intp)
        chars = np.array(list(self.ramp), dtype="U1")[indices]
        chars[~valid] = " "
        foreground = None
        if self.color and pixels.ndim == 3:
            foreground_float = np.ones_like(rgb)
            np.divide(rgb, brightness[..., None], out=foreground_float,
                      where=brightness[..., None] > 0)
            foreground = np.rint(foreground_float * 255).astype(np.uint8)
        return TextImage(chars, foreground=foreground)

    def __repr__(self) -> str:
        return (f"TextRenderer(mode={self.mode!r}, color={self.color}, "
                f"invert={self.invert}, cell_aspect={self.cell_aspect})")
