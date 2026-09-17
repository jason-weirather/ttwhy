"""Camera geometry in base-image pixel-edge coordinates."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from ._validation import positive_integer


@dataclass(frozen=True, slots=True)
class Viewport:
    """A rectangular field of view; negative origins provide letterboxing."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (self.x, self.y, self.width, self.height)):
            raise ValueError("Viewport coordinates must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Viewport width and height must be positive")
        if not isfinite(self.x + self.width) or not isfinite(self.y + self.height):
            raise ValueError("Viewport endpoints must be finite")

    @classmethod
    def fit(
        cls,
        shape: Sequence[int],
        *,
        columns: int,
        rows: int,
        cell_aspect: float = 2.0,
        pixel_aspect: float = 1.0,
    ) -> "Viewport":
        """Fit a YX image, preserving physical shape and terminal-cell aspect.

        pixel_aspect is physical pixel height / physical pixel width. It is
        1 when calibration is unknown. The viewport remains in base pixels.
        """
        if len(shape) != 2:
            raise ValueError("shape must contain height and width")
        height, width = (positive_integer(x, "shape") for x in shape)
        columns = positive_integer(columns, "columns")
        rows = positive_integer(rows, "rows")
        if not all(isfinite(x) and x > 0 for x in (cell_aspect, pixel_aspect)):
            raise ValueError("Cell and pixel aspect ratios must be finite and positive")
        ratio = rows / columns * cell_aspect / pixel_aspect
        span_x = max(width, height / ratio)
        span_y = span_x * ratio
        return cls((width - span_x) / 2, (height - span_y) / 2, span_x, span_y)

    def pan(self, dx: float, dy: float) -> "Viewport":
        """Return a translated viewport; offsets are base-image pixels."""
        return type(self)(self.x + dx, self.y + dy, self.width, self.height)

    def zoom(self, factor: float, *, anchor: Sequence[float] = (0.5, 0.5)) -> "Viewport":
        """Return a zoomed viewport, preserving the image point under the anchor.

        factor > 1 zooms in. Anchor coordinates are fractional canvas positions.
        """
        if not isfinite(factor) or factor <= 0:
            raise ValueError("Zoom factor must be finite and positive")
        if len(anchor) != 2 or not all(isfinite(x) and 0 <= x <= 1 for x in anchor):
            raise ValueError("anchor must contain two values in [0, 1]")
        width, height = self.width / factor, self.height / factor
        return type(self)(self.x + anchor[0] * (self.width - width),
                          self.y + anchor[1] * (self.height - height), width, height)

    def centered_within(self, shape: Sequence[int]) -> "Viewport":
        """Keep the camera center on the image, while still allowing letterboxing."""
        if len(shape) != 2:
            raise ValueError("shape must contain height and width")
        height, width = (positive_integer(value, "shape") for value in shape)
        center_x = min(width, max(0.0, self.x + self.width / 2))
        center_y = min(height, max(0.0, self.y + self.height / 2))
        return type(self)(center_x - self.width / 2, center_y - self.height / 2,
                          self.width, self.height)
