"""Display-only pseudocolor composition of independently windowed markers."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .intensity_window import IntensityWindow

# Deterministic display colors, not inferred marker identities.
CHANNEL_COLORS = (
    (0.45, 0.65, 1.0),
    (1.0, 0.35, 0.65),
    (0.35, 1.0, 0.55),
    (1.0, 0.85, 0.25),
    (0.25, 1.0, 1.0),
    (0.85, 0.45, 1.0),
    (1.0, 0.60, 0.25),
)


@dataclass(frozen=True, slots=True)
class ChannelStyle:
    """A raw-intensity window and an additive RGB color in [0, 1]."""

    window: IntensityWindow
    color: Sequence[float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.window, IntensityWindow):
            raise TypeError("window must be an IntensityWindow")
        if len(self.color) != 3 or not all(np.isfinite(self.color)):
            raise ValueError("color must contain three finite RGB values")
        if any(x < 0 or x > 1 for x in self.color):
            raise ValueError("RGB color components must be in [0, 1]")
        # The record owns retained mutable input, rather than borrowing a list.
        object.__setattr__(self, "color", tuple(self.color))


def compose_channels(
    planes: ArrayLike, styles: Sequence[ChannelStyle]
) -> NDArray[np.float32]:
    """Window CYX planes, add their colors, and clip the composite into [0, 1].

    Input planes and styles remain unchanged. Invalid samples contribute no
    signal; a cell with no finite selected channel is marked NaN in all RGB
    samples. Channel order must match the explicit styles sequence.
    """
    values = np.asarray(planes)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("planes must be a nonempty CYX array")
    if values.dtype.kind not in "buif":
        raise TypeError("planes must have a real numeric dtype")
    if len(styles) != values.shape[0]:
        raise ValueError("Provide exactly one style for each selected channel")
    result = np.zeros((*values.shape[1:], 3), dtype=np.float32)
    valid = np.zeros(values.shape[1:], dtype=bool)
    for plane, style in zip(values, styles, strict=True):
        normalized = style.window.apply(plane)
        valid |= np.isfinite(normalized)
        result += np.nan_to_num(normalized, nan=0.0)[..., None] * np.asarray(style.color)
    np.clip(result, 0.0, 1.0, out=result)
    result[~valid] = np.nan
    return result
