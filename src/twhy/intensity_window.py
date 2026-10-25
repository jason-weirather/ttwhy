"""Explicit raw-intensity windows; no source pixels are ever modified."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import positive_integer


@dataclass(frozen=True, slots=True)
class IntensityWindow:
    """Map raw values into [0, 1], then apply ``normalized ** (1 / gamma)``.

    Larger gamma brightens midtones. Nonfinite source values remain NaN so the
    renderer can distinguish missing pixels from valid dark pixels.
    """

    lower: float
    upper: float
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (self.lower, self.upper, self.gamma)):
            raise ValueError("Window limits and gamma must be finite")
        if self.upper <= self.lower or not isfinite(self.upper - self.lower):
            raise ValueError("Upper limit must exceed lower limit by a finite amount")
        if self.gamma <= 0:
            raise ValueError("Gamma must be positive")

    def apply(self, values: ArrayLike) -> NDArray[np.float32]:
        """Return an independent normalized array, preserving the source's shape."""
        source = np.asarray(values)
        if source.dtype.kind not in "buif":
            raise TypeError("Intensity values must be real numbers")
        # Float64 arithmetic avoids unsigned underflow and avoids collapsing
        # narrow windows on uint32 data before the subtraction has happened.
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (source.astype(np.float64) - self.lower) / (self.upper - self.lower)
            normalized = np.clip(normalized, 0.0, 1.0) ** (1.0 / self.gamma)
        normalized = np.where(np.isfinite(source), normalized, np.nan)
        return normalized.astype(np.float32)


def estimate_window(
    values: ArrayLike,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    max_samples: int = 262_144,
) -> IntensityWindow:
    """Estimate a deterministic finite-value window from an in-memory preview.

    Zeros participate. A sparse image whose percentiles collapse falls back to
    the sampled finite min/max. An all-zero image uses [0, 1]. A positive
    constant uses [0, constant]; other constants use [constant, constant + 1]
    where representable. This is display estimation, not stain normalization.
    No whole-slide scan is performed by this function.
    """
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
    max_samples = positive_integer(max_samples, "max_samples")
    source = np.asarray(values)
    if source.dtype.kind not in "buif":
        raise TypeError("Intensity values must be real numbers")
    flat = source.reshape(-1)
    if flat.size > max_samples:
        flat = flat[np.linspace(0, flat.size - 1, max_samples, dtype=np.intp)]
    finite = flat[np.isfinite(flat)].astype(np.float64)
    if not finite.size:
        raise ValueError("Cannot estimate contrast without finite image pixels")
    lower, upper = np.percentile(finite, [lower_percentile, upper_percentile])
    if upper <= lower:
        lower, upper = float(finite.min()), float(finite.max())
    if upper <= lower:
        if upper > 0:
            lower = 0.0
        else:
            upper = max(lower + 1.0, float(np.nextafter(lower, np.inf)))
    return IntensityWindow(float(lower), float(upper))
