"""Budgeted, visibly approximate overview when a stored pyramid is too coarse to fit.

The ordinary ViewSession path performs area integration over every source pixel
in its bounded rectangle.  This fallback deliberately samples small, separated
source patches and interpolates their means for *navigation only*.  It neither
claims an exact overview nor materializes a huge stored TIFF level.
"""

from dataclasses import dataclass
from math import ceil, floor, isqrt, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from .view_info import PyramidLevel
from .viewport import Viewport


@dataclass(frozen=True, slots=True)
class SampledOverview:
    """One immutable grid of bounded level-local patch reads, in row-major order."""

    grid_shape: tuple[int, int]
    base_bounds: tuple[float, float, float, float]  # x0, y0, x1, y1
    regions: tuple[tuple[int, int, int, int], ...]  # y0, y1, x0, x1

    @property
    def pixels_per_channel(self) -> int:
        return sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in self.regions)


def plan_sampled_overview(
    viewport: Viewport, *, shape: tuple[int, int], level: PyramidLevel,
    rows: int, columns: int, pixel_budget: int,
) -> SampledOverview:
    """Cover the visible image with <=256 independent patches within pixel_budget.

    Allocate sample points according to the image's *projected* canvas extent,
    not its storage aspect. A portrait slide gets more vertical than horizontal
    observations. No patch requests pixels outside the selected stored level.
    """
    height, width = shape
    left, top = max(0.0, viewport.x), max(0.0, viewport.y)
    right = min(float(width), viewport.x + viewport.width)
    bottom = min(float(height), viewport.y + viewport.height)
    if right <= left or bottom <= top or pixel_budget < 1:
        raise ValueError("A sampled overview needs visible image pixels and a positive budget")

    visible_rows = rows * (bottom - top) / viewport.height
    visible_cols = columns * (right - left) / viewport.width
    # Reserve roughly an 8x8 native patch per observation when possible.
    # A tiny caller-specified budget should not cause hundreds of I/O calls.
    points = min(256, max(1, pixel_budget // 64))
    ratio = max(1.0, visible_rows) / max(1.0, visible_cols)
    grid_rows = max(1, min(points, level.height, ceil(visible_rows), round(sqrt(points * ratio))))
    grid_cols = max(1, min(points // grid_rows, level.width, ceil(visible_cols)))
    patches = grid_rows * grid_cols
    # Small localized averages mitigate single-pixel aliases while retaining
    # bounded I/O. Capping at 64 also avoids large vendor TIFF tile requests.
    side = max(1, min(64, isqrt(pixel_budget // patches)))
    step_y = (bottom - top) / (grid_rows * level.scale_y)
    step_x = (right - left) / (grid_cols * level.scale_x)
    patch_h = max(1, min(side, level.height, ceil(step_y)))
    patch_w = max(1, min(side, level.width, ceil(step_x)))

    regions = []
    for row in range(grid_rows):
        center_y = (top + (row + .5) * (bottom - top) / grid_rows) / level.scale_y
        y0 = max(0, min(level.height - patch_h, floor(center_y - patch_h / 2)))
        for col in range(grid_cols):
            center_x = (left + (col + .5) * (right - left) / grid_cols) / level.scale_x
            x0 = max(0, min(level.width - patch_w, floor(center_x - patch_w / 2)))
            regions.append((y0, y0 + patch_h, x0, x0 + patch_w))
    result = SampledOverview((grid_rows, grid_cols), (left, top, right, bottom), tuple(regions))
    assert result.pixels_per_channel <= pixel_budget
    return result


def patch_mean(pixels: NDArray) -> NDArray | np.float64:
    """Finite mean per logical sample, preserving RGB components independently."""
    finite = np.isfinite(pixels)
    count = finite.sum(axis=(0, 1))
    total = np.where(finite, pixels, 0).astype(np.float64).sum(axis=(0, 1))
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64),
                     where=count > 0)


def interpolate_overview(
    grid: NDArray, overview: SampledOverview, *, viewport: Viewport,
    shape: tuple[int, int], rows: int, columns: int,
) -> NDArray:
    """Place observed patch means on the canvas, leaving letterbox pixels NaN.

    Interpolation is a display approximation, not a physical pixel-scale estimate.
    Missing source values are handled by normalized finite-value weights.
    """
    grid_rows, grid_cols = overview.grid_shape
    if grid.shape[:2] != (grid_rows, grid_cols):
        raise ValueError("Overview grid dimensions do not match the read plan")
    left, top, right, bottom = overview.base_bounds
    ys = viewport.y + (np.arange(rows) + .5) * viewport.height / rows
    xs = viewport.x + (np.arange(columns) + .5) * viewport.width / columns
    inside = ((ys >= 0) & (ys < shape[0]))[:, None] & (
        (xs >= 0) & (xs < shape[1]))[None, :]
    gy = np.broadcast_to(((ys - top) / (bottom - top) * grid_rows - .5)[:, None],
                         (rows, columns))
    gx = np.broadcast_to(((xs - left) / (right - left) * grid_cols - .5)[None, :],
                         (rows, columns))
    coordinates = np.array((gy, gx), dtype=np.float64)
    planes = grid[..., None] if grid.ndim == 2 else grid
    result = np.full((rows, columns, planes.shape[-1]), np.nan, dtype=np.float64)
    for component in range(planes.shape[-1]):
        values = planes[..., component]
        valid = np.isfinite(values)
        numerator = map_coordinates(np.where(valid, values, 0), coordinates,
                                    order=1, mode="nearest", prefilter=False)
        denominator = map_coordinates(valid.astype(np.float64), coordinates,
                                      order=1, mode="nearest", prefilter=False)
        np.divide(numerator, denominator, out=result[..., component], where=denominator > 0)
    result[~inside] = np.nan
    return result[..., 0] if grid.ndim == 2 else result
