"""Separable pixel-area integration for small, bounded display rasters."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from ._validation import image_array, positive_integer


def _area_weights(length: int, start: float, span: float, count: int, extent: float) -> csr_matrix:
    """Intersection lengths between destination footprints and source pixels."""
    edges = start + np.arange(count + 1, dtype=np.float64) * (span / count)
    rows, columns, weights = [], [], []
    for row, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        first, last = max(0, int(np.floor(left))), min(length, int(np.ceil(right)))
        if first >= last:
            continue
        index = np.arange(first, last)
        overlap = np.maximum(0.0, np.minimum(np.minimum(right, index + 1), extent)
                             - np.maximum(left, index))
        rows.extend([row] * len(index))
        columns.extend(index)
        weights.extend(overlap)
    return csr_matrix((weights, (rows, columns)), shape=(count, length), dtype=np.float64)


def resample_image(
    image: ArrayLike,
    *,
    rows: int,
    columns: int,
    bounds: Sequence[float] | None = None,
    coverage_shape: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Return a new YX/YXS raster using exact separable area averaging.

    ``bounds`` is (x, y, width, height), in source pixel-edge coordinates.
    Omitted bounds cover the image. Bounds may extend outside it. Cells with
    no finite source coverage become NaN. Partially covered cells average
    only covered finite pixels. Missing samples never poison adjacent cells.
    Float64 output preserves narrow intensity windows on uint32 source values.
    Large files must be region-read before calling this in-memory operation.
    coverage_shape optionally gives the valid (height, width) in pixel-edge units.
    It can clip the last coarse pixel's support at an odd-sized base-image edge;
    it does not rescale pixel centers or alter the input array.
    """
    value = image_array(image)
    rows = positive_integer(rows, "rows")
    columns = positive_integer(columns, "columns")
    height, width = value.shape[:2]
    if bounds is None:
        bounds = (0.0, 0.0, float(width), float(height))
    if len(bounds) != 4:
        raise ValueError("bounds must contain x, y, width, height")
    x, y, span_x, span_y = bounds
    if (not all(np.isfinite(bounds)) or span_x <= 0 or span_y <= 0
            or not np.isfinite(x + span_x) or not np.isfinite(y + span_y)):
        raise ValueError("Bounds must be finite with positive width and height")
    coverage = (height, width) if coverage_shape is None else coverage_shape
    if (len(coverage) != 2 or not all(np.isfinite(coverage))
            or not 0 <= coverage[0] <= height or not 0 <= coverage[1] <= width):
        raise ValueError("coverage_shape must be finite nonnegative extents within the image")
    horizontal = _area_weights(width, x, span_x, columns, float(coverage[1]))
    vertical = _area_weights(height, y, span_y, rows, float(coverage[0]))
    planes = value[..., None] if value.ndim == 2 else value
    result = np.full((rows, columns, planes.shape[-1]), np.nan, dtype=np.float64)
    for index in range(planes.shape[-1]):
        plane = planes[..., index]
        finite = np.isfinite(plane)
        numerator = horizontal.dot(vertical.dot(np.where(finite, plane, 0.0)).T).T
        denominator = horizontal.dot(vertical.dot(finite.astype(np.float64)).T).T
        np.divide(numerator, denominator, out=result[..., index], where=denominator > 0)
    return result[..., 0] if value.ndim == 2 else result
