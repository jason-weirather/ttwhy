"""File-free polygon geometry and union masks in terminal-cell edge coordinates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, floor, isfinite

import numpy as np
from numpy.typing import NDArray

Point = tuple[float, float]


def signed_area(points: tuple[Point, ...]) -> float:
    """Use a local origin to avoid cancellation for distant, small regions."""
    ox, oy = points[0]
    return sum((a[0] - ox) * (b[1] - oy) - (b[0] - ox) * (a[1] - oy)
               for a, b in zip(points, points[1:] + points[:1], strict=True)) / 2


def validate_polygon(points: Iterable[Point]) -> tuple[Point, ...]:
    """Validate one simple, nonzero-area ring; return vertices without closure."""
    values = tuple(tuple(float(v) for v in point) for point in points)
    if values and values[-1] == values[0]:
        values = values[:-1]
    if len(values) < 3:
        raise ValueError("A polygon needs at least three points")
    if any(len(point) != 2 or not all(isfinite(v) for v in point) for point in values):
        raise ValueError("Polygon points must be finite (x, y) pairs")
    if len(set(values)) != len(values):
        raise ValueError("Polygon vertices must not repeat")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    edges = tuple(zip(values, values[1:] + values[:1], strict=True))
    for i, (a, b) in enumerate(edges):
        c = values[(i + 2) % len(values)]
        if cross(a, b, c) == 0 and on_segment(a, b, c):
            raise ValueError("Polygon edges must not double back")
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            c, d = edges[j]
            ab_c, ab_d, cd_a, cd_b = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
            touching = ((ab_c == 0 and on_segment(a, b, c))
                        or (ab_d == 0 and on_segment(a, b, d))
                        or (cd_a == 0 and on_segment(c, d, a))
                        or (cd_b == 0 and on_segment(c, d, b)))
            crossing = ((ab_c > 0) != (ab_d > 0) and (cd_a > 0) != (cd_b > 0))
            if touching or crossing:
                raise ValueError("Polygon edges must not cross or touch")
    if abs(signed_area(values)) <= 1e-12:
        raise ValueError("Polygon must enclose a nonzero area")
    return values


@dataclass(frozen=True)
class SelectionGeometry:
    """An immutable shape snapshot. Polygon checkpoints lie at cell centers."""

    kind: str
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("rectangle", "polygon"):
            raise ValueError("Selection kind must be rectangle or polygon")
        object.__setattr__(self, "points", validate_polygon(self.points))

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        xs, ys = zip(*self.points, strict=True)
        return floor(min(ys)), ceil(max(ys)), floor(min(xs)), ceil(max(xs))


def union_bounds(regions: Iterable[SelectionGeometry]) -> tuple[int, int, int, int]:
    boxes = [region.bounds for region in regions]
    if not boxes:
        raise ValueError("Create a selection first")
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


def union_mask(regions: Iterable[SelectionGeometry], *, shape: tuple[int, int],
               origin: Point = (0, 0), scale: Point = (1, 1),
               image_bounds: tuple[float, float, float, float] | None = None) -> NDArray[np.bool_]:
    """Sample the union at output pixel/cell centers; edges count as selected.

    origin is the top-left in cell coordinates; scale is output pixels per cell
    (x, y). Row chunks bound scratch independently of the PNG output size.
    image_bounds optionally excludes letterboxing, never valid dark image pixels.
    """
    if (len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
            or len(scale) != 2 or any(not isfinite(v) or v <= 0 for v in scale)
            or len(origin) != 2 or not all(isfinite(v) for v in origin)):
        raise ValueError("Mask shape, origin and scale must describe a finite nonempty grid")
    regions = tuple(regions)
    rows, columns = shape
    mask = np.zeros(shape, dtype=bool)
    x = origin[0] + (np.arange(columns, dtype=np.float64)[None, :] + .5) / scale[0]
    for first in range(0, rows, 64):
        last = min(rows, first + 64)
        y = origin[1] + (np.arange(first, last, dtype=np.float64)[:, None] + .5) / scale[1]
        target = mask[first:last]
        for region in regions:
            inside = np.zeros(target.shape, bool)
            boundary = np.zeros(target.shape, bool)
            for (ax, ay), (bx, by) in zip(region.points, region.points[1:] + region.points[:1],
                                          strict=True):
                dx, dy = bx - ax, by - ay
                cross = (x - ax) * dy - (y - ay) * dx
                boundary |= ((np.abs(cross) <= 1e-9 * max(1, abs(dx), abs(dy)))
                             & (x >= min(ax, bx)) & (x <= max(ax, bx))
                             & (y >= min(ay, by)) & (y <= max(ay, by)))
                if dy:
                    inside ^= ((ay > y) != (by > y)) & (x < ax + (y - ay) * dx / dy)
            target |= inside | boundary
        if image_bounds is not None:
            left, top, right, bottom = image_bounds
            target &= (x >= left) & (x < right) & (y >= top) & (y < bottom)
    mask.flags.writeable = False
    return mask
