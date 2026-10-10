"""GeoJSON export for viewer selections in source-image pixel coordinates.

Coordinates are expressed in the source image's native pixel grid (x, y),
matching the image-space coordinates ttwhy already uses for viewports. The
output is plain GeoJSON FeatureCollection text so callers may save it directly
or place it on the clipboard.
"""

from __future__ import annotations

import json
from typing import Iterable


def _validate_shape(shape) -> tuple[int, int]:
    if (not isinstance(shape, tuple) or len(shape) != 2 or any(int(v) != v or v <= 0 for v in shape)):
        raise ValueError("raster shape must be a positive (rows, columns) pair")
    rows, columns = (int(shape[0]), int(shape[1]))
    return rows, columns


def _clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


def map_cell_point(x: float, y: float, *, viewport, frame_shape: tuple[int, int],
                   raster_shape: tuple[int, int]) -> tuple[float, float]:
    """Map a terminal-cell point to source-image pixel coordinates."""
    rows, columns = _validate_shape(frame_shape)
    image_rows, image_columns = _validate_shape(raster_shape)
    px = viewport.x + float(x) / max(1, columns) * viewport.width
    py = viewport.y + float(y) / max(1, rows) * viewport.height
    return (_clamp(px, 0.0, float(image_columns)),
            _clamp(py, 0.0, float(image_rows)))


def rectangle_feature(*, bounds: tuple[int, int, int, int], viewport,
                      frame_shape: tuple[int, int], raster_shape: tuple[int, int],
                      name: str | None = None, properties: dict | None = None) -> dict:
    y0, y1, x0, x1 = bounds
    ring = [
        map_cell_point(x0, y0, viewport=viewport, frame_shape=frame_shape, raster_shape=raster_shape),
        map_cell_point(x1, y0, viewport=viewport, frame_shape=frame_shape, raster_shape=raster_shape),
        map_cell_point(x1, y1, viewport=viewport, frame_shape=frame_shape, raster_shape=raster_shape),
        map_cell_point(x0, y1, viewport=viewport, frame_shape=frame_shape, raster_shape=raster_shape),
    ]
    ring.append(ring[0])
    props = dict(properties or {})
    if name:
        props.setdefault("name", name)
    props.setdefault("selection_type", "rectangle")
    return {"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]},
            "properties": props}


def polygon_feature(*, vertices: Iterable[tuple[int, int]], viewport,
                    frame_shape: tuple[int, int], raster_shape: tuple[int, int],
                    name: str | None = None, properties: dict | None = None) -> dict:
    points = list(vertices)
    if len(points) < 3:
        raise ValueError("A polygon selection needs at least three points")
    ring = [
        map_cell_point(x + 0.5, y + 0.5, viewport=viewport,
                       frame_shape=frame_shape, raster_shape=raster_shape)
        for x, y in points
    ]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    props = dict(properties or {})
    if name:
        props.setdefault("name", name)
    props.setdefault("selection_type", "polygon")
    return {"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]},
            "properties": props}


def feature_collection(feature: dict) -> dict:
    return {"type": "FeatureCollection", "features": [feature]}


def dumps(feature: dict) -> str:
    """Return indented GeoJSON text terminated by one newline on file write."""
    return json.dumps(feature_collection(feature), ensure_ascii=False, indent=2)
