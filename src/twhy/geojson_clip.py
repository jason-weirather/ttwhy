"""Image-pixel GeoJSON with explicit identities and no inherited source metadata."""

from __future__ import annotations

import json
from collections.abc import Iterable
from numbers import Integral
from uuid import uuid4

from .annotation_identity import AnnotationIdentity, IdentityFactory
from .selection_geometry import SelectionGeometry, signed_area, validate_polygon


def _validate_shape(shape) -> tuple[int, int]:
    if (len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
                               for v in shape)):
        raise ValueError("raster shape must be a positive (rows, columns) pair")
    return int(shape[0]), int(shape[1])


def map_cell_point(x: float, y: float, *, viewport, frame_shape: tuple[int, int],
                   raster_shape: tuple[int, int]) -> tuple[float, float]:
    """Map a cell-edge point to the source pixel grid, clamped to image bounds."""
    rows, columns = _validate_shape(frame_shape)
    height, width = _validate_shape(raster_shape)
    px = viewport.x + x / columns * viewport.width
    py = viewport.y + y / rows * viewport.height
    return min(width, max(0., px)), min(height, max(0., py))


def _clip_ring(points, height, width):
    """Intersect edges with the image, rather than distorting outside vertices.

    The common single-ring intersection is supported. If clipping disconnects a
    concave ROI, validation rejects the touching ring instead of exporting a
    misleading bridge. Select those components separately in that case.
    """
    for axis, limit, greater in ((0, 0., True), (0, float(width), False),
                                 (1, 0., True), (1, float(height), False)):
        if not points:
            break
        output = []
        previous = points[-1]
        prev_in = previous[axis] >= limit if greater else previous[axis] <= limit
        for point in points:
            inside = point[axis] >= limit if greater else point[axis] <= limit
            if inside != prev_in:
                t = (limit - previous[axis]) / (point[axis] - previous[axis])
                other = 1 - axis
                intersection = [0., 0.]
                intersection[axis] = limit
                intersection[other] = previous[other] + t * (point[other] - previous[other])
                output.append(tuple(intersection))
            if inside:
                output.append(point)
            previous, prev_in = point, inside
        points = []
        for point in output:
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
    if len(points) < 3 or abs(signed_area(tuple(points))) <= 1e-12:
        raise ValueError("Selection has no area inside the image")
    try:
        ring = validate_polygon(points)
    except ValueError as error:
        raise ValueError("Image-edge clipping splits this ROI; select its parts separately") from error
    if signed_area(ring) < 0:
        ring = (ring[0], *reversed(ring[1:]))
    return [list(point) for point in (*ring, ring[0])]


def geometry_feature(geometry: SelectionGeometry, *, viewport, frame_shape, raster_shape,
                     identity: AnnotationIdentity, properties: dict | None = None) -> dict:
    rows, columns = _validate_shape(frame_shape)
    height, width = _validate_shape(raster_shape)
    points = [(viewport.x + x / columns * viewport.width,
               viewport.y + y / rows * viewport.height) for x, y in geometry.points]
    ring = _clip_ring(points, height, width)
    # Deliberately constructed, not copied from a document, session or descriptor.
    props = dict(properties or {})
    props.update(name=identity.name, name_generated=identity.name_generated,
                 selection_type=geometry.kind)
    return {"type": "Feature", "id": identity.id,
            "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": props}


def _identity(name: str | None, identity: AnnotationIdentity | None) -> AnnotationIdentity:
    if identity is not None:
        return identity
    if name is not None:
        return AnnotationIdentity(str(uuid4()), name, False)
    return IdentityFactory().create()


def rectangle_feature(*, bounds: tuple[int, int, int, int], viewport,
                      frame_shape: tuple[int, int], raster_shape: tuple[int, int],
                      name: str | None = None, properties: dict | None = None,
                      identity: AnnotationIdentity | None = None) -> dict:
    y0, y1, x0, x1 = bounds
    geometry = SelectionGeometry("rectangle", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    return geometry_feature(geometry, viewport=viewport, frame_shape=frame_shape,
                            raster_shape=raster_shape, identity=_identity(name, identity),
                            properties=properties)


def polygon_feature(*, vertices: Iterable[tuple[int, int]], viewport,
                    frame_shape: tuple[int, int], raster_shape: tuple[int, int],
                    name: str | None = None, properties: dict | None = None,
                    identity: AnnotationIdentity | None = None) -> dict:
    geometry = SelectionGeometry("polygon", tuple((x + .5, y + .5) for x, y in vertices))
    return geometry_feature(geometry, viewport=viewport, frame_shape=frame_shape,
                            raster_shape=raster_shape, identity=_identity(name, identity),
                            properties=properties)


def feature_collection(features: dict | Iterable[dict]) -> dict:
    if isinstance(features, dict):
        features = [features]
    return {"type": "FeatureCollection", "features": list(features)}


def dumps(features: dict | Iterable[dict]) -> str:
    return json.dumps(feature_collection(features), ensure_ascii=False, allow_nan=False, indent=2)
