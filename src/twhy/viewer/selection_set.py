"""Multiple independently named areas on a single frozen terminal frame."""

from __future__ import annotations

from dataclasses import dataclass

from ..annotation_identity import AnnotationIdentity, IdentityFactory
from ..selection_geometry import SelectionGeometry, union_bounds, union_mask
from ..text_image import TextImage
from .rendering import terminal_lines
from .selection import PolygonSelection, RectangleSelection, _merge_runs


@dataclass(frozen=True)
class AreaSnapshot:
    geometry: SelectionGeometry
    identity: AnnotationIdentity


@dataclass
class SelectionArea:
    shape: RectangleSelection | PolygonSelection
    identity: AnnotationIdentity


class SelectionSet:
    """Editing targets one area; all copy/save operations export their union.

    `a` adds an area; switching s/v keeps the previous one and changes tools.
    Brackets cycle the active area. An untouched new-area cursor is not exported;
    an unfinished polygon is an error, not silently omitted from the collection.
    """

    supports_text = supports_image = supports_geojson = True
    MAX_AREAS = 256

    def __init__(self, frame: TextImage, lines: list, kind: str, identities: IdentityFactory,
                 *, image_bounds=None) -> None:
        self.frame, self.lines = frame, lines
        self.identities = identities
        self.image_bounds = image_bounds
        self.areas = [self._new_area(kind)]
        self.active_index = 0

    def _new_area(self, kind: str) -> SelectionArea:
        if kind not in ("rectangle", "polygon"):
            raise ValueError("Unknown selection tool")
        cls = RectangleSelection if kind == "rectangle" else PolygonSelection
        return SelectionArea(cls(self.frame, self.lines), self.identities.create())

    @property
    def active(self) -> SelectionArea:
        return self.areas[self.active_index]

    @property
    def kind(self):
        return self.active.shape.kind

    @property
    def cursor(self):
        return self.active.shape.cursor

    @property
    def anchor(self):
        return getattr(self.active.shape, "anchor", None)

    @property
    def points(self):
        return getattr(self.active.shape, "points", [])

    @property
    def closed(self):
        return getattr(self.active.shape, "closed", False)

    @property
    def dragging(self):
        return self.active.shape.dragging

    @dragging.setter
    def dragging(self, value):
        self.active.shape.dragging = value

    @property
    def bounds(self):
        boxes = [area.shape.bounds for area in self.areas if not area.shape.empty]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), max(b[1] for b in boxes),
                min(b[2] for b in boxes), max(b[3] for b in boxes))

    @property
    def count(self):
        return sum(not area.shape.empty for area in self.areas)

    def rename(self, name: str) -> None:
        self.active.identity = self.active.identity.renamed(name)
        self.identities.used_names.add(self.active.identity.name)

    def add(self, kind: str | None = None) -> None:
        kind = kind or self.kind
        if self.active.shape.empty:
            if kind != self.kind:
                cls = RectangleSelection if kind == "rectangle" else PolygonSelection
                self.active.shape = cls(self.frame, self.lines)
            return
        self.active.shape.geometry()  # Validate without losing the current area.
        if len(self.areas) >= self.MAX_AREAS:
            raise ValueError("Selection limit is 256 areas; remove an area or finish this selection")
        area = self._new_area(kind)
        area.shape.cursor = self.cursor
        self.areas.append(area)
        self.active_index = len(self.areas) - 1

    def cycle(self, delta: int) -> None:
        self.dragging = False
        self.active_index = (self.active_index + delta) % len(self.areas)

    def remove(self) -> None:
        if len(self.areas) == 1:
            area = self._new_area(self.kind)
            self.areas[0] = area
        else:
            self.areas.pop(self.active_index)
            self.active_index = min(self.active_index, len(self.areas) - 1)

    def snapshots(self) -> tuple[AreaSnapshot, ...]:
        result = tuple(AreaSnapshot(a.shape.geometry(), a.identity)
                       for a in self.areas if not a.shape.empty)
        if not result:
            raise ValueError("Create a selection first")
        return result

    def vertices(self):
        return self.active.shape.vertices()

    def move(self, dx, dy, *, extend=False):
        self.active.shape.move(dx, dy, extend=extend)

    def mark(self):
        self.active.shape.mark()

    def swap_corner(self):
        self.active.shape.swap_corner()

    def finish(self):
        if self.kind == "polygon":
            self.active.shape.finish()

    def start_drag(self, x, y):
        self.active.shape.start_drag(x, y)

    def drag(self, x, y, *, finish=False):
        self.active.shape.drag(x, y, finish=finish)

    def set_frame(self, frame: TextImage) -> None:
        if frame.shape != self.frame.shape:
            raise ValueError("A selection frame must keep its cell geometry")
        self.frame, self.lines = frame, terminal_lines(frame)
        for area in self.areas:
            area.shape.frame, area.shape.lines = self.frame, self.lines

    def clip(self) -> TextImage:
        geometries = tuple(s.geometry for s in self.snapshots())
        y0, y1, x0, x1 = union_bounds(geometries)
        cropped = self.frame.crop(y0, y1, x0, x1)
        mask = union_mask(geometries, shape=cropped.shape, origin=(x0, y0))
        if mask.all():
            return cropped
        return TextImage(cropped.characters, foreground=cropped.foreground,
                         background=cropped.background, mask=mask)

    def row(self, y: int):
        if not 0 <= y < self.frame.shape[0]:
            return []
        cells = [(style, char) for style, text in self.lines[y] for char in text]
        order = [i for i in range(len(self.areas)) if i != self.active_index] + [self.active_index]
        for index in order:
            active = index == self.active_index
            fragments = self.areas[index].shape.row(y, show_cursor=active)
            column = 0
            for style, text in fragments:
                if style.startswith("class:clip."):
                    paint = style if active else "class:clip.saved"
                    for offset, char in enumerate(text):
                        cells[column + offset] = (paint, char)
                column += len(text)
        return _merge_runs(cells)
