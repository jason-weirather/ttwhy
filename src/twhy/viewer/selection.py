"""Cell-grid selections over one frozen, immutable TextImage."""

from __future__ import annotations

from ..text_image import TextImage
from ..selection_geometry import SelectionGeometry, union_mask


def _merge_runs(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for style, text in cells:
        if merged and merged[-1][0] == style:
            merged[-1] = (style, merged[-1][1] + text)
        else:
            merged.append((style, text))
    return merged


class RectangleSelection:
    """Arrows move a cursor; anchoring makes them adjust the opposite corner."""

    kind = "rectangle"
    supports_text = True
    supports_image = True
    supports_geojson = True

    def __init__(self, frame: TextImage, lines: list) -> None:
        self.frame, self.lines = frame, lines
        self.cursor = (frame.shape[1] // 2, frame.shape[0] // 2)
        self.anchor: tuple[int, int] | None = None
        self.dragging = False

    def clamp(self, x: int, y: int) -> tuple[int, int]:
        return (min(self.frame.shape[1] - 1, max(0, x)),
                min(self.frame.shape[0] - 1, max(0, y)))

    def move(self, dx: int, dy: int, *, extend: bool = False) -> None:
        if extend and self.anchor is None:
            self.anchor = self.cursor
        self.cursor = self.clamp(self.cursor[0] + dx, self.cursor[1] + dy)
        self.dragging = False

    def mark(self) -> None:
        """Start a new rectangle at the current cursor, even after an old selection."""
        self.anchor = self.cursor
        self.dragging = False

    def swap_corner(self) -> None:
        if self.anchor is not None:
            self.cursor, self.anchor = self.anchor, self.cursor
        self.dragging = False

    def start_drag(self, x: int, y: int) -> None:
        self.cursor = self.anchor = self.clamp(x, y)
        self.dragging = True

    def drag(self, x: int, y: int, *, finish: bool = False) -> None:
        if self.dragging:
            self.cursor = self.clamp(x, y)
        if finish:
            self.dragging = False

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """Return (y0, y1, x0, x1); both selected endpoint cells are included."""
        if self.anchor is None:
            return None
        x0, x1 = sorted((self.anchor[0], self.cursor[0]))
        y0, y1 = sorted((self.anchor[1], self.cursor[1]))
        return y0, y1 + 1, x0, x1 + 1

    def geometry(self) -> SelectionGeometry:
        if self.bounds is None:
            raise ValueError("Drag a box, or press Space then move with the arrows")
        return SelectionGeometry(self.kind, tuple(self.vertices()))

    @property
    def empty(self) -> bool:
        return self.anchor is None

    def clip(self) -> TextImage:
        if self.bounds is None:
            raise ValueError("Drag a box, or press Space then move with the arrows")
        return self.frame.crop(*self.bounds)

    def vertices(self) -> list[tuple[int, int]]:
        if self.bounds is None:
            return []
        y0, y1, x0, x1 = self.bounds
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def row(self, y: int, *, show_cursor: bool = True) -> list[tuple[str, str]]:
        """Overlay a visible ASCII border, never mutate the frame used for copying."""
        if not 0 <= y < self.frame.shape[0]:
            return []
        bounds = self.bounds
        x = 0
        result = []
        for style, text in self.lines[y]:
            for char in text:
                display_style = style
                if bounds is not None:
                    y0, y1, x0, x1 = bounds
                    inside = y0 <= y < y1 and x0 <= x < x1
                    horizontal, vertical = y in (y0, y1 - 1), x in (x0, x1 - 1)
                    if inside and (horizontal or vertical):
                        char = "+" if horizontal and vertical else "-" if horizontal else "|"
                        display_style = "class:clip.border"
                if show_cursor and (x, y) == self.cursor:
                    char, display_style = "+", "class:clip.cursor"
                if result and result[-1][0] == display_style:
                    result[-1] = (display_style, result[-1][1] + char)
                else:
                    result.append((display_style, char))
                x += 1
        return result


class PolygonSelection:
    """Keyboard and mouse polygon checkpoints shared by every export format."""

    kind = "polygon"
    supports_text = True
    supports_image = True
    supports_geojson = True

    def __init__(self, frame: TextImage, lines: list) -> None:
        self.frame, self.lines = frame, lines
        self.cursor = (frame.shape[1] // 2, frame.shape[0] // 2)
        self.points: list[tuple[int, int]] = []
        self.dragging = False
        self.closed = False

    def clamp(self, x: int, y: int) -> tuple[int, int]:
        return (min(self.frame.shape[1] - 1, max(0, x)),
                min(self.frame.shape[0] - 1, max(0, y)))

    def move(self, dx: int, dy: int, *, extend: bool = False) -> None:
        self.cursor = self.clamp(self.cursor[0] + dx, self.cursor[1] + dy)
        if extend:
            self.mark()
        self.dragging = False

    def mark(self) -> None:
        point = self.cursor
        if self.closed:
            self.dragging = False
            return
        if self.points and point == self.points[0] and len(self.points) >= 3:
            self.finish()
            return
        if not self.points or point != self.points[-1]:
            self.points.append(point)
        self.dragging = False

    def swap_corner(self) -> None:
        if self.closed:
            self.closed = False
        elif self.points:
            self.points.pop()
        self.dragging = False

    def start_drag(self, x: int, y: int) -> None:
        point = self.clamp(x, y)
        self.cursor = point
        if self.closed:
            self.dragging = False
            return
        if self.points and point == self.points[0] and len(self.points) >= 3:
            self.finish()
            return
        if not self.points or point != self.points[-1]:
            self.points.append(point)
        self.dragging = True

    def drag(self, x: int, y: int, *, finish: bool = False) -> None:
        self.cursor = self.clamp(x, y)
        if finish:
            self.dragging = False

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        points = self.points
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(ys), max(ys) + 1, min(xs), max(xs) + 1

    @property
    def empty(self) -> bool:
        return not self.points

    def finish(self) -> None:
        # Validate before changing state; a bad ring remains editable.
        SelectionGeometry(self.kind, tuple((x + .5, y + .5) for x, y in self.points))
        self.closed = True
        self.dragging = False

    def geometry(self) -> SelectionGeometry:
        if not self.closed:
            raise ValueError("Close the polygon: click its first point or press Enter")
        return SelectionGeometry(self.kind, tuple((x + .5, y + .5) for x, y in self.points))

    def clip(self) -> TextImage:
        geometry = self.geometry()
        y0, y1, x0, x1 = geometry.bounds
        cropped = self.frame.crop(y0, y1, x0, x1)
        mask = union_mask((geometry,), shape=cropped.shape, origin=(x0, y0))
        return TextImage(cropped.characters, foreground=cropped.foreground,
                         background=cropped.background, mask=mask)

    def vertices(self) -> list[tuple[int, int]]:
        return list(self.points)

    def _edge_points(self) -> dict[tuple[int, int], str]:
        overlay: dict[tuple[int, int], str] = {}

        def draw_segment(start: tuple[int, int], end: tuple[int, int]) -> None:
            x0, y0 = start
            x1, y1 = end
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            error = dx + dy
            x, y = x0, y0
            while True:
                char = "+" if (x, y) in (start, end) else ("|" if abs(y1 - y0) > abs(x1 - x0) else "-")
                overlay[(x, y)] = char
                if x == x1 and y == y1:
                    break
                e2 = 2 * error
                if e2 >= dy:
                    error += dy
                    x += sx
                if e2 <= dx:
                    error += dx
                    y += sy

        for left, right in zip(self.points, self.points[1:]):
            draw_segment(left, right)
        if self.closed and len(self.points) >= 3:
            draw_segment(self.points[-1], self.points[0])
        elif self.points:
            draw_segment(self.points[-1], self.cursor)
        return overlay

    def row(self, y: int, *, show_cursor: bool = True) -> list[tuple[str, str]]:
        if not 0 <= y < self.frame.shape[0]:
            return []
        overlay = self._edge_points()
        rendered: list[tuple[str, str]] = []
        x = 0
        for style, text in self.lines[y]:
            for char in text:
                display_style = style
                if (x, y) in overlay:
                    char = overlay[(x, y)]
                    display_style = "class:clip.border"
                if show_cursor and (x, y) == self.cursor:
                    char, display_style = "+", "class:clip.cursor"
                rendered.append((display_style, char))
                x += 1
        return _merge_runs(rendered)
