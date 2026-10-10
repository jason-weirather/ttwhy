"""Cell-grid rectangle selection over one frozen, immutable TextImage."""

from __future__ import annotations

from ..text_image import TextImage


class RectangleSelection:
    """Arrows move a cursor; anchoring makes them adjust the opposite corner."""

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

    def clip(self) -> TextImage:
        if self.bounds is None:
            raise ValueError("Drag a box, or press Space then move with the arrows")
        return self.frame.crop(*self.bounds)

    def row(self, y: int) -> list[tuple[str, str]]:
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
                if (x, y) == self.cursor:
                    char, display_style = "+", "class:clip.cursor"
                if result and result[-1][0] == display_style:
                    result[-1] = (display_style, result[-1][1] + char)
                else:
                    result.append((display_style, char))
                x += 1
        return result
