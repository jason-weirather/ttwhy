"""One structured cell grid shared by plain text, ANSI, and the live viewer."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CellRun:
    """Adjacent characters sharing optional 8-bit RGB foreground/background."""

    text: str
    foreground: tuple[int, int, int] | None = None
    background: tuple[int, int, int] | None = None


class TextImage:
    """An owned, read-only rectangular character grid with optional RGB colors.

    Colors describe terminal cells, not quantitatively color-managed pixels.
    Plain serialization preserves spaces and line widths. ANSI serialization
    emits truecolor SGR only, never cursor movement or terminal control input.
    """

    def __init__(
        self,
        characters: NDArray,
        *,
        foreground: NDArray | None = None,
        background: NDArray | None = None,
    ) -> None:
        chars = np.asarray(characters)
        if chars.ndim != 2 or 0 in chars.shape or chars.dtype.kind != "U":
            raise ValueError("characters must be a nonempty 2D Unicode array")
        if any(len(x) != 1 or not x.isprintable() for x in np.unique(chars)):
            raise ValueError("Each cell must contain exactly one printable character")
        # All modes emit individual single-column codepoints, not grapheme clusters.
        from wcwidth import wcwidth

        if any(wcwidth(x) != 1 for x in np.unique(chars)):
            raise ValueError("Each character must occupy one terminal column")
        self.characters = np.array(chars, dtype="U1", copy=True)
        self.characters.flags.writeable = False
        self.foreground = self._own_colors(foreground)
        self.background = self._own_colors(background)

    def _own_colors(self, colors: NDArray | None) -> NDArray | None:
        if colors is None:
            return None
        value = np.asarray(colors)
        if value.shape != (*self.characters.shape, 3) or value.dtype != np.uint8:
            raise ValueError("Cell colors must be uint8 arrays with shape (rows, columns, 3)")
        result = value.copy()
        result.flags.writeable = False
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return self.characters.shape

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in (self.characters, self.foreground, self.background)
                   if x is not None)

    def iter_runs(self, row: int) -> Iterator[CellRun]:
        """Yield style runs for one row without exposing a framework dependency."""
        if not 0 <= row < self.shape[0]:
            raise IndexError("Text row is outside the cell grid")
        if self.foreground is None and self.background is None:
            yield CellRun("".join(self.characters[row]))
            return
        start = 0
        previous = None
        for column in range(self.shape[1]):
            foreground = (None if self.foreground is None else
                          tuple(int(x) for x in self.foreground[row, column]))
            background = (None if self.background is None else
                          tuple(int(x) for x in self.background[row, column]))
            colors = (foreground, background)
            if column and colors != previous:
                yield CellRun("".join(self.characters[row, start:column]), *previous)
                start = column
            previous = colors
        yield CellRun("".join(self.characters[row, start:]), *previous)

    def to_text(self, *, ansi: bool = False) -> str:
        """Serialize the grid, without a trailing newline or terminal side effects."""
        if not ansi or (self.foreground is None and self.background is None):
            return "\n".join("".join(row) for row in self.characters)
        lines = []
        for row in range(self.shape[0]):
            parts = []
            for run in self.iter_runs(row):
                codes = ["0"]
                if run.foreground is not None:
                    codes.append("38;2;" + ";".join(map(str, run.foreground)))
                if run.background is not None:
                    codes.append("48;2;" + ";".join(map(str, run.background)))
                parts.append("\x1b[" + ";".join(codes) + "m" + run.text)
            lines.append("".join(parts) + "\x1b[0m")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return (f"TextImage(rows={self.shape[0]}, columns={self.shape[1]}, "
                f"colored={self.foreground is not None})")
