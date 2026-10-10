"""Small, stateful navigation between colored artwork and strictly ASCII ramps.

The display algorithm is unchanged. Colored styles use existing render modes;
ASCII styles are three density vocabularies of the original simple renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..text_renderer import DEFAULT_RAMP, RENDER_MODES, TextRenderer


ASCII_VARIANTS = (
    ("classic", DEFAULT_RAMP),
    ("fine", "  .,:;irsXA253hMHGS#9B&@"),
    ("bold", " .oO0@#"),
)


@dataclass
class ModeNavigation:
    """Remember the last mode in each family, without a one-shot restore flag."""

    color_mode: str
    base_ramp: str = DEFAULT_RAMP
    ascii_index: int = 0
    family: str = "color"

    @classmethod
    def from_renderer(cls, renderer: TextRenderer) -> ModeNavigation:
        state = cls(renderer.mode, base_ramp=renderer.ramp)
        state.observe(renderer)
        return state

    def clone(self) -> ModeNavigation:
        return replace(self)

    @property
    def ascii_variants(self) -> tuple[tuple[str, str], ...]:
        return (("custom" if self.base_ramp != DEFAULT_RAMP else "classic", self.base_ramp),
                *ASCII_VARIANTS[1:])

    @property
    def label(self) -> str:
        return (f"ASCII {self.ascii_variants[self.ascii_index][0]}"
                if self.family == "ascii" else self.color_mode)

    def observe(self, renderer: TextRenderer) -> None:
        """Synchronize explicit color/blocks/inversion changes with family navigation."""
        if renderer.color:
            self.family = "color"
            self.color_mode = renderer.mode
            if renderer.mode == "simple":
                self.base_ramp = renderer.ramp
        elif renderer.mode == "simple":
            ramps = tuple(ramp for _, ramp in self.ascii_variants)
            if renderer.ramp in ramps:
                self.family = "ascii"
                self.ascii_index = ramps.index(renderer.ramp)
            else:
                self.base_ramp = renderer.ramp
                self.ascii_index = 0
                self.family = "ascii"
        else:
            # Monochrome Braille/shape from the legacy t toggle is not strict ASCII.
            self.family = "other"
            self.color_mode = renderer.mode

    def advance_color(self) -> None:
        """From ASCII, restore the last color style first; otherwise cycle."""
        if self.family == "color":
            index = RENDER_MODES.index(self.color_mode)
            self.color_mode = RENDER_MODES[(index + 1) % len(RENDER_MODES)]
        self.family = "color"

    def advance_ascii(self) -> None:
        """From color, enter the last ASCII style first; otherwise cycle."""
        if self.family == "ascii":
            self.ascii_index = (self.ascii_index + 1) % len(self.ascii_variants)
        self.family = "ascii"

    def make_renderer(self, current: TextRenderer) -> TextRenderer:
        if self.family == "ascii":
            mode, color = "simple", False
            ramp = self.ascii_variants[self.ascii_index][1]
        else:
            mode, color = self.color_mode, True
            ramp = self.base_ramp if mode == "simple" else DEFAULT_RAMP
        return TextRenderer(mode=mode, color=color, ramp=ramp,
                            invert=current.invert, cell_aspect=current.cell_aspect)
