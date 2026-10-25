"""Navigate artwork algorithms, strict ASCII ramps, and independent color state.

``m`` is the artwork family (including monochrome Braille and Unicode shape).
``M`` is a separate, always-monochrome ASCII family. Switching families restores
its last style without consuming a cycle step, and does not change the artwork
family's remembered color preference. Half-blocks require color.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..text_renderer import DEFAULT_RAMP, RENDER_MODES, TEXT_MODES, TextRenderer


ASCII_VARIANTS = (
    ("classic", DEFAULT_RAMP),
    ("fine", "  .,:;irsXA253hMHGS#9B&@"),
    ("bold", " .oO0@#"),
)


@dataclass
class ModeNavigation:
    """Track the m/M families separately from the artwork's color preference.

    ``family='color'`` is the historical name for the ``m`` *artwork* family;
    it may be monochrome. ``color_enabled`` controls the actual use of color.
    """

    color_mode: str
    base_ramp: str = DEFAULT_RAMP
    ascii_index: int = 0
    family: str = "color"
    color_enabled: bool = True

    @classmethod
    def from_renderer(cls, renderer: TextRenderer) -> ModeNavigation:
        # Even ``--mode simple --no-color`` starts in the m family. Only pressing
        # capital M explicitly chooses the strict-ASCII ramp family.
        state = cls(renderer.mode, base_ramp=renderer.ramp, color_enabled=renderer.color)
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
        """Sync an explicit renderer edit without misclassifying monochrome art.

        In the M family, an inversion or other style edit does not alter the
        remembered m style or its color preference. A new render mode or colored
        renderer, however, is an explicit return to the m family.
        """
        if self.family == "ascii" and renderer.mode == "simple" and not renderer.color:
            ramps = tuple(ramp for _, ramp in self.ascii_variants)
            if renderer.ramp in ramps:
                self.ascii_index = ramps.index(renderer.ramp)
            else:
                self.base_ramp = renderer.ramp
                self.ascii_index = 0
            return
        self.family = "color"
        self.color_mode = renderer.mode
        self.color_enabled = renderer.color
        if renderer.mode == "simple":
            self.base_ramp = renderer.ramp

    def set_color(self, enabled: bool, *, fallback_mode: str = "detail") -> None:
        """Change only m's remembered color; M stays monochrome if active."""
        if not enabled and self.color_mode == "blocks":
            if fallback_mode not in TEXT_MODES:
                raise ValueError("A monochrome half-block fallback must be a text mode")
            self.color_mode = fallback_mode
        self.color_enabled = enabled

    def advance_color(self) -> None:
        """m: restore the artwork family, then cycle modes allowed by its color."""
        if self.family == "color":
            modes = RENDER_MODES if self.color_enabled else TEXT_MODES
            index = modes.index(self.color_mode)
            self.color_mode = modes[(index + 1) % len(modes)]
        self.family = "color"

    def advance_ascii(self) -> None:
        """M: restore the last strict ASCII style, then cycle its density ramps."""
        if self.family == "ascii":
            self.ascii_index = (self.ascii_index + 1) % len(self.ascii_variants)
        self.family = "ascii"

    def make_renderer(self, current: TextRenderer) -> TextRenderer:
        if self.family == "ascii":
            mode, color = "simple", False
            ramp = self.ascii_variants[self.ascii_index][1]
        else:
            mode, color = self.color_mode, self.color_enabled
            ramp = self.base_ramp if mode == "simple" else DEFAULT_RAMP
        return TextRenderer(mode=mode, color=color, ramp=ramp,
                            invert=current.invert, cell_aspect=current.cell_aspect)
