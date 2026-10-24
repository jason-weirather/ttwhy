"""Display-only channel colors shared by the viewer and snapshot frontend."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral
import re

from .compositing import CHANNEL_COLORS, ChannelStyle

RGBColor = tuple[float, float, float]
# A small explicit palette, not a CSS parser. Green is the additive RGB primary.
_COLOR_NAMES = {
    "red": "FF0000", "green": "00FF00", "blue": "0000FF",
    "cyan": "00FFFF", "magenta": "FF00FF", "yellow": "FFFF00",
    "orange": "FF8000", "purple": "8000FF", "violet": "8000FF",
    "white": "FFFFFF", "black": "000000", "gray": "808080", "grey": "808080",
}


def parse_color(text: str) -> RGBColor | None:
    """Parse hex or a named color; None explicitly requests the channel default."""
    if not isinstance(text, str):
        raise ValueError("Color must be a hex string, color name, or default")
    value = text.strip().casefold()
    if value == "default":
        return None
    value = _COLOR_NAMES.get(value, value)
    if not re.fullmatch(r"#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", value):
        raise ValueError("Use #RGB, #RRGGBB, a color name (e.g. blue/cyan/white), or default")
    value = value.removeprefix("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def color_hex(color: Sequence[float]) -> str:
    """Display a normalized color as uppercase 8-bit RGB hex."""
    return "#" + "".join(f"{round(component * 255):02X}" for component in color)


def default_color(index: int, channel_count: int, *, is_rgb: bool = False) -> RGBColor:
    """Preserve the existing palette exactly, including white single-channel views."""
    return ((1.0, 1.0, 1.0) if is_rgb or channel_count == 1 else
            CHANNEL_COLORS[index % len(CHANNEL_COLORS)])


def channel_palette(channel_count: int, *, is_rgb: bool = False,
                    overrides: Mapping[int, str] | None = None) -> list[RGBColor]:
    """Own a per-source palette; overrides use source indices, not selection order."""
    colors = [default_color(i, channel_count, is_rgb=is_rgb) for i in range(channel_count)]
    if is_rgb and overrides:
        raise ValueError("Channel pseudocolors apply only to scalar intensity channels, "
                         "not RGB images or PDF pages")
    for index, text in (overrides or {}).items():
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("Channel-color indices must be integers")
        if not 0 <= index < channel_count:
            raise ValueError(f"Channel-color index {index} is outside 0..{channel_count - 1}")
        color = parse_color(text)
        if color is not None:
            colors[index] = color
    return colors


def apply_channel_colors(styles: Sequence[ChannelStyle], channels: Sequence[int],
                         colors: Sequence[RGBColor] | None) -> tuple[ChannelStyle, ...]:
    """Keep the prepared windows; attach each sampled channel's current pseudocolor."""
    if colors is None:
        return tuple(styles)
    return tuple(replace(style, color=colors[index])
                 for style, index in zip(styles, channels, strict=True))
