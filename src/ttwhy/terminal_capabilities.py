"""Conservative terminal rendering defaults; never probe by writing escape sequences.

The caller supplies its intended output context.  In particular, snapshots are not
terminals: ANSI snapshots are portable truecolor files by explicit request, while
plain snapshots stay ordinary text.  Terminal font coverage cannot be queried
reliably, so the encoding and advertised color depth are only proxies.
"""

import os

from prompt_toolkit.output import ColorDepth, create_output


def configured_color_depth() -> ColorDepth | None:
    """Match the full-screen application's color-depth override policy."""
    depth = ColorDepth.from_env()  # Also honors NO_COLOR, before any other hint.
    if depth is not None:
        return depth
    if os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
        return ColorDepth.DEPTH_24_BIT
    return None


def available_color_depth() -> ColorDepth:
    """Color depth that the normal prompt_toolkit terminal output would use."""
    return configured_color_depth() or create_output().get_default_color_depth()


def resolve_auto_mode(
    *,
    snapshot: bool,
    ansi: bool,
    color: bool,
    is_tty: bool = False,
    encoding: str | None = None,
    term: str | None = None,
    depth: ColorDepth | None = None,
) -> str:
    """Choose blocks when appropriate, otherwise portable simple text.

    256-color and truecolor terminals are eligible for blocks; prompt_toolkit
    quantizes RGB when necessary.  An explicit --mode blocks remains available
    independent of this heuristic.  No attempt is made to infer installed glyphs.
    """
    if not color:
        return "simple"
    if snapshot:
        return "blocks" if ansi else "simple"
    if not is_tty or (term or "").lower() in {"dumb", "unknown"}:
        return "simple"
    try:
        if encoding is None or "▀".encode(encoding).decode(encoding) != "▀":
            return "simple"
    except (UnicodeError, LookupError):
        return "simple"
    if depth in (ColorDepth.DEPTH_8_BIT, ColorDepth.DEPTH_24_BIT):
        return "blocks"
    return "simple"
