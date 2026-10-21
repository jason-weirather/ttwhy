"""Bounded quick whole-image fit policy shared by the viewer and text snapshots.

A source's declared pyramid can often fit immediately. Otherwise a bounded
full-coverage reduction gets the caller's requested time budget before the
first image frame. The session's configured scan and cache limits protect huge
sources; timeouts are cooperative between TIFF region reads, since a native
decoder call cannot be interrupted safely.
"""

from math import isfinite
from time import monotonic

from .viewport import Viewport

DEFAULT_STARTUP_OVERVIEW_SECONDS = 2.0
def validate_startup_seconds(seconds: float) -> float:
    if isinstance(seconds, bool) or not isfinite(seconds) or seconds < 0:
        raise ValueError("startup_overview_seconds must be finite and nonnegative")
    return float(seconds)


def startup_fit(session, renderer, *, columns: int, rows: int, channels):
    """Return (full viewport, already-native) without reading image pixels."""
    full = Viewport.fit(session.info.shape, columns=columns, rows=rows,
                        cell_aspect=renderer.cell_aspect,
                        pixel_aspect=session.info.pixel_aspect)
    height, width = renderer.sample_shape(columns=columns, rows=rows)
    plan = session.plan(full, rows=height, columns=width, channels=channels,
                        allow_sampled_overview=True)
    return full, plan.sampled_overview is None


def prepare_fast_overview(session, channels, *, seconds: float) -> bool:
    """Synchronous snapshot preparation, one region per deadline check.

    The scan is eligible within the configured total-work and RAM caps. If incomplete,
    retained partial state can be resumed by a subsequent caller; no incomplete
    image is ever published. One native read may overrun the deadline.
    """
    if session.overview_ready(channels):
        return True
    if not seconds or not session.quick_overview_eligible(channels):
        return False
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        progress = session.step_overview(channels)
        if not progress.pending:
            break
    return session.overview_ready(channels)
