"""Reusable pathology-to-text rendering. The terminal UI is imported separately."""

from ._version import resolve_version
from .compositing import ChannelStyle, compose_channels
from .intensity_window import IntensityWindow, estimate_window
from .view_info import ReadBudgetExceeded
from .resampling import resample_image
from .text_image import TextImage
from .text_renderer import TextRenderer
from .view_session import ViewSession
from .viewport import Viewport

__version__ = resolve_version()

__all__ = [
    "ChannelStyle", "IntensityWindow", "ReadBudgetExceeded",
    "TextImage", "TextRenderer", "ViewSession", "Viewport", "__version__",
    "compose_channels", "estimate_window", "resample_image",
]
