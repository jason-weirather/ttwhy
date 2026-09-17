"""One worker request shares the same public sampling and rendering stages as Python."""

from dataclasses import dataclass

from ..intensity_window import IntensityWindow
from ..text_image import TextImage
from ..text_renderer import TextRenderer
from ..view_session import ViewSamples, ViewSession
from ..viewport import Viewport


@dataclass(frozen=True, slots=True)
class FrameRequest:
    generation: int
    viewport: Viewport
    columns: int
    rows: int
    channels: tuple[int, ...]
    windows: tuple[IntensityWindow | None, ...]
    renderer: TextRenderer


@dataclass(frozen=True, slots=True)
class FrameResult:
    request: FrameRequest
    frame: TextImage
    samples: ViewSamples
    windows: tuple[IntensityWindow, ...]
    lines: list[list[tuple[str, str]]]


def terminal_lines(frame: TextImage) -> list[list[tuple[str, str]]]:
    """Translate framework-free style runs into prompt_toolkit fragments once."""
    lines = []
    for row in range(frame.shape[0]):
        fragments = []
        for run in frame.iter_runs(row):
            style = "class:image"
            if run.foreground is not None:
                style += " fg:#" + "".join(f"{x:02x}" for x in run.foreground)
            if run.background is not None:
                style += " bg:#" + "".join(f"{x:02x}" for x in run.background)
            fragments.append((style, run.text))
        lines.append(fragments)
    return lines


def make_frame(session: ViewSession, request: FrameRequest) -> FrameResult:
    rows, columns = request.renderer.sample_shape(columns=request.columns, rows=request.rows)
    plan = session.plan(request.viewport, rows=rows, columns=columns, channels=request.channels)
    samples = session.sample(plan)
    styles = session.prepare_styles(samples, windows=[request.windows[i] for i in request.channels])
    frame = session.render(samples, styles=styles, renderer=request.renderer,
                           columns=request.columns, rows=request.rows)
    return FrameResult(
        request, frame, samples, tuple(style.window for style in styles), terminal_lines(frame)
    )
