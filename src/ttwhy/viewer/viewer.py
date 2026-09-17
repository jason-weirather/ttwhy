"""A full-screen microscopy viewer with one coalescing render worker."""

import asyncio
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.input import Input
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from ..intensity_window import IntensityWindow
from ..ome_tiff_source import ReadBudgetExceeded
from ..text_image import TextImage
from ..text_renderer import TextRenderer
from ..view_session import ViewSamples, ViewSession
from ..viewport import Viewport
from .bindings import create_bindings
from .controls import CanvasControl, ChannelControl, safe_label
from .rendering import FrameRequest, make_frame


class Viewer:
    """Interactive frontend for a borrowed ViewSession, run with ``await run()``.

    The UI loop owns camera and control state. One executor thread owns all
    sampling and rendering work. Only one pending request is retained, replacing
    older pending requests. In-flight reads finish safely, but stale frames do
    not replace newer views. ``run`` waits for the worker before returning, so
    the source's context manager cannot close underneath a decode.
    """

    def __init__(
        self,
        session: ViewSession,
        *,
        channels: Sequence[int] = (0,),
        window: IntensityWindow | None = None,
        renderer: TextRenderer | None = None,
        viewport: Viewport | None = None,
    ) -> None:
        self.session = session
        self.selected_channels = session.source.select_channels(channels)
        self.focused_channel = self.selected_channels[0]
        self.windows = [window] * len(session.source.info.channels)
        self.renderer = renderer or TextRenderer(invert=session.source.info.is_rgb)
        self.viewport = viewport
        self.frame: TextImage | None = None
        self.samples: ViewSamples | None = None
        self.cell_lines: list[list[tuple[str, str]]] = []
        self.canvas_size = (0, 0)
        self.panel_requested = True
        self.show_help = False
        self.editing: str | None = None
        self.message = "Opening image..."
        self.busy = False
        self.generation = 0
        self.displayed_generation = -1
        self._pending: FrameRequest | None = None
        self._wake: asyncio.Event | None = None
        self._executor: ThreadPoolExecutor | None = None
        self.application: Application | None = None
        self.canvas = CanvasControl(self)
        self.channel_control = ChannelControl(self)
        self.entry = TextArea(
            multiline=False, height=1, accept_handler=self._accept_edit,
            prompt=self._entry_prompt, style="class:entry",
        )

    def _entry_prompt(self) -> str:
        units = "unitless" if self.editing == "gamma" else "raw units"
        return f" {self.editing or ''} ({units}; Enter / Esc): "

    @property
    def panel_visible(self) -> bool:
        return bool(self.panel_requested and self.application is not None
                    and self.application.output.get_size().columns >= 70)

    def _build_application(self, input: Input | None, output: Output | None) -> Application:
        panel = ConditionalContainer(
            VSplit([Window(width=1, char="│", style="class:divider"),
                    Window(self.channel_control, width=31, wrap_lines=False,
                           style="class:panel", always_hide_cursor=True)]),
            filter=Condition(lambda: self.panel_visible),
        )
        body = VSplit([Window(self.canvas, style="class:image", always_hide_cursor=True), panel])
        container = HSplit([
            Window(FormattedTextControl(self._title), height=1, style="class:title"),
            body,
            Window(FormattedTextControl(self._status), height=1, style="class:status"),
            ConditionalContainer(self.entry, filter=Condition(lambda: self.editing is not None)),
            Window(FormattedTextControl(self._footer), height=2, style="class:footer"),
        ])
        return Application(
            layout=Layout(container, focused_element=self.canvas),
            key_bindings=create_bindings(self), full_screen=True, mouse_support=True,
            input=input, output=output, min_redraw_interval=1 / 30,
            color_depth=(ColorDepth.from_env() or
                         (ColorDepth.DEPTH_24_BIT if os.environ.get("COLORTERM", "").lower()
                          in {"truecolor", "24bit"} else None)),
            style=Style.from_dict({
                "image": "bg:#000000 #eeeeee", "title": "bg:#253247 #ffffff bold",
                "panel": "bg:#141c29 #bdc8da", "panel.title": "#83d5c8 bold",
                "panel.selected": "bg:#31415b #ffffff bold", "divider": "#405570",
                "status": "bg:#253247 #d7e1f0", "footer": "bg:#141c29 #a8b7cc",
                "entry": "bg:#394a63 #ffffff",
            }),
        )

    async def run(self, *, input: Input | None = None, output: Output | None = None) -> None:
        """Run until quit. Optional terminal I/O supports embedding and real input tests."""
        if self.application is not None:
            raise RuntimeError("Create a new Viewer for each terminal session")
        self._wake = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ttwhy-render")
        self.application = self._build_application(input, output)
        try:
            await self.application.run_async(
                pre_run=lambda: self.application.create_background_task(self._pump())
            )
        finally:
            # Worker cancellation does not interrupt a TIFF codec. Join before
            # allowing the caller to leave its source context manager.
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._pending = None
            self.session.clear_cache()

    async def _pump(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            request, self._pending = self._pending, None
            if request is None:
                continue
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self._executor, make_frame, self.session, request
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.application.exit(exception=error)
                return
            if request.generation != self.generation:
                continue
            self.frame, self.samples = result.frame, result.samples
            self.cell_lines = result.lines
            self.displayed_generation = request.generation
            for position, channel in enumerate(request.channels):
                # Do not freeze a fictitious auto-window on an empty field.
                values = (result.samples.values if result.samples.is_rgb
                          else result.samples.values[position])
                if np.isfinite(values).any():
                    self.windows[channel] = result.windows[position]
            self.busy = False
            self.application.invalidate()

    def request_frame(self) -> None:
        if self.viewport is None or not all(self.canvas_size) or self._wake is None:
            return
        self.generation += 1
        columns, rows = self.canvas_size
        self._pending = FrameRequest(self.generation, self.viewport, columns, rows,
                                     self.selected_channels, tuple(self.windows), self.renderer)
        self.busy = True
        self._wake.set()
        self.application.invalidate()

    def resize_canvas(self, columns: int, rows: int) -> None:
        if columns < 1 or rows < 1 or (columns, rows) == self.canvas_size:
            return
        previous = self.canvas_size
        self.canvas_size = (columns, rows)
        if self.viewport is None:
            self.fit()
        elif all(previous):
            old = self.viewport
            # Preserve the center and horizontal base pixels per terminal cell.
            width = old.width * columns / previous[0]
            height = width / columns * rows * self.renderer.cell_aspect
            height /= self.session.source.info.pixel_aspect
            candidate = Viewport(old.x + (old.width - width) / 2,
                                 old.y + (old.height - height) / 2, width, height)
            self._set_viewport(candidate, limit=True)
        else:
            self._set_viewport(self.viewport, limit=True)

    def _set_viewport(self, viewport: Viewport, *, limit: bool = False) -> None:
        rows, columns = self.renderer.sample_shape(columns=self.canvas_size[0],
                                                   rows=self.canvas_size[1])
        viewport = viewport.centered_within(self.session.source.info.shape)
        for _ in range(128):
            try:
                self.session.plan(viewport, rows=rows, columns=columns,
                                  channels=self.selected_channels)
                break
            except ReadBudgetExceeded as error:
                if not limit:
                    self.message = str(error)
                    self.application.invalidate()
                    return
                viewport = viewport.zoom(1.5)
                self.message = "Read budget: showing a bounded field, not a whole-slide overview."
        else:
            raise ReadBudgetExceeded("Read budget cannot accommodate this channel selection")
        self.viewport = viewport
        self.request_frame()

    def fit(self) -> None:
        if not all(self.canvas_size):
            return
        viewport, limited = self.session.fit(
            columns=self.canvas_size[0], rows=self.canvas_size[1], renderer=self.renderer,
            channels=self.selected_channels,
        )
        self.message = ("Read budget: bounded center crop; no usable overview at this size."
                        if limited else "Fit slide. Contrast stays fixed until a / l / u / g.")
        self.viewport = viewport
        self.request_frame()

    def zoom(self, factor: float, *, anchor: Sequence[float] = (0.5, 0.5)) -> None:
        if self.viewport is not None:
            candidate = self.viewport.zoom(factor, anchor=anchor)
            # Extreme magnification adds no information and risks loss of
            # floating-point coordinate precision on large slides.
            if candidate.width / max(1, self.canvas_size[0]) < 0.05:
                self.message = "Maximum display magnification reached."
                self.application.invalidate()
                return
            self.message = "Wheel zoom is anchored to the pointer."
            self._set_viewport(candidate)

    def pan_cells(self, dx: float, dy: float) -> None:
        if self.viewport is not None:
            self.message = ""
            self._set_viewport(self.viewport.pan(dx * self.viewport.width / self.canvas_size[0],
                                                dy * self.viewport.height / self.canvas_size[1]))

    def focus_channel(self, delta: int) -> None:
        self.focused_channel = min(len(self.windows) - 1, max(0, self.focused_channel + delta))
        self.application.invalidate()

    def select_one(self, index: int) -> None:
        if index < len(self.windows):
            self.focused_channel = index
            self.selected_channels = (index,)
            self._set_viewport(self.viewport, limit=True)

    def toggle_channel(self) -> None:
        selected = set(self.selected_channels)
        if self.focused_channel in selected:
            if len(selected) == 1:
                self.message = "Keep at least one channel selected."
                self.application.invalidate()
                return
            selected.remove(self.focused_channel)
        else:
            selected.add(self.focused_channel)
        self.selected_channels = tuple(sorted(selected))
        self._set_viewport(self.viewport, limit=True)

    def change_renderer(self, **changes) -> None:
        options = {"mode": self.renderer.mode, "color": self.renderer.color,
                   "invert": self.renderer.invert, "ramp": self.renderer.ramp,
                   "cell_aspect": self.renderer.cell_aspect}
        options.update(changes)
        self.renderer = TextRenderer(**options)
        self.message = "Half-block colors are display previews, not color-managed pixels."
        self._set_viewport(self.viewport, limit=True)

    def auto_window(self) -> None:
        index = self.focused_channel
        if self.samples is None or index not in self.samples.plan.channels:
            self.message = "Select the focused channel and wait for its image before auto contrast."
        elif self.samples.plan.viewport != self.viewport:
            self.message = "Wait for the current viewport before auto contrast."
        else:
            from ..intensity_window import estimate_window

            position = self.samples.plan.channels.index(index)
            values = self.samples.values if self.samples.is_rgb else self.samples.values[position]
            try:
                estimated = estimate_window(values)
                gamma = 1.0 if self.windows[index] is None else self.windows[index].gamma
                self.windows[index] = replace(estimated, gamma=gamma)
            except ValueError as error:
                self.message = str(error)
            else:
                self.message = "Contrast estimated from the visible preview and locked."
                self.request_frame()
        self.application.invalidate()

    def begin_edit(self, field: str) -> None:
        window = self.windows[self.focused_channel]
        if window is None:
            self.message = "Select this channel and wait for its initial image before editing."
            self.application.invalidate()
            return
        self.editing = field
        self.entry.text = f"{getattr(window, field):g}"
        self.entry.buffer.cursor_position = len(self.entry.text)
        self.application.layout.focus(self.entry)
        self.application.invalidate()

    def _accept_edit(self, buffer: Buffer) -> bool:
        try:
            value = float(buffer.text)
            current = self.windows[self.focused_channel]
            window = replace(current, **{self.editing: value})
        except (ValueError, TypeError) as error:
            self.message = f"Invalid {self.editing}: {error}"
            self.application.invalidate()
            return True
        self.windows[self.focused_channel] = window
        self.message = f"Channel {self.focused_channel}: {self.editing} = {value:g}"
        self.cancel_edit()
        self.request_frame()
        return True

    def cancel_edit(self) -> None:
        self.editing = None
        self.application.layout.focus(self.canvas)
        self.application.invalidate()

    def _title(self) -> str:
        info = self.session.source.info
        return (f" ttwhy | {safe_label(info.path.name)} | series {info.series} "
                f"| {'RGB' if info.is_rgb else str(len(info.channels)) + ' channels'}")

    def _status(self) -> str:
        if self.viewport is None:
            return " Opening image..."
        view = self.viewport
        level = "?" if self.samples is None else str(self.samples.plan.level.index)
        size = self.session.source.info.pixel_size
        calibration = "uncalibrated" if size is None else f"{size[0]:g} x {size[1]:g} {size[2]}/px"
        return (f" {'loading' if self.busy else 'ready'} | L{level} | {self.renderer.mode} "
                f"| x={view.x + view.width / 2:,.0f} y={view.y + view.height / 2:,.0f} "
                f"| {view.width / max(1, self.canvas_size[0]):.2f} px/col | {calibration}")

    def _footer(self) -> str:
        if self.show_help:
            return (" arrows: pan | wheel/+/-: zoom | drag: pan | r: fit | c: panel | Tab: focus\n"
                    " 1..9: solo channel | ,/.: focus | space: toggle | l/u/g: limits "
                    "| a: auto | t: color | m: mode | i: density | q: quit")
        return (" q quit | +/- zoom | arrows pan | c channels | l/u limits | a auto | ? help\n "
                + safe_label(self.message))

    def __repr__(self) -> str:
        return (f"Viewer(channels={self.selected_channels}, mode={self.renderer.mode!r}, "
                f"canvas={self.canvas_size})")
