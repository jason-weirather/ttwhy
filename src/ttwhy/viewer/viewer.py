"""A full-screen microscopy viewer with one coalescing render worker."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.input import Input
from prompt_toolkit.layout import (
    ConditionalContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from ..intensity_window import IntensityWindow
from ..view_info import ReadBudgetExceeded
from ..text_image import TextImage
from ..terminal_capabilities import configured_color_depth
from ..text_renderer import DEFAULT_RAMP, TEXT_MODES, TextRenderer
from ..view_session import ViewSamples, ViewSession
from ..viewport import Viewport
from .bindings import create_bindings
from .clipping import ClipController
from .mode_navigation import ModeNavigation
from .controls import CanvasControl, ChannelControl, clip_fragments, navigation_fragments, safe_label
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
        navigation=None,
        source_path=None, output_text_clip=None, output_ansi_clip=None,
        output_png_clip=None, output_geojson_clip=None,
        clipboard="auto", clip_source="auto", clip_overwrite=True,
    ) -> None:
        self.session = session
        self.navigation = navigation
        self.selected_channels = session._select_channels(channels)
        self.focused_channel = self.selected_channels[0]
        self.windows = [window] * len(session.info.channels)
        self.renderer = renderer or TextRenderer(invert=session.info.is_rgb)
        self.text_mode = (self.renderer.mode if self.renderer.mode in TEXT_MODES else "detail")
        self.text_color = self.renderer.color
        self.simple_ramp = self.renderer.ramp
        self.mode_navigation = ModeNavigation.from_renderer(self.renderer)
        self.viewport = viewport
        self.frame: TextImage | None = None
        self.samples: ViewSamples | None = None
        self.cell_lines: list[list[tuple[str, str]]] = []
        self.canvas_size = (0, 0)
        self.panel_requested = True
        self.image_only = False
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
        self.clipping = ClipController(
            self, source_path=source_path, output_text_clip=output_text_clip,
            output_ansi_clip=output_ansi_clip, output_png_clip=output_png_clip,
            output_geojson_clip=output_geojson_clip, clipboard=clipboard,
            clip_source=clip_source, clip_overwrite=clip_overwrite,
        )
        self.entry = TextArea(
            multiline=False, height=1, accept_handler=self._accept_edit,
            prompt=self._entry_prompt, style="class:entry",
        )

    @property
    def selecting(self) -> bool:
        return self.clipping.selection is not None

    def _entry_prompt(self) -> str:
        if self.editing in ("save-text", "save-ansi", "save-png", "save-geojson"):
            return " Save (Enter/Esc): "
        if self.editing == "geojson-name":
            return " Name: "
        if self.editing == "destination" and self.navigation is not None:
            doc = self.navigation.document
            return f" Go to {doc.label} ({doc.first_index}..{doc.last_index}; Enter / Esc): "
        units = "unitless" if self.editing == "gamma" else "raw units"
        return f" {self.editing or ''} ({units}; Enter / Esc): "

    @property
    def panel_visible(self) -> bool:
        return bool(not self.image_only and self.panel_requested and self.application is not None
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
            ConditionalContainer(
                Window(FormattedTextControl(self._title), height=1, style="class:title"),
                filter=Condition(lambda: not self.image_only),
            ),
            ConditionalContainer(
                Window(FormattedTextControl(lambda: navigation_fragments(self)),
                       height=1, style="class:status"),
                filter=Condition(lambda: self.navigation is not None and not self.image_only),
            ),
            body,
            ConditionalContainer(
                Window(FormattedTextControl(self._status), height=1, style="class:status"),
                filter=Condition(lambda: not self.image_only),
            ),
            ConditionalContainer(
                Window(FormattedTextControl(self._footer),
                       height=lambda: 1 if self.editing is not None else 2, style="class:footer"),
                filter=Condition(lambda: not self.image_only),
            ),
            ConditionalContainer(self.entry, filter=Condition(
                lambda: self.editing is not None and not self.image_only)),
        ])
        # In image-only mode, entry overlays one row instead of resizing the
        # frozen canvas and throwing away the user's selections.
        container = FloatContainer(container, floats=[Float(
            left=0, right=0, bottom=0, height=1,
            content=ConditionalContainer(self.entry, filter=Condition(
                lambda: self.image_only and self.editing is not None)),
        )])
        return Application(
            layout=Layout(container, focused_element=self.canvas),
            key_bindings=create_bindings(self), full_screen=True, mouse_support=True,
            input=input, output=output, min_redraw_interval=1 / 30,
            color_depth=configured_color_depth(),
            style=Style.from_dict({
                "image": "bg:#000000 #eeeeee", "title": "bg:#253247 #ffffff bold",
                "panel": "bg:#141c29 #bdc8da", "panel.title": "#83d5c8 bold",
                "panel.selected": "bg:#31415b #ffffff bold", "divider": "#405570",
                "status": "bg:#253247 #d7e1f0", "footer": "bg:#141c29 #a8b7cc",
                "entry": "bg:#394a63 #ffffff",
                "clip.border": "bg:#ffd75f #000000 bold",
                "clip.cursor": "bg:#ffffff #000000 bold reverse",
                "clip.button": "bg:#31415b #ffffff bold",
                "clip.saved": "bg:#83d5c8 #000000",
            }),
        )

    async def run(self, *, input: Input | None = None, output: Output | None = None) -> None:
        """Run until quit. Optional terminal I/O supports embedding and real input tests."""
        if self.application is not None:
            raise RuntimeError("Create a new Viewer for each terminal session")
        self._wake = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ttwhy-render")
        self.application = self._build_application(input, output)
        if self.navigation is not None:
            self.navigation.start(self)
        try:
            await self.application.run_async(
                pre_run=lambda: self.application.create_background_task(self._pump())
            )
        finally:
            # Worker cancellation does not interrupt a TIFF codec. Join before
            # allowing the caller to leave its source context manager.
            try:
                if self.navigation is not None:
                    await self.navigation.stop()
            finally:
                self.clipping.close()
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
                self._pending = None
                if self.navigation is not None:
                    self.navigation.close()
                else:
                    self.session.clear_cache()

    async def _pump(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            if self.selecting:
                continue  # Keep displayed cells and PDF coordinates stable during selection.
            if self.navigation is not None:
                self.navigation.install_pending()
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
            if request.generation != self.generation or self.selecting:
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
            if self.navigation is not None:
                self.navigation.frame_displayed()

    def request_frame(self) -> None:
        if self.selecting or self.viewport is None or not all(self.canvas_size) or self._wake is None:
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
        if self.selecting:
            self.clipping.end(message="Selection cleared after resize; select the new frame.",
                              resume=False)
        previous = self.canvas_size
        self.canvas_size = (columns, rows)
        if self.viewport is None:
            self.fit()
        elif all(previous):
            old = self.viewport
            # Preserve the center and horizontal base pixels per terminal cell.
            width = old.width * columns / previous[0]
            height = width / columns * rows * self.renderer.cell_aspect
            height /= self.session.info.pixel_aspect
            candidate = Viewport(old.x + (old.width - width) / 2,
                                 old.y + (old.height - height) / 2, width, height)
            self._set_viewport(candidate, limit=True)
        else:
            self._set_viewport(self.viewport, limit=True)

    def _set_viewport(self, viewport: Viewport, *, limit: bool = False) -> None:
        if self.selecting:
            return
        rows, columns = self.renderer.sample_shape(columns=self.canvas_size[0],
                                                   rows=self.canvas_size[1])
        viewport = viewport.centered_within(self.session.info.shape)
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
        if self.selecting:
            return
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
        if self.selecting:
            return
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
        if self.selecting:
            return
        if self.viewport is not None:
            self.message = ""
            self._set_viewport(self.viewport.pan(dx * self.viewport.width / self.canvas_size[0],
                                                dy * self.viewport.height / self.canvas_size[1]))

    def focus_channel(self, delta: int) -> None:
        if self.selecting:
            return
        self.focused_channel = min(len(self.windows) - 1, max(0, self.focused_channel + delta))
        self.application.invalidate()

    def select_one(self, index: int) -> None:
        if self.selecting:
            return
        if index < len(self.windows):
            self.focused_channel = index
            self.selected_channels = (index,)
            self._set_viewport(self.viewport, limit=True)

    def toggle_channel(self) -> None:
        if self.selecting:
            return
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

    def change_renderer(self, *, preserve_family: bool = False, **changes) -> None:
        if self.selecting:
            return
        if "invert" in changes and self.navigation is not None:
            self.navigation.auto_invert = False
        options = {"mode": self.renderer.mode, "color": self.renderer.color,
                   "invert": self.renderer.invert, "ramp": self.renderer.ramp,
                   "cell_aspect": self.renderer.cell_aspect}
        options.update(changes)
        if "ramp" not in changes:
            options["ramp"] = (self.simple_ramp if options["mode"] in ("simple", "ascii")
                               else DEFAULT_RAMP)
        self.renderer = TextRenderer(**options)
        if not preserve_family:
            self.mode_navigation.observe(self.renderer)
            if self.renderer.mode == "simple":
                self.simple_ramp = self.renderer.ramp
        if self.renderer.mode in TEXT_MODES:
            self.text_mode = self.renderer.mode
            self.text_color = self.renderer.color
        self.message = {
            "detail": "Detail: spatial Braille, tone-balanced dots, no background tiles.",
            "shape": "Shape: artistic strokes and texture, not a segmentation.",
            "simple": "Simple: original ASCII density renderer.",
            "blocks": "Blocks: two-color half-block preview.",
        }[self.renderer.mode]
        self._set_viewport(self.viewport, limit=True)

    def cycle_mode(self) -> None:
        """m: resume colored artwork, then cycle its four existing algorithms."""
        if self.selecting or self.editing is not None:
            return
        self.mode_navigation.advance_color()
        next_renderer = self.mode_navigation.make_renderer(self.renderer)
        self.change_renderer(preserve_family=True, mode=next_renderer.mode,
                             color=next_renderer.color, ramp=next_renderer.ramp)
        self.message = f"Color {self.mode_navigation.label} | m: next | M: ASCII"
        self.application.invalidate()

    def cycle_ascii(self) -> None:
        """M: enter strict ASCII, then cycle classic/fine/bold density vocabularies."""
        if self.selecting or self.editing is not None:
            return
        self.mode_navigation.advance_ascii()
        next_renderer = self.mode_navigation.make_renderer(self.renderer)
        self.change_renderer(preserve_family=True, mode=next_renderer.mode,
                             color=False, ramp=next_renderer.ramp)
        self.message = f"{self.mode_navigation.label} | M: next | m: color"
        self.application.invalidate()

    def toggle_blocks(self) -> None:
        """Temporarily show the original half-block preview; remember text style."""
        if self.renderer.mode == "blocks":
            self.change_renderer(mode=self.text_mode, color=self.text_color)
        else:
            self.change_renderer(mode="blocks", color=True)

    def auto_window(self) -> None:
        if self.selecting:
            return
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

    def toggle_fullscreen(self, *, enable: bool | None = None) -> None:
        if self.selecting or self.editing is not None:
            return
        target = (not self.image_only) if enable is None else bool(enable)
        if target == self.image_only:
            return
        self.image_only = target
        self.message = ("Full-terminal image mode. Press f or Esc to return."
                        if self.image_only else "Viewer controls restored.")
        if self.application is not None:
            self.application.layout.focus(self.canvas)
            self.application.invalidate()

    def begin_edit(self, field: str) -> None:
        if self.selecting:
            return
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

    def begin_destination(self) -> None:
        if self.selecting:
            return
        if self.navigation is None:
            self.message = "Page/series navigation is available for PDF and OME-TIFF files."
            self.application.invalidate()
            return
        self.editing = "destination"
        self.entry.text = ""
        self.entry.buffer.cursor_position = len(self.entry.text)
        self.application.layout.focus(self.entry)
        self.application.invalidate()

    def step_destination(self, delta: int) -> None:
        if self.selecting:
            return
        if self.navigation is not None:
            self.navigation.step(delta)
        else:
            self.message = "This view has no page/series navigator."
            self.application.invalidate()

    def _accept_edit(self, buffer: Buffer) -> bool:
        if self.editing in ("save-text", "save-ansi", "save-png", "save-geojson"):
            self.clipping.accept_path(buffer.text, ansi=self.editing == "save-ansi",
                                      image=self.editing == "save-png",
                                      geojson=self.editing == "save-geojson")
            return True
        if self.editing == "geojson-name":
            self.clipping.accept_name(buffer.text)
            return True
        if self.editing == "destination":
            try:
                destination = self.navigation.document.validate_index(int(buffer.text.strip()))
            except (ValueError, TypeError) as error:
                self.message = f"Invalid {self.navigation.document.label}: {error}"
                self.application.invalidate()
                return True
            self.cancel_edit()
            self.navigation.go(destination)
            return True
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
        info = self.session.info
        return (f" ttwhy | {safe_label(info.name)} "
                f"| {'RGB' if info.is_rgb else str(len(info.channels)) + ' channels'}")

    def _status(self) -> str:
        if self.selecting:
            return " " + safe_label(self.clipping.status())
        if self.viewport is None:
            return " Opening image..."
        view = self.viewport
        level = "?" if self.samples is None else str(self.samples.plan.level.index)
        if self.samples is not None and self.samples.plan.level.approximate:
            level += " extent-fit"
        size = self.session.info.pixel_size
        calibration = "uncalibrated" if size is None else f"{size[0]:g} x {size[1]:g} {size[2]}/px"
        mode_label = (self.mode_navigation.label if self.mode_navigation.family == "ascii"
                      else self.renderer.mode)
        return (f" {'loading' if self.busy else 'ready'} | L{level} | {mode_label} "
                f"| x={view.x + view.width / 2:,.0f} y={view.y + view.height / 2:,.0f} "
                f"| {view.width / max(1, self.canvas_size[0]):.2f} px/col | {calibration}")

    def _footer(self):
        buttons = clip_fragments(self)
        if self.selecting:
            return [*buttons, ("", "\n " + safe_label(self.message))]
        return [*buttons, ("", self._footer_text())]

    def _footer_text(self) -> str:
        if self.show_help:
            return (" | m: color | M: ASCII | t: toggle color | q: quit\n"
                    " Arrows: pan | +/-: zoom | r: fit | Tab: channels | l/u/g: limits | ?: less")
        return " | m:color M:ASCII ?:help q:quit\n " + safe_label(self.message)

    def __repr__(self) -> str:
        return (f"Viewer(channels={self.selected_channels}, mode={self.renderer.mode!r}, "
                f"canvas={self.canvas_size})")