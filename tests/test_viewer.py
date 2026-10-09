"""Drive the actual full-screen application with decoded keyboard and mouse input."""

import asyncio
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from omeify import OMETiffReader
from ttwhy import ViewSession
from ttwhy.viewer import Viewer


class SizedOutput(DummyOutput):
    def __init__(self):
        self.size = Size(rows=30, columns=100)

    def get_size(self):
        return self.size


async def eventually(predicate, task, *, timeout=5.0):
    start = time.monotonic()
    while not predicate():
        if task.done():
            await task
            raise AssertionError("Application exited before the expected state")
        if time.monotonic() - start > timeout:
            raise AssertionError("Timed out waiting for viewer state")
        await asyncio.sleep(.01)


async def ready(viewer, task):
    await eventually(lambda: viewer.frame is not None and not viewer.busy
                     and viewer.displayed_generation == viewer.generation, task)
    await asyncio.sleep(.04)


@asynccontextmanager
async def running(path, **options):
    with (OMETiffReader(path) as source, ViewSession(source) as session,
          create_pipe_input() as pipe):
        viewer = Viewer(session, **options)
        output = SizedOutput()
        task = asyncio.create_task(viewer.run(input=pipe, output=output))
        try:
            await ready(viewer, task)
            yield viewer, pipe, output, task
        finally:
            if not task.done():
                pipe.send_bytes(b"\x11")  # Ctrl-Q is always available.
            await asyncio.wait_for(task, 5)
        assert source.is_open  # Viewer borrows, rather than closes, its source.
        assert viewer.session.cache_bytes == 0


async def send(pipe, text, viewer, task):
    pipe.send_text(text)
    await asyncio.sleep(.09)
    await ready(viewer, task)


async def test_keyboard_zoom_pan_panel_and_modes(mif):
    async with running(mif[0]) as (viewer, pipe, output, task):
        initial = viewer.viewport
        assert viewer.frame.shape == (26, 68)
        assert np.isfinite(viewer.samples.values).any()
        await send(pipe, "+", viewer, task)
        assert viewer.viewport.width < initial.width
        old_x = viewer.viewport.x
        await send(pipe, "\x1b[C", viewer, task)
        assert viewer.viewport.x > old_x
        await send(pipe, "c", viewer, task)
        assert not viewer.panel_visible and viewer.canvas_size[0] == 100
        await send(pipe, "c", viewer, task)
        assert viewer.panel_visible and viewer.canvas_size[0] == 68
        assert viewer.renderer.mode == "detail"
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "shape" and viewer.frame.background is None
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "simple" and viewer.frame.background is None
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "blocks" and "▀" in viewer.frame.to_text()
        await send(pipe, "t", viewer, task)
        assert viewer.renderer.mode == "simple" and not viewer.renderer.color
        await send(pipe, "?r", viewer, task)
        assert viewer.show_help
        assert "Fit slide" in viewer.message
        output.size = Size(rows=19, columns=50)
        viewer.application._on_resize()
        await asyncio.sleep(.1)
        await ready(viewer, task)
        assert not viewer.panel_visible
        assert viewer.frame.shape == (15, 50)


async def test_channels_multiselect_and_locked_windows(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        first_window = viewer.windows[0]
        await send(pipe, "+\x1b[C", viewer, task)
        assert viewer.windows[0] == first_window
        await send(pipe, "2", viewer, task)
        assert viewer.selected_channels == (1,) and viewer.focused_channel == 1
        await send(pipe, ". ", viewer, task)
        assert viewer.selected_channels == (1, 2)
        assert viewer.samples.values.shape[0] == 2
        await send(pipe, " ", viewer, task)
        assert viewer.selected_channels == (1,)
        await send(pipe, ", ", viewer, task)
        assert viewer.selected_channels == (1,) and "at least one" in viewer.message
        await send(pipe, "\t\x1b[A", viewer, task)
        assert viewer.focused_channel == 0
        await send(pipe, "\t", viewer, task)
        assert viewer.application.layout.has_focus(viewer.canvas)


async def test_numeric_limits_gamma_cancel_and_invalid_entry(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        await send(pipe, "u\x01\x0b25000\r", viewer, task)
        assert viewer.windows[0].upper == 25000
        await send(pipe, "l\x01\x0b100\r", viewer, task)
        assert viewer.windows[0].lower == 100
        await send(pipe, "g\x01\x0b2\r", viewer, task)
        assert viewer.windows[0].gamma == 2
        await send(pipe, "u\x01\x0b0\r", viewer, task)
        assert viewer.editing == "upper" and "Invalid upper" in viewer.message
        assert viewer.windows[0].upper == 25000
        # Letters belong to input while editing, not to global viewer bindings.
        await send(pipe, "q", viewer, task)
        assert not task.done() and "q" in viewer.entry.text
        await send(pipe, "\x03", viewer, task)  # Ctrl-C cancels only the editor.
        assert viewer.editing is None and not task.done()
        await send(pipe, "a", viewer, task)
        assert viewer.windows[0].upper != 25000
        assert viewer.windows[0].gamma == 2
        assert "estimated" in viewer.message


async def test_rgb_initial_limits_and_rendering(rgb):
    async with running(rgb[0]) as (viewer, pipe, _, task):
        assert viewer.samples.is_rgb
        assert viewer.selected_channels == (0,)
        assert viewer.windows[0].lower == 0 and viewer.windows[0].upper == 255
        assert viewer.renderer.invert
        assert viewer.frame.foreground is not None
        await send(pipe, "b", viewer, task)
        assert viewer.renderer.mode == "blocks" and viewer.frame.background is not None
        await send(pipe, "b", viewer, task)
        assert viewer.renderer.mode == "detail" and viewer.frame.background is None
        await send(pipe, "u\x01\x0b200\r", viewer, task)
        assert viewer.windows[0].upper == 200


async def test_sgr_mouse_wheel_drag_and_channel_click(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        before = viewer.viewport
        # SGR coordinates are 1-based terminal coordinates, including the title.
        await send(pipe, "\x1b[<64;20;10M", viewer, task)
        assert viewer.viewport.width < before.width
        x = (19 + .5) / viewer.canvas_size[0]
        y = (8 + .5) / viewer.canvas_size[1]
        assert viewer.viewport.x + viewer.viewport.width * x == pytest.approx(
            before.x + before.width * x)
        assert viewer.viewport.y + viewer.viewport.height * y == pytest.approx(
            before.y + before.height * y)
        before_drag = viewer.viewport
        await send(pipe, "\x1b[<0;20;10M\x1b[<32;24;12M\x1b[<0;24;12m", viewer, task)
        assert viewer.viewport.x < before_drag.x
        # Sidebar: terminal x=75, row=5 targets channel 2.
        await send(pipe, "\x1b[<0;75;5M\x1b[<0;75;5m", viewer, task)
        assert viewer.focused_channel == 2
        assert viewer.selected_channels == (0, 2)
        await send(pipe, "\x1b[<65;75;5M", viewer, task)
        assert viewer.focused_channel == 2


async def test_latest_request_wins_and_pending_reads_are_coalesced(mif, monkeypatch):
    with (OMETiffReader(mif[0]) as source, ViewSession(source) as session,
          create_pipe_input() as pipe):
        started, release = threading.Event(), threading.Event()
        original = session.sample
        calls = []
        def slow(plan):
            calls.append(plan)
            if len(calls) == 1:
                started.set()
                assert release.wait(3)
            return original(plan)
        monkeypatch.setattr(session, "sample", slow)
        viewer = Viewer(session)
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            await eventually(started.is_set, task)
            pipe.send_text("++++++++")
            await eventually(lambda: viewer.generation >= 9, task)
            assert viewer.frame is None
            release.set()
            await ready(viewer, task)
            assert viewer.samples.plan.viewport == viewer.viewport
            assert len(calls) == 2  # initial in-flight + newest pending, not nine decodes
        finally:
            release.set()
            pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 5)


async def test_worker_failure_propagates_and_releases_resources(mif, monkeypatch):
    with (OMETiffReader(mif[0]) as source, ViewSession(source) as session,
          create_pipe_input() as pipe):
        def broken(*args, **kwargs):
            raise OSError("intentional decode failure")
        monkeypatch.setattr(source, "read_region", broken)
        viewer = Viewer(session)
        with pytest.raises(OSError, match="intentional decode failure"):
            await asyncio.wait_for(viewer.run(input=pipe, output=SizedOutput()), 5)
        assert viewer._executor is None
        assert source.is_open
        assert viewer.session.cache_bytes == 0


async def test_quit_waits_for_inflight_decode(mif, monkeypatch):
    with (OMETiffReader(mif[0]) as source, ViewSession(source) as session,
          create_pipe_input() as pipe):
        started, finished = threading.Event(), threading.Event()
        original = source.read_region
        def slow(*args, **kwargs):
            started.set()
            time.sleep(.15)
            try:
                assert source.is_open
                return original(*args, **kwargs)
            finally:
                finished.set()
        monkeypatch.setattr(source, "read_region", slow)
        viewer = Viewer(session)
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        await eventually(started.is_set, task)
        pipe.send_text("q")
        await asyncio.wait_for(task, 5)
        assert finished.is_set() and source.is_open


async def test_new_mode_cycle_keeps_windows_and_reuses_high_resolution_preview(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        assert viewer.renderer.mode == "detail"
        windows = tuple(viewer.windows)
        samples = viewer.samples
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "shape"
        assert viewer.samples is samples
        assert tuple(viewer.windows) == windows
        assert viewer.frame.background is None
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "simple"
        assert viewer.samples.plan.sample_shape == (
            viewer.canvas_size[1], viewer.canvas_size[0])
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "blocks"
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "detail"
        assert tuple(viewer.windows) == windows


async def test_monochrome_mode_cycle_and_blocks_roundtrip(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        await send(pipe, "t", viewer, task)
        assert not viewer.renderer.color and viewer.renderer.mode == "detail"
        for expected in ("shape", "simple", "detail"):
            await send(pipe, "m", viewer, task)
            assert viewer.renderer.mode == expected and not viewer.renderer.color
            assert viewer.frame.foreground is viewer.frame.background is None
        await send(pipe, "b", viewer, task)
        assert viewer.renderer.mode == "blocks" and viewer.renderer.color
        await send(pipe, "b", viewer, task)
        assert viewer.renderer.mode == "detail" and not viewer.renderer.color


async def test_custom_simple_ramp_survives_other_render_modes(mif):
    from ttwhy import TextRenderer

    renderer = TextRenderer(mode="simple", ramp=" .#", color=False)
    async with running(mif[0], renderer=renderer) as (viewer, pipe, _, task):
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == "detail"
        await send(pipe, "mm", viewer, task)
        assert viewer.renderer.mode == "simple" and viewer.renderer.ramp == " .#"
