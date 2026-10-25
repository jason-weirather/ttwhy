"""Channel color parsing, pixel-level composition, and actual terminal controls."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import numpy as np
import pytest
from click.testing import CliRunner
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from conftest import write_pyramid
from test_document_navigation import browsing, write_series
from test_viewer import SizedOutput, eventually, ready, running, send
from twhy import IntensityWindow, TextRenderer, ViewSession, open_image
from twhy.channel_colors import (
    apply_channel_colors, channel_palette, color_hex, parse_color,
)
from twhy.cli import main
from twhy.compositing import CHANNEL_COLORS
from twhy.viewer import Viewer, view_file
from twhy.viewer.controls import _padded_label
from twhy.viewer.rendering import FrameRequest, make_frame
from twhy.viewport import Viewport


@pytest.mark.parametrize("text,expected", [
    ("#3366ff", (51, 102, 255)), ("3366FF", (51, 102, 255)),
    ("#36f", (51, 102, 255)), ("36F", (51, 102, 255)),
    ("  CyAn  ", (0, 255, 255)), ("blue", (0, 0, 255)),
    ("green", (0, 255, 0)), ("magenta", (255, 0, 255)),
    ("red", (255, 0, 0)), ("yellow", (255, 255, 0)),
    ("orange", (255, 128, 0)), ("purple", (128, 0, 255)),
    ("violet", (128, 0, 255)), ("white", (255, 255, 255)),
    ("black", (0, 0, 0)), ("gray", (128, 128, 128)), ("grey", (128, 128, 128)),
])
def test_parse_color(text, expected):
    color = parse_color(text)
    assert color == tuple(component / 255 for component in expected)
    assert color_hex(color) == "#" + "".join(f"{component:02X}" for component in expected)


@pytest.mark.parametrize("text", ["", "#", "##36f", "#1234", "#12345678", "#gggggg",
                                  "transparent", "rgb(0,0,255)", "nan", "-1", "256,0,0",
                                  "#12 3456", "#36f\x1b[0m", None, 123])
def test_bad_color_is_an_error_not_a_fallback(text):
    with pytest.raises(ValueError, match="[Cc]olor|#[Rr]"):
        parse_color(text)


def test_defaults_remain_exact_and_overrides_follow_source_indices():
    assert parse_color(" Default ") is None
    defaults = channel_palette(10)
    assert defaults == [CHANNEL_COLORS[i % len(CHANNEL_COLORS)] for i in range(10)]
    assert channel_palette(1) == [(1., 1., 1.)]
    assert channel_palette(1, overrides={0: "blue"}) == [(0., 0., 1.)]
    assert channel_palette(1, is_rgb=True) == [(1., 1., 1.)]
    assert channel_palette(10, overrides={0: "default"}) == defaults
    result = channel_palette(10, overrides={9: "cyan", 0: "#36f"})
    assert result[0] == parse_color("#36f")
    assert result[9] == (0., 1., 1.)
    assert result[1:9] == defaults[1:9]
    assert defaults[0] == CHANNEL_COLORS[0]


@pytest.mark.parametrize("overrides", [{-1: "red"}, {3: "red"}, {True: "red"},
                                       {"0": "red"}, {0: "not-a-color"}])
def test_bad_palette_overrides_are_rejected(overrides):
    with pytest.raises(ValueError):
        channel_palette(3, overrides=overrides)


@pytest.mark.parametrize("mode", ["blocks", "detail", "shape", "simple"])
def test_default_palette_leaves_existing_rendering_unchanged(mif, mode):
    with open_image(mif[0]) as image, ViewSession(image) as session:
        request = FrameRequest(1, Viewport(0, 0, 97, 65), 24, 12, (2, 0),
                               (None, None, None), TextRenderer(mode=mode))
        old = make_frame(session, request)
        new = make_frame(session, replace(request, colors=tuple(channel_palette(3))))
        assert new.frame.to_text(ansi=True) == old.frame.to_text(ansi=True)


@asynccontextmanager
async def uncached_view(path, **kwargs):
    with (open_image(path) as image,
          ViewSession(image, cache_mib=0, overview_cache_mib=0) as session,
          create_pipe_input() as pipe):
        viewer = Viewer(session, renderer=TextRenderer(mode="blocks"), **kwargs)
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            await ready(viewer, task)
            yield viewer, pipe, task
        finally:
            if not task.done():
                pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 10)


async def test_color_edit_reuses_displayed_samples_even_without_lru(mif, monkeypatch):
    async with uncached_view(mif[0], channels=(2, 0)) as (viewer, pipe, task):
        viewer.focus_channel(-2)
        samples, camera = viewer.samples, viewer.viewport
        windows = list(viewer.windows)
        assert viewer.session.cache_bytes == 0

        def forbid(*args, **kwargs):
            raise AssertionError("Color-only edits must reuse displayed samples, not reread pixels")

        monkeypatch.setattr(viewer.session, "sample", forbid)
        await send(pipe, "h\x01\x0b#3366ff\r", viewer, task)
        assert viewer.channel_colors[0] == parse_color("#3366ff")
        assert viewer.channel_colors[2] == CHANNEL_COLORS[2]
        assert viewer.samples is samples
        assert viewer.viewport == camera
        assert viewer.windows == windows
        assert viewer.selected_channels == (2, 0)
        assert viewer.session.cache_bytes == 0
        # Independent composition checks channel order, window retention and emitted RGB.
        values = np.zeros((*samples.values.shape[1:], 3), np.float32)
        for position, index in enumerate((2, 0)):
            plane = windows[index].apply(samples.values[position])
            values += np.nan_to_num(plane, nan=0)[..., None] * viewer.channel_colors[index]
        np.clip(values, 0, 1, out=values)
        valid = np.isfinite(samples.values).any(axis=0)
        values[~valid] = np.nan
        expected = viewer.renderer.render(values, columns=viewer.canvas_size[0],
                                          rows=viewer.canvas_size[1])
        assert viewer.frame.to_text(ansi=True) == expected.to_text(ansi=True)
        # Rapid style requests retain immutable colors and coalesce correctly.
        await send(pipe, "h\x01\x0bblue\rh\x01\x0bcyan\r", viewer, task)
        assert viewer.channel_colors[0] == (0., 1., 1.)
        assert viewer.samples is samples


async def test_entry_validation_cancel_and_exact_default(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        original = viewer.channel_colors[0]
        generation = viewer.generation
        await send(pipe, "h\r", viewer, task)
        assert viewer.channel_colors[0] == original  # No implicit 8-bit quantization.
        assert viewer.generation == generation
        await send(pipe, "h\x01\x0bnot-a-color\r", viewer, task)
        assert viewer.editing == "color" and "Invalid color" in viewer.message
        assert viewer.channel_colors[0] == original
        await send(pipe, "q", viewer, task)
        assert not task.done() and "q" in viewer.entry.text
        await send(pipe, "\x03", viewer, task)
        assert viewer.editing is None and not task.done()
        await send(pipe, "h\x01\x0b#36f\x1b", viewer, task)
        await eventually(lambda: viewer.editing is None, task)
        assert viewer.channel_colors[0] == original
        await send(pipe, "h\x01\x0bcyan\r", viewer, task)
        await send(pipe, "h\x01\x0bdefault\r", viewer, task)
        assert viewer.channel_colors[0] == original


async def test_color_without_auto_window_does_not_select_or_read_channel(mif, monkeypatch):
    async with uncached_view(mif[0]) as (viewer, pipe, task):
        assert viewer.windows[1] is None
        viewer.focus_channel(1)
        generation = viewer.generation
        frame = viewer.frame
        monkeypatch.setattr(viewer.session, "sample", lambda *a: pytest.fail("Unexpected read"))
        await send(pipe, "h\x01\x0bmagenta\r", viewer, task)
        assert viewer.channel_colors[1] == (1., 0., 1.)
        assert viewer.windows[1] is None and viewer.selected_channels == (0,)
        assert viewer.generation == generation and viewer.frame is frame


async def test_rgb_color_knob_is_explicitly_inapplicable(rgb):
    async with running(rgb[0]) as (viewer, pipe, _, task):
        frame, generation = viewer.frame, viewer.generation
        await send(pipe, "h", viewer, task)
        assert viewer.editing is None and "native colors" in viewer.message
        assert viewer.frame is frame and viewer.generation == generation
        assert viewer.channel_colors == [(1., 1., 1.)]


async def test_color_survives_other_controls_and_fullscreen(mif):
    async with running(mif[0], channel_colors={0: "#36f", 1: "cyan"}) as (viewer, pipe, _, task):
        colors = list(viewer.channel_colors)
        await send(pipe, "u\x01\x0b30000\rg\x01\x0b2\ra", viewer, task)
        assert viewer.channel_colors == colors
        await send(pipe, "2", viewer, task)
        await send(pipe, "1", viewer, task)
        await send(pipe, "+Mmh\x01\x0bcyan\r", viewer, task)
        assert viewer.channel_colors[0] == (0., 1., 1.)
        await send(pipe, "f", viewer, task)
        camera = viewer.viewport
        size = viewer.canvas_size
        await send(pipe, "h\x01\x0b#36f\r", viewer, task)
        assert viewer.image_only and viewer.viewport == camera and viewer.canvas_size == size
        assert viewer.channel_colors == colors


async def test_monochrome_does_not_force_color_on(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        await send(pipe, "M", viewer, task)
        await send(pipe, "h\x01\x0bcyan\r", viewer, task)
        assert not viewer.renderer.color
        assert "color display is off" in viewer.message
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.color and viewer.channel_colors[0] == (0., 1., 1.)


async def test_mouse_swatches_and_color_row_edit_without_toggling(mif):
    async with running(mif[0]) as (viewer, pipe, _, task):
        control = viewer.channel_control
        control.create_content(31, 26)
        def click(x, y):
            control.mouse_handler(MouseEvent(Point(x, y), MouseEventType.MOUSE_UP,
                                             MouseButton.LEFT, frozenset()))
        click(28, control.channel_row_start + 1)
        assert viewer.editing == "color" and viewer.focused_channel == 1
        assert viewer.selected_channels == (0,)
        await send(pipe, "\x01\x0bcyan\r", viewer, task)
        content = control.create_content(31, 26)
        swatch = content.get_line(control.channel_row_start + 1)[1]
        assert swatch == ("bg:#00FFFF", "  ")
        click(11, control.color_row)
        assert viewer.editing == "color"
        await send(pipe, "\x03", viewer, task)
        click(5, control.channel_row_start + 1)
        await ready(viewer, task)
        assert viewer.selected_channels == (0, 1)


async def test_scrolled_swatch_edits_absolute_channel_index(tmp_path):
    path = tmp_path / "many.ome.tif"
    write_pyramid(path, np.full((128, 17, 23), 500, np.uint16), levels=1)
    async with running(path) as (viewer, pipe, _, task):
        viewer.focus_channel(127)
        control = viewer.channel_control
        content = control.create_content(31, 10)
        row = control.channel_row_start + 127 - control.first_channel
        assert control.first_channel > 0 and 0 <= row < content.line_count
        control.mouse_handler(MouseEvent(Point(28, row), MouseEventType.MOUSE_UP,
                                         MouseButton.LEFT, frozenset()))
        await send(pipe, "\x01\x0bcyan\r", viewer, task)
        assert viewer.channel_colors[127] == (0., 1., 1.)
        assert viewer.channel_colors[0] == CHANNEL_COLORS[0]
        assert viewer.selected_channels == (0,)


def test_long_unicode_labels_do_not_move_swatch():
    from wcwidth import wcswidth
    for text in ("Very long marker " * 5, "細胞 DAPI " * 8, "e\u0301 " * 30):
        assert wcswidth(_padded_label(text, 28)) == 28


async def test_selection_style_cycles_keep_custom_colors(mif, monkeypatch):
    # Naming is independent of this rendering test; deterministic label fixture.
    monkeypatch.setattr("twhy.annotation_identity.random_name", lambda: "quiet-blue-otter")
    async with running(mif[0], channel_colors={0: "cyan"}) as (viewer, pipe, _, task):
        await send(pipe, "s", viewer, task)
        assert viewer.selecting
        colors = list(viewer.channel_colors)
        await send(pipe, "h", viewer, task)
        assert viewer.editing is None  # Selections stay frozen.
        monkeypatch.setattr(viewer.session, "sample", lambda *a: pytest.fail("Unexpected read"))
        await send(pipe, "Mmm", viewer, task)
        assert viewer.channel_colors == colors
        renderer = viewer.clipping.preview_modes.make_renderer(viewer.renderer)
        styles = viewer.session.prepare_styles(viewer.samples, windows=[viewer.windows[0]])
        styles = apply_channel_colors(styles, (0,), colors)
        expected = viewer.session.render(viewer.samples, styles=styles, renderer=renderer,
                                         columns=viewer.canvas_size[0], rows=viewer.canvas_size[1])
        assert viewer.clipping.selection.frame.to_text(ansi=True) == expected.to_text(ansi=True)


async def test_series_changes_reset_colors_and_pending_edit_is_safe(tmp_path):
    path = tmp_path / "series.ome.tif"
    write_series(path)
    async with browsing(path, series=1) as (nav, viewer, pipe, _, task):
        await send(pipe, "h\x01\x0bcyan\r", viewer, task)
        assert viewer.channel_colors[0] == (0., 1., 1.)
        await send(pipe, "h\x01\x0bred", viewer, task)
        nav.go(2)  # Simulate a source load already underway while entry is open.
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        await send(pipe, "\r", viewer, task)
        assert viewer.editing is None and "Image changed" in viewer.message
        assert viewer.channel_colors == [(1., 1., 1.)]
        await send(pipe, "[", viewer, task)
        await eventually(lambda: nav.active.index == 1 and not viewer.busy, task)
        assert viewer.channel_colors == channel_palette(2)


@pytest.mark.parametrize("source_kind", ["ome", "png"])
async def test_file_frontend_passes_palette_to_viewer(tmp_path, mif, monkeypatch, source_kind):
    if source_kind == "png":
        from PIL import Image
        path = tmp_path / "scalar.png"
        Image.fromarray(np.full((16, 24), 500, np.uint16)).save(path)
    else:
        path = mif[0]
    observed = []
    async def fake_run(self, **kwargs):
        observed.append(tuple(self.channel_colors))
    monkeypatch.setattr(Viewer, "run", fake_run)
    await view_file(path, channel_colors={0: "#36f"})
    assert observed[0][0] == parse_color("#36f")


def test_cli_snapshot_color_and_default(tmp_path):
    path = tmp_path / "constant.ome.tif"
    write_pyramid(path, np.full((16, 16), 255, np.uint8), levels=1)
    args = [str(path), "--snapshot", "--ansi", "--columns", "8", "--rows", "4",
            "--lower", "0", "--upper", "255"]
    runner = CliRunner()
    result = runner.invoke(main, [*args, "--channel-color", "0", "#36f"])
    assert result.exit_code == 0, result.output
    assert "38;2;51;102;255" in result.stdout and "48;2;51;102;255" in result.stdout
    original = runner.invoke(main, args)
    restored = runner.invoke(main, [*args, "--channel-color", "0", "default"])
    assert restored.exit_code == 0
    assert original.stdout == restored.stdout


def test_cli_multiple_colors_map_by_index_not_selection_order(mif, monkeypatch):
    seen = []
    original = ViewSession.render
    def capture(self, samples, *, styles, **kwargs):
        seen.append((samples.plan.channels, tuple(style.color for style in styles)))
        return original(self, samples, styles=styles, **kwargs)
    monkeypatch.setattr(ViewSession, "render", capture)
    result = CliRunner().invoke(main, [str(mif[0]), "--snapshot", "--ansi",
        "--channel", "2", "--channel", "0", "--channel-color", "0", "#36f",
        "--channel-color", "2", "cyan"])
    assert result.exit_code == 0, result.output
    assert seen == [((2, 0), (parse_color("cyan"), parse_color("#36f")))]


@pytest.mark.parametrize("flags,error", [
    (["--channel-color", "0", "nope"], "Use #RGB"),
    (["--channel-color", "0", "blue", "--channel-color", "0", "red"], "more than one"),
    (["--channel-color", "-1", "red"], "range"),
    (["--channel-color", "3", "red"], "outside"),
])
def test_cli_bad_colors_fail_without_reading_pixels(mif, monkeypatch, flags, error):
    monkeypatch.setattr(ViewSession, "sample", lambda *a: pytest.fail("Unexpected read"))
    result = CliRunner().invoke(main, [str(mif[0]), "--snapshot", *flags])
    assert result.exit_code != 0 and error in result.output


def test_rgb_cli_rejects_pseudocolor_before_pixels(rgb, monkeypatch):
    monkeypatch.setattr(ViewSession, "sample", lambda *a: pytest.fail("Unexpected read"))
    result = CliRunner().invoke(main, [str(rgb[0]), "--snapshot", "--channel-color", "0", "red"])
    assert result.exit_code != 0 and "not RGB" in result.output


def test_cli_interactive_passes_overrides(mif, monkeypatch):
    import sys
    import twhy.viewer.file_viewer as frontend
    seen = []
    async def fake_view(path, **kwargs):
        seen.append(kwargs["channel_colors"])
    monkeypatch.setattr(frontend, "view_file", fake_view)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    # CliRunner replaces stdin/stdout itself; set tty indicators inside isolation.
    runner = CliRunner()
    with runner.isolation():
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        main.main([str(mif[0]), "--channel-color", "0", "cyan"], standalone_mode=False)
    assert seen == [{0: "cyan"}]


def test_color_only_request_rejects_samples_from_another_view(mif, monkeypatch):
    """A pending camera change must not borrow stale pixels for its newer color."""
    with open_image(mif[0]) as image, ViewSession(image, cache_mib=0) as session:
        request = FrameRequest(1, Viewport(0, 0, 97, 65), 24, 12, (0,),
                               (None, None, None), TextRenderer(mode="blocks"))
        old = make_frame(session, request)
        calls = []
        original = session.sample
        def sample(plan):
            calls.append(plan.viewport)
            return original(plan)
        monkeypatch.setattr(session, "sample", sample)
        target = request.viewport.zoom(2)
        result = make_frame(session, replace(request, generation=2, viewport=target,
                            colors=tuple(channel_palette(3, overrides={0: "cyan"})),
                            cached_samples=old.samples))
        assert calls == [target] and result.samples.plan.viewport == target
        assert result.samples is not old.samples


async def test_selected_png_and_ansi_keep_edited_color(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr("twhy.annotation_identity.random_name", lambda: "quiet-blue-otter")
    path, png, ansi = (tmp_path / name for name in ("signal.ome.tif", "clip.png", "clip.ansi"))
    write_pyramid(path, np.full((16, 32), 255, np.uint8), levels=1)
    async with running(path, renderer=TextRenderer(mode="blocks"),
                       window=IntensityWindow(0, 255), clipboard="off",
                       output_png_clip=png, output_ansi_clip=ansi) as (viewer, pipe, _, task):
        await send(pipe, "h\x01\x0b#36f\r", viewer, task)
        await send(pipe, "s ", viewer, task)
        assert viewer.selecting
        await send(pipe, "\x1b[C\x1b[B", viewer, task)
        await send(pipe, "I", viewer, task)
        await eventually(lambda: png.exists() and not viewer.clipping.busy, task)
        await send(pipe, "T", viewer, task)
        await eventually(lambda: ansi.exists() and not viewer.clipping.busy, task)
        with Image.open(png) as picture:
            pixels = np.array(picture)
            opaque = pixels[..., 3] == 255
            assert opaque.any()
            np.testing.assert_array_equal(pixels[..., :3][opaque],
                                          np.broadcast_to([51, 102, 255], (opaque.sum(), 3)))
        assert "38;2;51;102;255" in ansi.read_text()


def test_real_terminal_color_entry_and_restoration(tmp_path):
    import io
    import os
    import sys
    if sys.platform == "win32":
        pytest.skip("POSIX pseudo-terminal test")
    pexpect = pytest.importorskip("pexpect")
    path = tmp_path / "constant.ome.tif"
    write_pyramid(path, np.full((24, 32), 255, np.uint8), levels=1)
    transcript = io.StringIO()
    terminal = pexpect.spawn(
        sys.executable, ["-m", "twhy", str(path), "--mode", "blocks",
                        "--lower", "0", "--upper", "255"],
        encoding="utf-8", timeout=15, dimensions=(24, 100),
        env={**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"},
    )
    terminal.logfile_read = transcript
    try:
        terminal.expect(r"L[0-9]")
        terminal.send("h")
        terminal.expect("hex/name/default")
        terminal.send("\x01\x0b#3366ff\r")
        terminal.expect("38;2;51;102;255")
        terminal.send("q")
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    assert terminal.exitstatus == 0
    output = transcript.getvalue()
    assert "Traceback" not in output
    assert "\x1b[?1049h" in output and "\x1b[?1049l" in output
    assert "\x1b[?1006l" in output and "\x1b[?25h" in output


def test_color_request_does_not_pin_sparse_preview_after_overview_completion(tmp_path):
    from test_runtime_overview import drain
    path = tmp_path / "unreduced.ome.tif"
    write_pyramid(path, np.arange(64 * 96, dtype=np.uint16).reshape(64, 96), levels=1)
    with open_image(path) as image, ViewSession(image, cache_mib=0, max_read_pixels=128) as session:
        request = FrameRequest(1, Viewport(0, 0, 96, 64), 24, 12, (0,), (None,),
                               TextRenderer(mode="blocks"))
        sparse = make_frame(session, request)
        assert sparse.samples.plan.sampled_overview is not None
        assert not sparse.samples.cached_overview
        # This is the same completion that could occur behind a queued color request.
        drain(session)
        assert session.overview_ready((0,))
        result = make_frame(session, replace(request, generation=2, colors=((0., 1., 1.),),
                                             cached_samples=sparse.samples))
        assert result.samples.cached_overview
        assert result.samples.plan.viewport == request.viewport
