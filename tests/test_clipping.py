"""Exact clips, native PDF text, safety barriers and real selection controls."""

import asyncio
import base64
import os

import numpy as np
import pytest
from click.testing import CliRunner
from prompt_toolkit.data_structures import Size
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.data_structures import Point

from test_viewer import eventually, ready, running, send
from test_document_navigation import browsing, GatedWorkers
from test_image_input import write_pdf
from ttwhy import TextImage, TextRenderer
from ttwhy.cli import main
from ttwhy.pdf_text import extract_pdf_text
from ttwhy.text_clip import (
    clipboard_delivery, osc52_sequence, validate_clip_path, write_text_clip,
)
from ttwhy.viewer.clipping import clip_options
from ttwhy.viewer.controls import clip_fragments
from ttwhy.viewer.rendering import terminal_lines
from ttwhy.geojson_clip import polygon_feature, rectangle_feature
from ttwhy.viewer.selection import PolygonSelection, RectangleSelection


@pytest.fixture
def cells():
    chars = np.array([list(" a b  "), list("⠁▀ c  "), list("012345"), list("      ")])
    colors = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    return TextImage(chars, foreground=colors, background=255-colors)


def test_exact_crop_owns_colors_spaces_and_only_selected_cells(cells):
    clip = cells.crop(0, 2, 1, 5)
    assert clip.shape == (2, 4)
    assert clip.to_text() == "a b \n▀ c "
    assert not np.shares_memory(clip.characters, cells.characters)
    assert not np.shares_memory(clip.foreground, cells.foreground)
    assert not np.shares_memory(clip.background, cells.background)
    np.testing.assert_array_equal(clip.background, cells.background[:2, 1:5])
    assert "\x1b[" in clip.to_text(ansi=True)
    assert "48;2;" in clip.to_text(ansi=True)
    assert not clip.characters.flags.writeable
    plain = TextImage(cells.characters).crop(0, 2, 1, 5)
    assert plain.foreground is plain.background is None
    assert plain.to_text(ansi=True) == clip.to_text()


@pytest.mark.parametrize("bounds", [(-1, 1, 0, 1), (0, 0, 0, 1), (0, 5, 0, 1),
                                     (0, 1, 0, 7), (0, 1, 5, 3)])
def test_crop_rejects_invalid_rectangle(cells, bounds):
    with pytest.raises(ValueError):
        cells.crop(*bounds)


@pytest.mark.parametrize("bad", [True, 1.5, "1", np.bool_(False)])
def test_crop_rejects_noninteger_edges(cells, bad):
    with pytest.raises(TypeError):
        cells.crop(0, 1, bad, 2)


def test_rectangle_all_drag_directions_and_original_not_border(cells):
    original = cells.to_text(ansi=True)
    for a, b in [((1, 0), (4, 2)), ((4, 2), (1, 0)), ((1, 2), (4, 0)), ((4, 0), (1, 2))]:
        selection = RectangleSelection(cells, terminal_lines(cells))
        selection.start_drag(*a)
        selection.drag(*b, finish=True)
        assert selection.bounds == (0, 3, 1, 5)
        assert selection.clip().to_text() == cells.crop(0, 3, 1, 5).to_text()
        assert "+" in "".join(text for _, text in selection.row(0))
        assert selection.row(999) == []
        assert not selection.dragging
    assert cells.to_text(ansi=True) == original


def test_keyboard_anchor_swap_clamp_and_single_cell(cells):
    selection = RectangleSelection(cells, terminal_lines(cells))
    with pytest.raises(ValueError, match="Space"):
        selection.clip()
    selection.move(-100, -100)
    assert selection.cursor == (0, 0) and selection.bounds is None
    selection.mark()
    assert selection.bounds == (0, 1, 0, 1)
    selection.move(100, 100)
    assert selection.bounds == (0, 4, 0, 6)
    selection.swap_corner()
    selection.move(1, 1)
    assert selection.bounds == (1, 4, 1, 6)
    selection.mark()
    assert selection.bounds == (1, 2, 1, 2)




def test_polygon_selection_adds_points_can_close_and_can_reopen(cells):
    selection = PolygonSelection(cells, terminal_lines(cells))
    selection.mark()
    selection.move(2, 0)
    selection.mark()
    selection.move(0, 1)
    selection.mark()
    assert selection.vertices() == [(3, 2), (5, 2), (5, 3)]
    assert selection.bounds == (2, 4, 3, 6)
    assert "+" in "".join(text for _, text in selection.row(2))
    selection.start_drag(3, 2)
    assert selection.closed
    row = "".join(text for _, text in selection.row(2))
    assert "-" in row or "|" in row or "+" in row
    selection.swap_corner()
    assert not selection.closed
    selection.swap_corner()
    assert selection.vertices() == [(3, 2), (5, 2)]
    with pytest.raises(ValueError, match="Close the polygon"):
        selection.clip()


def test_rectangle_geojson_uses_source_pixel_coordinates(cells):
    selection = RectangleSelection(cells, terminal_lines(cells))
    selection.start_drag(1, 0)
    selection.drag(4, 2, finish=True)
    feature = rectangle_feature(bounds=selection.bounds,
                                viewport=type("Viewport", (), {"x": 100, "y": 200, "width": 60, "height": 40})(),
                                frame_shape=selection.frame.shape, raster_shape=(400, 800),
                                name="box")
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == [110.0, 200.0]
    assert ring[1] == [150.0, 200.0]
    assert ring[2] == [150.0, 230.0]
    assert ring[4] == ring[0]
    assert feature["properties"]["name"] == "box"


def test_polygon_geojson_uses_vertex_centers():
    feature = polygon_feature(vertices=[(0, 0), (5, 0), (5, 3)],
                              viewport=type("Viewport", (), {"x": 10, "y": 20, "width": 60, "height": 40})(),
                              frame_shape=(4, 6), raster_shape=(400, 800),
                              name="poly")
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == [15.0, 25.0]
    assert ring[1] == [65.0, 25.0]
    assert ring[2] == [65.0, 55.0]
    assert ring[-1] == ring[0]

def test_shift_arrow_starts_selection_without_space(cells):
    selection = RectangleSelection(cells, terminal_lines(cells))
    start = selection.cursor
    selection.move(-1, -1, extend=True)
    assert selection.anchor == start
    assert selection.bounds == (1, 3, 2, 4)


@pytest.mark.parametrize("text", ["hello", " a \n b ", "café ⠁▀ 日本語", "\x1b[31m▀\x1b[0m"])
def test_osc52_is_exact_base64_clipboard_write_never_query(text):
    sequence = osc52_sequence(text)
    assert sequence.startswith("\x1b]52;c;") and sequence.endswith("\x07")
    assert base64.b64decode(sequence[7:-1]).decode("utf-8") == text
    assert len(sequence.split("\x1b]")) == 2
    assert ";?" not in sequence


def test_empty_and_oversized_osc52_are_not_sent_or_truncated():
    with pytest.raises(ValueError, match="Nothing"):
        osc52_sequence("")
    with pytest.raises(ValueError, match="64 KiB"):
        osc52_sequence("▀" * 30_000)


def clean_clipboard_env(monkeypatch):
    for key in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "TMUX", "STY", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


def test_auto_local_mac_helper_is_optional_no_shell(monkeypatch):
    import ttwhy.text_clip as module
    clean_clipboard_env(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    result = clipboard_delivery("café")
    assert not result.sequence and result.message == "Copied via pbcopy"
    assert calls[0][0][0] == ["/usr/bin/pbcopy"]
    assert calls[0][1]["input"] == "café".encode()
    assert not calls[0][1].get("shell", False) and calls[0][1]["timeout"] == 2
    assert calls[0][1]["env"]["LC_ALL"] == "en_US.UTF-8"


@pytest.mark.parametrize("envkey", ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "TMUX", "STY"])
def test_remote_and_multiplexer_skip_host_clipboard(monkeypatch, envkey):
    clean_clipboard_env(monkeypatch)
    monkeypatch.setenv(envkey, "present")
    def prohibited(*a, **kw):
        raise AssertionError("Do not modify a remote desktop clipboard")
    monkeypatch.setattr("ttwhy.text_clip.shutil.which", prohibited)
    result = clipboard_delivery("local destination")
    assert result.sequence == osc52_sequence("local destination")
    assert "Sent" in result.message and "Copied" not in result.message


@pytest.mark.parametrize("platform,env,expected,encoding", [
    ("win32", {}, "clip.exe", "utf-16le"),
    ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, "wl-copy", "utf-8"),
    ("linux", {"DISPLAY": ":0"}, "xclip", "utf-8"),
])
def test_optional_local_clipboard_helpers(monkeypatch, platform, env, expected, encoding):
    import ttwhy.text_clip as module
    clean_clipboard_env(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", platform)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(module.shutil, "which", lambda name: name)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    assert clipboard_delivery("▀").message == f"Copied via {expected}"
    assert calls[0][1]["input"] == "▀".encode(encoding)


def test_helper_failure_falls_back_and_off_has_no_side_effect(monkeypatch):
    import subprocess
    clean_clipboard_env(monkeypatch)
    monkeypatch.setattr("ttwhy.text_clip.sys.platform", "darwin")
    monkeypatch.setattr("ttwhy.text_clip.shutil.which", lambda name: name)
    def failure(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], 2)
    monkeypatch.setattr("ttwhy.text_clip.subprocess.run", failure)
    assert clipboard_delivery("abc").sequence == osc52_sequence("abc")
    assert clipboard_delivery("abc", "off").sequence == ""
    monkeypatch.setenv("TERM", "dumb")
    with pytest.raises(RuntimeError, match="Save"):
        clipboard_delivery("abc")
    with pytest.raises(ValueError):
        clipboard_delivery("abc", "invented")


def test_atomic_utf8_clip_preserves_rectangle_and_no_overwrite(tmp_path):
    source, out = tmp_path / "source.png", tmp_path / "clip.txt"
    source.write_bytes(b"unchanged")
    clip_options(source_path=source, output_text_clip=out)
    assert not out.exists()
    write_text_clip(out, " ⠁  \n ▀  ", source=source)
    assert out.read_bytes() == " ⠁  \n ▀  \n".encode()
    with pytest.raises(FileExistsError):
        write_text_clip(out, "replacement", source=source, overwrite=False)
    write_text_clip(out, "replacement", source=source)
    assert out.read_text() == "replacement\n"
    assert source.read_bytes() == b"unchanged"
    assert not list(tmp_path.glob(".*.tmp"))


def test_clip_input_alias_and_output_symlink_safety(tmp_path):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"keep")
    hard, soft = tmp_path / "hard.txt", tmp_path / "soft.txt"
    os.link(source, hard)
    soft.symlink_to(source)
    for out in (source, hard, soft):
        with pytest.raises(ValueError, match="input image"):
            write_text_clip(out, "bad", source=source)
    other = tmp_path / "other.txt"
    other.write_text("keep")
    soft.unlink()
    soft.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        write_text_clip(soft, "bad", source=source)
    with pytest.raises(ValueError, match="different"):
        clip_options(output_text_clip=source, output_ansi_clip=hard)
    assert other.read_text() == "keep" and source.read_bytes() == b"keep"


def test_clip_file_destination_rechecked_before_commit(tmp_path, monkeypatch):
    import ttwhy.text_clip as module
    source, out = tmp_path / "source", tmp_path / "clip"
    source.write_bytes(b"keep")
    original = module.validate_clip_path
    calls = []
    def race(path, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            path.symlink_to(source)
        original(path, **kwargs)
    monkeypatch.setattr(module, "validate_clip_path", race)
    with pytest.raises(ValueError, match="input image"):
        write_text_clip(out, "bad", source=source)
    assert source.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".*.tmp"))


def test_clip_rejects_directory_fifo_and_missing_parent(tmp_path):
    with pytest.raises(ValueError, match="regular"):
        validate_clip_path(tmp_path)
    with pytest.raises(FileNotFoundError):
        validate_clip_path(tmp_path / "missing" / "clip.txt")
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular"):
            validate_clip_path(fifo)


def test_clip_allows_symlink_directory_component(tmp_path):
    directory = tmp_path / "directory"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    write_text_clip(alias / "clip.txt", "works")
    assert (directory / "clip.txt").read_text() == "works\n"


@pytest.mark.parametrize("options", [["--output-text-clip", "a.txt"],
                                     ["--output-ansi-clip", "a.ansi"],
                                     ["--clipboard", "off"], ["--clip-source", "screen"]])
def test_clip_flags_are_interactive_only(tmp_path, options):
    source = tmp_path / "input.png"
    source.write_bytes(b"not read")
    result = CliRunner().invoke(main, [str(source), "--snapshot", *options])
    assert result.exit_code != 0 and "interactive" in result.output


def write_text_pdf(path, *, rotation=0, cropped=False):
    """Independent standard-font PDF with two pages, fractional extents and origins."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Count 2 /Kids [3 0 R 5 0 R] >>"]
    for index in range(2):
        content = (b"BT /F1 12 Tf 50 160 Td (ALPHA one) Tj 0 -50 Td (BETA two) Tj ET "
                   b"BT /F1 10 Tf 175 110 Td (RIGHT) Tj ET" if index == 0 else
                   b"BT /F1 12 Tf 50 160 Td (PAGE TWO) Tj ET")
        crop = " /CropBox [35 45 245.2 195.1]" if cropped else ""
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [20 30 260.4 210.2]"
                        f"{crop} /Rotate {rotation} /Resources << /Font << /F1 7 0 R >> >> "
                        f"/Contents {4+index*2} 0 R >>").encode())
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode()+content+b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode()+obj+b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("dpi", [144, 300, 600])
def test_native_pdf_text_rectangles_rotation_crop_and_fractional_extents(tmp_path, rotation, cropped, dpi):
    from ttwhy.image_input import pdf_raster_shape
    pytest.importorskip("pypdfium2")
    path = write_text_pdf(tmp_path / "text.pdf", rotation=rotation, cropped=cropped)
    height, width = pdf_raster_shape(path, 1, dpi)
    l, b, r, t = (35, 45, 245.2, 195.1) if cropped else (20, 30, 260.4, 210.2)
    # Independent analytic oracle for the four intrinsic quarter rotations.
    def device(x, y):
        u, v = (x-l)/(r-l), (t-y)/(t-b)
        u, v = {0: (u, v), 90: (1-v, u), 180: (1-u, 1-v), 270: (v, 1-u)}[rotation]
        return u*width, v*height
    corners = [device(x, y) for x in (45, 155) for y in (155, 178)]
    xs, ys = zip(*corners)
    bounds = min(xs), min(ys), max(xs), max(ys)
    before = set(tmp_path.iterdir())
    assert extract_pdf_text(path, 1, bounds=bounds, raster_shape=(height, width)) == "ALPHA one"
    assert extract_pdf_text(path, 2, bounds=bounds, raster_shape=(height, width)) == "PAGE TWO"
    assert set(tmp_path.iterdir()) == before


def test_pdf_empty_letterbox_and_invalid_geometry(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_text_pdf(tmp_path / "text.pdf")
    assert extract_pdf_text(path, 1, bounds=(-100, -100, -10, -10), raster_shape=(400, 500)) == ""
    assert extract_pdf_text(path, 1, bounds=(0, 0, 10, 10), raster_shape=(400, 500)) == ""
    for bounds in [(1, 1, 0, 0), (0, 0, float("inf"), 3)]:
        with pytest.raises(ValueError):
            extract_pdf_text(path, 1, bounds=bounds, raster_shape=(400, 500))
    with pytest.raises(IndexError):
        extract_pdf_text(path, 3, bounds=(0, 0, 500, 400), raster_shape=(400, 500))


async def test_selection_keyboard_and_save_freezes_exact_frame(rgb, tmp_path):
    out, ansi = tmp_path / "clip.txt", tmp_path / "clip.ansi"
    async with running(rgb[0], output_text_clip=out, output_ansi_clip=ansi,
                       source_path=rgb[0], clipboard="off") as (viewer, pipe, _, task):
        original, camera = viewer.frame, viewer.viewport
        assert not out.exists()
        await send(pipe, "s\x1b[H \x1b[C\x1b[C\x1b[B", viewer, task)
        selection = viewer.clipping.selection
        assert selection.bounds == (0, 2, 0, 3)
        expected = original.crop(0, 2, 0, 3)
        await send(pipe, "+-rmt2c\x1b[<64;20;10M", viewer, task)
        assert viewer.viewport == camera and viewer.frame is original
        assert viewer.selected_channels == (0,)
        await send(pipe, "y", viewer, task)
        await eventually(lambda: out.exists() and not viewer.clipping.busy, task)
        assert out.read_text() == expected.to_text() + "\n"
        await send(pipe, "W", viewer, task)
        await eventually(lambda: ansi.exists() and not viewer.clipping.busy, task)
        assert ansi.read_text() == expected.to_text(ansi=True) + "\n"
        assert "+" in "".join(x[1] for x in selection.row(0))
        assert viewer.frame is original
        await send(pipe, "s", viewer, task)
        assert not viewer.selecting
        await send(pipe, "+", viewer, task)
        assert viewer.viewport.width < camera.width


async def test_mouse_selection_reversed_inclusive_and_osc52_on_ui_output(rgb, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    async with running(rgb[0], clipboard="osc52") as (viewer, pipe, output, task):
        writes = []
        monkeypatch.setattr(output, "write_raw", writes.append)
        original = viewer.frame
        await send(pipe, "s", viewer, task)
        # Image starts at terminal row 2 in a borrowed Viewer, not row 3.
        await send(pipe, "\x1b[<0;12;9M\x1b[<32;5;4M\x1b[<0;5;4m", viewer, task)
        assert viewer.clipping.selection.bounds == (2, 8, 4, 12)
        await send(pipe, "\x03", viewer, task)  # Ctrl-C copies in selection mode, not quit.
        await eventually(lambda: not viewer.clipping.busy, task)
        assert not task.done()
        osc = [text for text in writes if text.startswith("\x1b]52;")]
        assert osc == [osc52_sequence(original.crop(2, 8, 4, 12).to_text())]
        await send(pipe, "s", viewer, task)
        assert not viewer.selecting


async def test_save_prompt_does_not_resize_or_shift_selection(rgb, tmp_path):
    out = tmp_path / "named clip.txt"
    async with running(rgb[0], clipboard="off") as (viewer, pipe, _, task):
        await send(pipe, "s\x1b[H \x1b[C", viewer, task)
        frame, size, box = viewer.frame, viewer.canvas_size, viewer.clipping.selection.bounds
        await send(pipe, "w", viewer, task)
        assert viewer.editing == "save-text" and viewer.selecting
        assert viewer.canvas_size == size and viewer.frame is frame
        assert viewer.clipping.selection.bounds == box
        await send(pipe, "\x01\x0b" + str(out) + "\r", viewer, task)
        await eventually(lambda: out.exists() and not viewer.clipping.busy, task)
        assert viewer.editing is None and viewer.selecting
        assert out.read_text() == frame.crop(*box).to_text() + "\n"


async def test_save_prompt_cancel_and_bad_path_preserve_selection(rgb, tmp_path):
    async with running(rgb[0], source_path=rgb[0]) as (viewer, pipe, _, task):
        await send(pipe, "s w", viewer, task)
        assert viewer.editing == "save-text"
        await send(pipe, "\x01\x0b" + str(rgb[0]) + "\r", viewer, task)
        assert "input image" in viewer.message and viewer.editing == "save-text"
        await send(pipe, "\x03", viewer, task)
        assert viewer.selecting and viewer.editing is None
        await send(pipe, "s", viewer, task)
        assert not viewer.selecting


async def test_clickable_selection_buttons_require_real_click_not_drag_release(rgb):
    async with running(rgb[0], clipboard="off") as (viewer, pipe, _, task):
        await send(pipe, "\x1b[<0;5;29M\x1b[<0;5;29m", viewer, task)
        assert viewer.selecting
        selection = viewer.clipping.selection
        selection.start_drag(1, 1)
        # Releasing a canvas drag onto Copy is not an explicit copy action.
        fragments = clip_fragments(viewer)
        next(f[2] for f in fragments if len(f) == 3 and f[1] == "y")(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
        assert not selection.dragging and viewer.editing is None
        await send(pipe, "s", viewer, task)
        assert not viewer.selecting


async def test_resize_clears_selection_and_does_not_copy_stale_cells(rgb, tmp_path):
    async with running(rgb[0]) as (viewer, pipe, output, task):
        await send(pipe, "s ", viewer, task)
        output.size = Size(rows=25, columns=80)
        viewer.application._on_resize()
        await ready(viewer, task)
        assert not viewer.selecting
        assert "resize" in viewer.message
        await send(pipe, "s ", viewer, task)
        assert viewer.clipping.selection.frame is viewer.frame


async def test_shift_keyboard_selection_and_tab_swap(rgb):
    async with running(rgb[0]) as (viewer, pipe, _, task):
        await send(pipe, "s\x1b[H\x1b[1;2C\x1b[1;2B", viewer, task)
        selection = viewer.clipping.selection
        assert selection.bounds == (0, 2, 0, 2)
        await send(pipe, "\t\x1b[C", viewer, task)
        assert selection.bounds == (0, 2, 1, 2)
        await send(pipe, "s", viewer, task)


async def test_pdf_copy_extracts_words_not_blocks_and_blank_does_not_overwrite(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_text_pdf(tmp_path / "text.pdf")
    out = tmp_path / "text.txt"
    async with browsing(path, pdf_max_dpi=144) as (nav, viewer, pipe, _, task):
        viewer.clipping.clipboard = "off"
        viewer.clipping.paths[False] = out
        await send(pipe, "s\x1b[H \x1b[Fy", viewer, task)
        await eventually(lambda: not viewer.clipping.busy and out.exists(), task)
        assert "ALPHA one" in out.read_text() and "▀" not in out.read_text()
        await send(pipe, "s]", viewer, task)
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        await send(pipe, "s\x1b[H \x1b[Fy", viewer, task)
        await eventually(lambda: not viewer.clipping.busy, task)
        assert "PAGE TWO" in out.read_text() and "ALPHA" not in out.read_text()
    path = write_pdf(tmp_path / "no-text.pdf")
    before = out.read_bytes()
    async with browsing(path, pdf_max_dpi=144) as (_, viewer, pipe, _, task):
        viewer.clipping.clipboard = "off"
        viewer.clipping.paths[False] = out
        await send(pipe, "s\x1b[H \x1b[Fy", viewer, task)
        await eventually(lambda: not viewer.clipping.busy, task)
        assert "No embedded PDF text" in viewer.message
        assert out.read_bytes() == before


async def test_pdf_refinement_waits_until_selection_exits(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    monkeypatch.setattr("ttwhy.viewer.navigation.PdfWorkers", GatedWorkers)
    path = write_text_pdf(tmp_path / "refinement.pdf")
    async with browsing(path, pdf_max_dpi=600) as (nav, viewer, pipe, _, task):
        await eventually(lambda: nav._workers is not None and len(nav._workers.calls) == 2, task)
        await send(pipe, "s ", viewer, task)
        frozen, camera = viewer.frame, viewer.viewport
        for gate in nav._workers.gates.values():
            gate.set()
        await eventually(lambda: nav._refine_task.done(), task)
        assert nav.cache[1].dpi == 600 and nav.active.dpi == 144
        assert viewer.frame is frozen and viewer.viewport == camera and viewer.selecting
        pipe.send_text("s")
        await eventually(lambda: nav.active.dpi == 600 and not viewer.busy, task)
        assert not viewer.selecting


async def test_pdf_source_changed_refuses_extraction(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_text_pdf(tmp_path / "changing.pdf")
    out = tmp_path / "words.txt"
    async with browsing(path, pdf_max_dpi=144) as (_, viewer, pipe, _, task):
        viewer.clipping.clipboard = "off"
        viewer.clipping.paths[False] = out
        await send(pipe, "s\x1b[H \x1b[F", viewer, task)
        path.write_bytes(path.read_bytes() + b"\n")
        await send(pipe, "y", viewer, task)
        await eventually(lambda: not viewer.clipping.busy, task)
        assert "changed on disk" in viewer.message and not out.exists()


async def test_explicit_screen_source_on_pdf_is_not_native_extraction(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    path = write_text_pdf(tmp_path / "screen.pdf")
    async with browsing(path, pdf_max_dpi=144) as (_, viewer, pipe, _, task):
        def prohibited(*a, **kw):
            raise AssertionError("Screen copying must not decode PDF text")
        monkeypatch.setattr("ttwhy.viewer.clipping.extract_pdf_text", prohibited)
        viewer.clipping.clip_source = "screen"
        viewer.clipping.clipboard = "off"
        out = tmp_path / "screen.txt"
        viewer.clipping.paths[False] = out
        await send(pipe, "s\x1b[H \x1b[Fy", viewer, task)
        await eventually(lambda: not viewer.clipping.busy and out.exists(), task)
        assert "▀" in out.read_text() and "ALPHA" not in out.read_text()


async def test_save_failure_retains_view_and_selection(rgb, tmp_path):
    out = tmp_path / "existing.txt"
    async with running(rgb[0], output_text_clip=out, clipboard="off", clip_overwrite=False) as (
        viewer, pipe, _, task,
    ):
        out.write_text("created after opening")
        await send(pipe, "s wy", viewer, task)
        await eventually(lambda: not viewer.clipping.busy, task)
        assert "already exists" in viewer.message
        assert out.read_text() == "created after opening"
        assert viewer.selecting and not task.done()


async def test_pdf_worker_cancellation_during_successful_finalizer(tmp_path, monkeypatch):
    from ttwhy.pdf_workers import PdfWorkers
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "cleanup.pdf")
    workers = PdfWorkers()
    original = workers._finish_job
    entered, release = asyncio.Event(), asyncio.Event()
    async def delayed(*args):
        entered.set()
        await release.wait()
        await original(*args)
    monkeypatch.setattr(workers, "_finish_job", delayed)
    job = asyncio.create_task(workers.render(path, 1, 144, (255, 255, 255)))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        job.cancel()
        await asyncio.sleep(.01)
        job.cancel()  # Repeated cancellation must not abandon the shielded finalizer.
        await asyncio.sleep(.01)
        assert not job.done() and workers.active_count == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(job, 10)
        assert workers.active_count == 0
    finally:
        release.set()
        if not job.done():
            await asyncio.gather(job, return_exceptions=True)
        workers.close()


async def test_accepted_save_finishes_on_immediate_quit(rgb, tmp_path):
    out = tmp_path / "last.txt"
    async with running(rgb[0], output_text_clip=out, clipboard="off") as (viewer, pipe, _, task):
        await send(pipe, "s\x1b[H \x1b[C", viewer, task)
        expected = viewer.clipping.selection.clip().to_text() + "\n"
        pipe.send_text("wq")
        await asyncio.wait_for(task, 5)
        assert out.read_text() == expected
        assert viewer.clipping._executor is None


async def test_saving_succeeds_even_if_clipboard_request_is_too_large(rgb, tmp_path, monkeypatch):
    out = tmp_path / "fallback.txt"
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr("ttwhy.text_clip.MAX_OSC52_BYTES", 1)
    async with running(rgb[0], output_text_clip=out, clipboard="osc52") as (viewer, pipe, _, task):
        await send(pipe, "s\x1b[H \x1b[Fy", viewer, task)
        await eventually(lambda: not viewer.clipping.busy and out.exists(), task)
        assert out.read_text() == viewer.clipping.selection.clip().to_text() + "\n"
        assert "Saved" in viewer.message and "Clipboard unavailable" in viewer.message
        assert viewer.selecting


async def test_clip_worker_does_not_block_selection_and_coalesces_requests(rgb, tmp_path, monkeypatch):
    import threading
    import ttwhy.viewer.clipping as module
    entered, release = threading.Event(), threading.Event()
    original = module.write_text_clip
    calls = []
    def slow(*a, **kw):
        calls.append(a)
        entered.set()
        assert release.wait(4)
        original(*a, **kw)
    monkeypatch.setattr(module, "write_text_clip", slow)
    out = tmp_path / "slow.txt"
    async with running(rgb[0], output_text_clip=out, clipboard="off") as (viewer, pipe, _, task):
        await send(pipe, "s\x1b[H \x1b[C", viewer, task)
        expected = viewer.clipping.selection.clip().to_text() + "\n"
        try:
            pipe.send_text("w")
            await eventually(entered.is_set, task)
            pipe.send_text("\x1b[Cww")
            await asyncio.sleep(.1)
            assert viewer.clipping.selection.cursor == (2, 0)
            assert len(calls) == 1 and viewer.clipping.busy
            assert "already" in viewer.message
            release.set()
            await eventually(lambda: not viewer.clipping.busy, task)
            assert out.read_text() == expected  # Not the rectangle changed during the write.
        finally:
            release.set()

# Shareable image clips: Discord cannot interpret truecolor ANSI/OSC 52 artwork.

def test_png_clip_exact_half_block_rgb_and_no_selection_overlay():
    from io import BytesIO
    from PIL import Image
    from ttwhy.png_clip import render_clip_png

    chars = np.array([list("▀▀ "), list("▀▀▀")])
    foreground = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 0]],
                           [[0, 0, 255], [255, 255, 255], [20, 30, 40]]], dtype=np.uint8)
    background = np.array([[[0, 0, 255], [255, 0, 0], [0, 0, 0]],
                           [[255, 255, 0], [5, 6, 7], [220, 230, 240]]], dtype=np.uint8)
    frame = TextImage(chars, foreground=foreground, background=background)
    data = render_clip_png(frame)
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and b"\x1b[" not in data
    with Image.open(BytesIO(data)) as image:
        image.load()
        assert image.size == (36, 48) and image.mode == "RGB"
        assert image.getpixel((5, 5)) == (255, 0, 0)
        assert image.getpixel((5, 15)) == (0, 0, 255)
        assert image.getpixel((18, 4)) == (0, 255, 0)
        assert image.getpixel((18, 20)) == (255, 0, 0)
        assert image.getpixel((31, 5)) == (0, 0, 0)
        assert image.getpixel((29, 36)) == (220, 230, 240)
    assert frame.to_text() == "▀▀ \n▀▀▀"


def test_png_clip_text_and_braille_have_visible_ink():
    from io import BytesIO
    from PIL import Image
    from ttwhy.png_clip import render_clip_png

    frame = TextImage(np.array([["A", "⠁", " "]]),
                      foreground=np.array([[[30, 150, 240], [255, 255, 255], [0, 0, 0]]],
                                          dtype=np.uint8))
    with Image.open(BytesIO(render_clip_png(frame))) as image:
        value = np.array(image)
    assert np.count_nonzero(value[:, :12]) > 0
    assert np.count_nonzero(value[:, 12:24]) > 0
    assert not np.any(value[:, 24:])


def test_png_clip_rejects_unbounded_output_before_allocating():
    from ttwhy.png_clip import render_clip_png

    with pytest.raises(ValueError, match="limit"):
        render_clip_png(TextImage(np.full((40, 40), "#")),
                        cell_width=128, cell_height=256)


def test_png_clip_atomic_file_and_hardlink_protection(tmp_path):
    from ttwhy.image_clip import write_png_clip
    from ttwhy.png_clip import render_clip_png

    source, output = tmp_path / "image.png", tmp_path / "detail.png"
    source.write_bytes(b"original source bytes")
    png = render_clip_png(TextImage(np.array([["▀"]]),
                                     foreground=np.array([[[255, 0, 0]]], dtype=np.uint8),
                                     background=np.array([[[0, 0, 255]]], dtype=np.uint8)))
    write_png_clip(output, png, source=source)
    assert output.read_bytes() == png
    with pytest.raises(FileExistsError):
        write_png_clip(output, png, source=source, overwrite=False)
    alias = tmp_path / "alias.png"
    os.link(source, alias)
    with pytest.raises(ValueError, match="input image"):
        write_png_clip(alias, png, source=source)
    with pytest.raises(ValueError, match="input image"):
        write_png_clip(source, png, source=source)
    assert source.read_bytes() == b"original source bytes"
    with pytest.raises(ValueError, match="different"):
        clip_options(output_png_clip=output, output_ansi_clip=output)
    assert not list(tmp_path.glob(".*.tmp"))


def test_image_clipboard_local_mac_uses_png_data_not_filename(monkeypatch):
    from ttwhy import image_clip
    from ttwhy.png_clip import render_clip_png
    clean_clipboard_env(monkeypatch)
    monkeypatch.setattr(image_clip.sys, "platform", "darwin")
    monkeypatch.setattr(image_clip.shutil, "which", lambda exe: "/usr/bin/osascript")
    png = render_clip_png(TextImage(np.array([["▀"]])))
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        assert Path(args[-1]).read_bytes() == png
        assert "PNGf" in " ".join(args)
        assert not kwargs.get("shell", False)
    from pathlib import Path
    monkeypatch.setattr(image_clip.subprocess, "run", run)
    result = image_clip.clipboard_png(png)
    assert result.message == "Copied PNG via macOS clipboard"
    assert len(calls) == 1 and not Path(calls[0][0][-1]).exists()


def test_image_clipboard_linux_native_and_remote_fallback(monkeypatch):
    from ttwhy import image_clip
    from ttwhy.png_clip import render_clip_png
    clean_clipboard_env(monkeypatch)
    monkeypatch.setattr(image_clip.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(image_clip.shutil, "which", lambda name: "/usr/bin/" + name)
    png = render_clip_png(TextImage(np.array([["▀"]])))
    calls = []
    monkeypatch.setattr(image_clip.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    assert image_clip.clipboard_png(png).message == "Copied PNG via wl-copy"
    assert calls[0][1]["input"] == png and calls[0][0][-1] == "image/png"
    with pytest.raises(RuntimeError, match="OSC 52"):
        image_clip.clipboard_png(png, "osc52")
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    with pytest.raises(RuntimeError, match="SSH/tmux"):
        image_clip.clipboard_png(png)
    with pytest.raises(RuntimeError, match="disabled"):
        image_clip.clipboard_png(png, "off")
    assert len(calls) == 1


async def test_png_selection_keyboard_copies_and_saves_real_image(rgb, tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image
    from ttwhy.text_clip import ClipboardDelivery
    from ttwhy.viewer import clipping

    output = tmp_path / "share.png"
    captured = []
    def fake_clipboard(png, mode):
        captured.append(png)
        return ClipboardDelivery("Native PNG clipboard helper accepted")
    monkeypatch.setattr(clipping, "clipboard_png", fake_clipboard)
    async with running(rgb[0], output_png_clip=output, source_path=rgb[0],
                       clipboard="auto", renderer=TextRenderer(mode="blocks")) as (
                           viewer, pipe, _, task):
        await send(pipe, "s\x1b[H \x1b[C\x1b[C\x1b[B", viewer, task)
        expected = viewer.clipping.selection.clip()
        await send(pipe, "i", viewer, task)
        await eventually(lambda: output.exists() and not viewer.clipping.busy, task)
        assert len(captured) == 1 and captured[0] == output.read_bytes()
        with Image.open(BytesIO(captured[0])) as actual:
            assert actual.size == (expected.shape[1]*12, expected.shape[0]*24)
        output.unlink()
        await send(pipe, "P", viewer, task)
        await eventually(lambda: output.exists() and not viewer.clipping.busy, task)
        assert output.read_bytes().startswith(b"\x89PNG")
        assert len(captured) == 1  # Save does not access clipboard.
        assert viewer.selecting and not task.done()


def test_png_flag_snapshot_rejected_before_decode(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"not actually decoded")
    result = CliRunner().invoke(main, [str(path), "--snapshot", "--output-png-clip",
                                       str(tmp_path / "share.png")])
    assert result.exit_code != 0 and "interactive" in result.output