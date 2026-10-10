"""End-to-end Linux/macOS PTY test of actual terminal output and restoration."""

import io
import os
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal test")
def test_real_terminal_mouse_resize_and_exit(rgb):
    pexpect = pytest.importorskip("pexpect")
    terminal = pexpect.spawn(
        sys.executable, ["-m", "ttwhy", str(rgb[0])], encoding="utf-8", timeout=10,
        dimensions=(30, 110),
        env={**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"},
    )
    transcript = io.StringIO()
    terminal.logfile_read = transcript
    try:
        terminal.expect("ttwhy")
        terminal.expect(r"L[0-9]")
        terminal.send("\x1b[<64;25;12M")
        time.sleep(.15)
        terminal.send("b")
        terminal.expect("▀")
        terminal.send("u\x01\x0b200\r")
        # Screen diffs may reuse characters through cursor motion rather than
        # emitting a contiguous label. Exact editor state is covered by pipe-input tests.
        time.sleep(.25)
        terminal.setwinsize(24, 85)
        time.sleep(.2)
        terminal.send("q")
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    assert terminal.exitstatus == 0
    output = transcript.getvalue()
    assert "Traceback" not in output
    assert "\x1b[?1049h" in output and "\x1b[?1049l" in output  # alternate screen
    assert "\x1b[?1006h" in output and "\x1b[?1006l" in output  # SGR mouse protocol
    assert "\x1b[?25h" in output  # cursor restored
    assert "38;2;" in output and "48;2;" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal test")
@pytest.mark.parametrize("mode", ["detail", "shape", "simple"])
def test_terminal_interrupt_restores_screen(mif, mode):
    pexpect = pytest.importorskip("pexpect")
    terminal = pexpect.spawn(sys.executable, ["-m", "ttwhy", str(mif[0]), "--mode", mode],
                             encoding="utf-8",
                             timeout=10, dimensions=(20, 80),
                             env={**os.environ, "TERM": "xterm-256color"})
    transcript = io.StringIO()
    terminal.logfile_read = transcript
    try:
        terminal.expect(r"L[0-9]")
        terminal.sendcontrol("c")
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    assert terminal.exitstatus == 0
    assert "\x1b[?1049l" in transcript.getvalue()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal test")
@pytest.mark.parametrize("mode_env,expected", [
    ({"COLORTERM": "truecolor"}, "blocks"),
    ({"PROMPT_TOOLKIT_COLOR_DEPTH": "DEPTH_8_BIT"}, "blocks"),
    ({"PROMPT_TOOLKIT_COLOR_DEPTH": "DEPTH_4_BIT"}, "simple"),
    ({"NO_COLOR": "1"}, "simple"),
])
def test_real_cli_auto_selects_terminal_renderer(mif, mode_env, expected):
    pexpect = pytest.importorskip("pexpect")
    env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "",
           "PROMPT_TOOLKIT_COLOR_DEPTH": "", "NO_COLOR": "", **mode_env}
    # NO_COLOR='' is not active; remove inherited overrides so each case is isolated.
    env = {k: v for k, v in env.items() if v != '' or k not in
           {"NO_COLOR", "COLORTERM", "PROMPT_TOOLKIT_COLOR_DEPTH"}}
    terminal = pexpect.spawn(
        sys.executable, ["-m", "ttwhy", str(mif[0])], encoding="utf-8", timeout=15,
        dimensions=(20, 80), env=env,
    )
    try:
        terminal.expect(f"\\| {expected} ")
        terminal.send("q")
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    assert terminal.exitstatus == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal test")
@pytest.mark.parametrize("kind", ["png", "pdf"])
def test_new_file_formats_real_cli_auto_zoom_and_terminal_restoration(tmp_path, kind):
    import numpy as np
    from PIL import Image as PILImage
    from test_image_input import write_pdf

    pexpect = pytest.importorskip("pexpect")
    path = tmp_path / f"terminal.{kind}"
    if kind == "pdf":
        pytest.importorskip("pypdfium2")
        write_pdf(path)
    else:
        y, x = np.mgrid[:120, :160]
        PILImage.fromarray(np.stack([x, y, x+y], -1).astype(np.uint8)).save(path)
    env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
    env.pop("NO_COLOR", None)
    env.pop("PROMPT_TOOLKIT_COLOR_DEPTH", None)
    terminal = pexpect.spawn(sys.executable, ["-m", "ttwhy", str(path)], encoding="utf-8",
                             dimensions=(24, 100), env=env, timeout=15)
    transcript = io.StringIO()
    terminal.logfile_read = transcript
    try:
        terminal.expect(r"\| blocks ")
        terminal.send("+\x1b[<64;20;10M\x1b[C")
        time.sleep(.2)
        terminal.send("q")
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    assert terminal.exitstatus == 0, transcript.getvalue()
    output = transcript.getvalue()
    assert "Traceback" not in output and "▀" in output
    assert "\x1b[?1049h" in output and "\x1b[?1049l" in output
    assert "\x1b[?1006l" in output and "\x1b[?25h" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal test")
@pytest.mark.parametrize("kind", ["pdf", "tiff"])
def test_document_navigation_in_real_terminal(tmp_path, kind):
    from test_document_navigation import write_book, write_series

    pexpect = pytest.importorskip("pexpect")
    if kind == "pdf":
        pytest.importorskip("pypdfium2")
        path = write_book(tmp_path / "book.pdf")
        options = ["--pdf-max-dpi", "144", "--pdf-cache-pages", "1"]
        label, target, final_title = "page", "4", "page 4/5"
    else:
        path = tmp_path / "series.ome.tif"
        write_series(path)
        options = []
        label, target, final_title = "series", "2", "series 2 of 3"
    env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
    env.pop("NO_COLOR", None)
    env.pop("PROMPT_TOOLKIT_COLOR_DEPTH", None)
    terminal = pexpect.spawn(sys.executable, ["-m", "ttwhy", str(path), *options],
                             encoding="utf-8", dimensions=(26, 110), env=env, timeout=15)
    transcript = io.StringIO()
    terminal.logfile_read = transcript
    try:
        terminal.expect(r"\| blocks ")
        terminal.send("p")
        terminal.expect(f"Go to {label}")
        terminal.send(target + "\r")
        # Repaint after selection so terminal screen-diff cursor reuse does not
        # hide the complete title from this output-level assertion.
        time.sleep(.25)
        terminal.setwinsize(27, 111)
        terminal.expect(final_title)
        terminal.send("q")
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    output = transcript.getvalue()
    assert terminal.exitstatus == 0, output
    assert "Traceback" not in output
    assert "\x1b[?1049l" in output and "\x1b[?1006l" in output and "\x1b[?25h" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal test")
@pytest.mark.parametrize("kind", ["png", "pdf"])
def test_real_terminal_rectangle_copy_and_clip_files(tmp_path, kind):
    import base64
    from PIL import Image as PILImage
    from test_clipping import write_text_pdf

    pexpect = pytest.importorskip("pexpect")
    if kind == "pdf":
        pytest.importorskip("pypdfium2")
        path = write_text_pdf(tmp_path / "select.pdf")
        options = ["--pdf-max-dpi", "144", "--pdf-cache-pages", "1"]
    else:
        path = tmp_path / "select.png"
        PILImage.new("RGB", (96, 64), (120, 60, 180)).save(path)
        options = []
    text_path, ansi_path = tmp_path / "clip.txt", tmp_path / "clip.ansi"
    env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
    env.pop("NO_COLOR", None)
    env.pop("PROMPT_TOOLKIT_COLOR_DEPTH", None)
    terminal = pexpect.spawn(
        sys.executable, ["-m", "ttwhy", str(path), *options,
                         "--clipboard", "osc52", "--output-text-clip", str(text_path),
                         "--output-ansi-clip", str(ansi_path)], encoding="utf-8",
        dimensions=(26, 110), env=env, timeout=15,
    )
    transcript = io.StringIO()
    terminal.logfile_read = transcript
    try:
        terminal.expect(r"\| blocks ")
        terminal.send("s\x1b[H \x1b[Fy")
        terminal.expect("\x1b\\]52;c;([^\x07]+)\x07")
        payload = base64.b64decode(terminal.match.group(1)).decode("utf-8")
        assert text_path.read_text() == payload + "\n"
        if kind == "pdf":
            assert "ALPHA one" in payload and "BETA two" in payload
        else:
            assert "▀" in payload and "\x1b" not in payload
        terminal.send("T")
        # File publication is asynchronous. Poll its explicit destination, not
        # transient status text which prompt_toolkit can render as screen diffs.
        deadline = time.monotonic() + 5
        while not ansi_path.exists() and time.monotonic() < deadline:
            try:
                terminal.read_nonblocking(65536, timeout=.1)
            except pexpect.TIMEOUT:
                pass
        assert ansi_path.exists()
        assert "38;2;" in ansi_path.read_text() and "48;2;" in ansi_path.read_text()
        terminal.send("sq")
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    assert terminal.exitstatus == 0, transcript.getvalue()
    output = transcript.getvalue()
    assert "Traceback" not in output
    assert "\x1b[?1049l" in output and "\x1b[?1006l" in output and "\x1b[?25h" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal test")
def test_terminal_mixed_regions_private_geojson_png_and_ascii(tmp_path):
    """Actual CLI: mixed shapes, all-area publication, ASCII and clean restoration."""
    import json
    from uuid import UUID

    import numpy as np
    from PIL import Image as PILImage

    pexpect = pytest.importorskip("pexpect")
    source = tmp_path / "private-subject-7821.png"
    y, x = np.mgrid[:64, :96]
    PILImage.fromarray(np.stack((x * 2, y * 3, x + y), axis=-1).astype(np.uint8)).save(source)
    geo, png, text, ansi = (tmp_path / name for name in
                            ("regions.geojson", "selection.png", "selection.txt", "selection.ansi"))
    env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
    env.pop("NO_COLOR", None)
    env.pop("PROMPT_TOOLKIT_COLOR_DEPTH", None)
    terminal = pexpect.spawn(
        sys.executable, ["-m", "ttwhy", str(source), "--clipboard", "off",
                         "--output-geojson-clip", str(geo), "--output-png-clip", str(png),
                         "--output-text-clip", str(text), "--output-ansi-clip", str(ansi)],
        encoding="utf-8", dimensions=(26, 110), env=env, timeout=15,
    )
    transcript = io.StringIO()
    terminal.logfile_read = transcript

    def click(cx, cy):
        # Terminal origin is one-based; canvas starts below the one-row title.
        return f"\x1b[<0;{cx+1};{cy+2}M\x1b[<0;{cx+1};{cy+2}m"

    def saved(path):
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            try:
                terminal.read_nonblocking(65536, timeout=.1)
            except pexpect.TIMEOUT:
                pass
        assert path.exists(), transcript.getvalue()
        time.sleep(.1)  # Allow the copy-result task to release its busy slot.

    try:
        terminal.expect(r"\| blocks ")
        terminal.send("s")
        time.sleep(.15)
        terminal.send("\x1b[<0;13;6M\x1b[<32;21;10M\x1b[<0;21;10m")
        terminal.send("av")  # Retain the rectangle; change the new empty area to polygon.
        for point in ((40, 6), (55, 6), (52, 17), (39, 14), (40, 6)):
            terminal.send(click(*point))
        terminal.send("J")
        saved(geo)
        features = json.loads(geo.read_text())["features"]
        assert len(features) == 2
        assert {f["properties"]["selection_type"] for f in features} == {"rectangle", "polygon"}
        assert len({UUID(f["id"]) for f in features}) == 2
        assert all(UUID(f["id"]).version == 4 for f in features)
        assert source.name not in geo.read_text() and str(tmp_path) not in geo.read_text()
        terminal.send("I")
        saved(png)
        with PILImage.open(png) as image:
            assert image.mode == "RGBA"
            pixels = np.asarray(image)
            absent = pixels[..., 3] == 0
            assert absent.any() and (pixels[..., 3] == 255).any()
            assert not pixels[absent, :3].any()
        terminal.send("MT")
        saved(ansi)
        value = ansi.read_text()
        assert value.isascii() and "\x1b" not in value and value.strip()
        terminal.send("Y")
        saved(text)
        assert text.read_bytes() == ansi.read_bytes()
        terminal.send("sq")  # Switch adds a new empty rectangle; q always exits.
        terminal.expect(pexpect.EOF)
    finally:
        terminal.close(force=True)
    assert terminal.exitstatus == 0, transcript.getvalue()
    output = transcript.getvalue()
    assert "Traceback" not in output
    assert "\x1b[?1049l" in output and "\x1b[?1006l" in output and "\x1b[?25h" in output
