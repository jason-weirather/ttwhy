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
