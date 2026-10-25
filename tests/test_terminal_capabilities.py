"""Auto selects terminal modes without writing probe sequences or opening an image."""

import pytest
from prompt_toolkit.output import ColorDepth

from twhy.terminal_capabilities import (
    available_color_depth, configured_color_depth, resolve_auto_mode,
)


@pytest.mark.parametrize('depth,expected', [
    (ColorDepth.DEPTH_24_BIT, 'blocks'),
    (ColorDepth.DEPTH_8_BIT, 'blocks'),
    (ColorDepth.DEPTH_4_BIT, 'simple'),
    (ColorDepth.DEPTH_1_BIT, 'simple'),
    (None, 'simple'),
])
def test_interactive_auto_uses_eligible_terminal_depth(depth, expected):
    assert resolve_auto_mode(
        snapshot=False, ansi=False, color=True, is_tty=True,
        term='xterm-256color', encoding='utf-8', depth=depth,
    ) == expected


@pytest.mark.parametrize('kwargs', [
    {'is_tty': False},
    {'term': 'dumb'},
    {'term': 'unknown'},
    {'encoding': 'ascii'},
    {'encoding': 'cp1252'},
    {'encoding': 'not-a-real-encoding'},
    {'encoding': None},
    {'color': False},
])
def test_interactive_auto_falls_back_to_simple(kwargs):
    options = dict(snapshot=False, ansi=False, color=True, is_tty=True,
                   term='xterm-256color', encoding='utf-8', depth=ColorDepth.DEPTH_24_BIT)
    options.update(kwargs)
    assert resolve_auto_mode(**options) == 'simple'


@pytest.mark.parametrize('ansi,color,expected', [
    (False, True, 'simple'), (False, False, 'simple'),
    (True, True, 'blocks'), (True, False, 'simple'),
])
def test_snapshot_auto_does_not_depend_on_terminal(ansi, color, expected):
    assert resolve_auto_mode(snapshot=True, ansi=ansi, color=color) == expected


def test_configured_color_depth_honors_explicit_preference(monkeypatch):
    monkeypatch.setenv('COLORTERM', 'truecolor')
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.delenv('PROMPT_TOOLKIT_COLOR_DEPTH', raising=False)
    assert configured_color_depth() == ColorDepth.DEPTH_24_BIT
    monkeypatch.setenv('PROMPT_TOOLKIT_COLOR_DEPTH', 'DEPTH_4_BIT')
    assert configured_color_depth() == ColorDepth.DEPTH_4_BIT
    monkeypatch.setenv('NO_COLOR', '1')
    assert configured_color_depth() == ColorDepth.DEPTH_1_BIT
    assert available_color_depth() == ColorDepth.DEPTH_1_BIT


def test_default_terminal_depth_uses_toolkit_not_assumed_truecolor(monkeypatch):
    class Output:
        def get_default_color_depth(self):
            return ColorDepth.DEPTH_8_BIT

    monkeypatch.delenv('COLORTERM', raising=False)
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.delenv('PROMPT_TOOLKIT_COLOR_DEPTH', raising=False)
    monkeypatch.setattr('twhy.terminal_capabilities.create_output', lambda: Output())
    assert configured_color_depth() is None
    assert available_color_depth() == ColorDepth.DEPTH_8_BIT


@pytest.mark.parametrize("flag,expected", [([], False), (["--no-color"], False),
                                            (["--color"], True)])
def test_cli_one_bit_environment_initializes_renderer_color(tmp_path, monkeypatch, flag, expected):
    """Automatic fallback must set the renderer flag, not only discard ANSI SGR."""
    from types import SimpleNamespace
    from click.testing import CliRunner
    from PIL import Image
    import twhy.cli as cli
    import twhy.viewer.file_viewer as file_viewer

    path = tmp_path / "mono.png"
    Image.new("RGB", (4, 4), (100, 140, 190)).save(path)
    observed = []

    async def fake_view_file(_path, **kwargs):
        observed.append(kwargs)

    monkeypatch.setattr(file_viewer, "view_file", fake_view_file)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(cli, "sys", SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: True),
        stdout=SimpleNamespace(isatty=lambda: True, encoding="utf-8"),
    ))
    result = CliRunner().invoke(cli.main, [str(path), *flag])
    assert result.exit_code == 0, result.output
    assert len(observed) == 1
    assert observed[0]["color"] is expected
    assert observed[0]["mode"] == "simple"
