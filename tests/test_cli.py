"""CLI tests exercise actual file reads, export bytes, and error preflights."""

import os
import re

import pytest
from click.testing import CliRunner

from ttwhy import __version__
from ttwhy.cli import main


def test_help_and_eager_version_without_input():
    runner = CliRunner()
    version = runner.invoke(main, ["--version"])
    assert version.exit_code == 0 and __version__ in version.output
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    assert "--snapshot" in help_result.output and "--max-read-pixels" in help_result.output


@pytest.mark.parametrize("fixture_name", ["mif", "rgb"])
def test_plain_snapshot_is_exact_rectangle(request, fixture_name):
    path, _ = request.getfixturevalue(fixture_name)
    result = CliRunner().invoke(main, [str(path), "--snapshot", "--columns", "33", "--rows", "11"])
    assert result.exit_code == 0, result.output
    assert len(result.stdout.splitlines()) == 11
    assert all(len(line) == 33 for line in result.stdout.splitlines())
    assert "\x1b" not in result.stdout
    assert len(set(result.stdout)) > 3


def test_ansi_blocks_and_multi_channel(mif, rgb):
    runner = CliRunner()
    output = runner.invoke(main, [str(rgb[0]), "--snapshot", "--mode", "blocks", "--ansi",
                                  "--columns", "30", "--rows", "10"])
    assert output.exit_code == 0, output.output
    assert "38;2;" in output.output and "48;2;" in output.output
    stripped = re.sub(r"\x1b\[[0-9;]+m", "", output.output)
    assert "▀" in stripped and len(stripped.splitlines()) == 10
    output = runner.invoke(main, [str(mif[0]), "--snapshot", "--channel", "0", "--channel", "2",
                                  "--lower", "100", "--upper", "10000"])
    assert output.exit_code == 0


def test_output_atomic_and_no_overwrite(mif, tmp_path):
    path = tmp_path / "image.txt"
    args = [str(mif[0]), "--snapshot", "-o", str(path)]
    runner = CliRunner()
    assert runner.invoke(main, args).exit_code == 0
    initial = path.read_bytes()
    assert len(initial) > 100
    path.write_text("keep me")
    result = runner.invoke(main, [*args, "--no-overwrite"])
    assert result.exit_code != 0 and "already exists" in result.output
    assert path.read_text() == "keep me"
    assert runner.invoke(main, args).exit_code == 0
    assert path.read_bytes() == initial
    assert not list(tmp_path.glob(".*.tmp"))


def test_never_overwrite_input_or_hardlink(mif, tmp_path):
    original = mif[0].read_bytes()
    alias = tmp_path / "alias.txt"
    os.link(mif[0], alias)
    for destination in (mif[0], alias):
        result = CliRunner().invoke(main, [str(mif[0]), "--snapshot", "-o", str(destination)])
        assert result.exit_code != 0 and "must not be the input" in result.output
    assert mif[0].read_bytes() == original


@pytest.mark.parametrize("options", [["--lower", "10"], ["--upper", "100"],
                                      ["--color"], ["--mode", "blocks"],
                                      ["--no-overwrite"], ["--lower", "5", "--upper", "4"],
                                      ["--columns", "0"], ["--channel", "99"]])
def test_bad_snapshot_options(mif, options):
    result = CliRunner().invoke(main, [str(mif[0]), "--snapshot", *options])
    assert result.exit_code != 0
    # Recognized invalid inputs should produce a Click error, not a traceback.
    assert "Error:" in result.output, (result.output, result.exception)


def test_interactive_requires_terminal(mif):
    result = CliRunner().invoke(main, [str(mif[0])])
    assert result.exit_code != 0 and "--snapshot" in result.output


def test_crop_and_budget_warning(mif):
    result = CliRunner().invoke(main, [
        str(mif[0]), "--snapshot", "--region", "10", "10", "20", "20",
        "--columns", "10", "--rows", "10",
    ])
    assert result.exit_code == 0
    result = CliRunner().invoke(main, [str(mif[0]), "--snapshot", "--max-read-pixels", "100"])
    assert result.exit_code == 0 and "bounded center crop" in result.output


def test_file_output_preflight_runs_before_decode(mif, tmp_path, monkeypatch):
    from ttwhy import OmeTiffSource
    def forbidden(*args):
        raise AssertionError("Input should not have been opened")
    monkeypatch.setattr(OmeTiffSource, "open", forbidden)
    out = tmp_path / "existing.txt"
    out.write_text("untouched")
    result = CliRunner().invoke(main, [str(mif[0]), "--snapshot", "-o", str(out), "--no-overwrite"])
    assert result.exit_code != 0 and "already exists" in result.output
