"""Spatial character, finite-coverage, byte-compatibility, and sampling contracts."""

import ast
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from omeify import OMETiffReader
from wcwidth import wcwidth

from twhy import TextRenderer, Viewport, ViewSession
from twhy.cli import main
from twhy.glyph_atlas import SAMPLE_YX, glyph_atlas
from twhy.structured_rendering import braille_cells, cell_patches
from twhy.text_renderer import MAX_STRUCTURED_SAMPLES


@pytest.mark.parametrize('mode', ['detail', 'shape', 'simple', 'ascii', 'blocks'])
def test_mode_sample_contract_and_geometry(mode):
    renderer = TextRenderer(mode=mode)
    scale = (8, 4) if mode in ('detail', 'shape') else (2, 1) if mode == 'blocks' else (1, 1)
    assert renderer.sample_shape(rows=5, columns=7) == (5 * scale[0], 7 * scale[1])
    frame = renderer.render(np.eye(35, 49), rows=5, columns=7)
    assert frame.shape == (5, 7)
    assert all(wcwidth(char) == 1 for char in np.unique(frame.characters))
    assert all(len(line) == 7 for line in frame.to_text().splitlines())
    assert not frame.characters.flags.writeable


def test_default_and_explicit_legacy_alias():
    assert TextRenderer().mode == 'detail'
    assert TextRenderer(mode='ascii').mode == 'simple'
    pixels = np.arange(100, dtype=np.float64).reshape(10, 10) / 99
    assert TextRenderer(mode='simple').render(pixels).to_text(ansi=True) == (
        TextRenderer(mode='ascii').render(pixels).to_text(ansi=True))
    with pytest.raises(ValueError, match='Custom density ramps'):
        TextRenderer(mode='detail', ramp=' .#')
    assert TextRenderer(mode='simple', ramp=' .#').ramp == ' .#'


@pytest.mark.parametrize('mode', ['detail', 'shape'])
@pytest.mark.parametrize('color', [False, True])
@pytest.mark.parametrize('invert', [False, True])
def test_structured_nonmutating_foreground_only_and_plain_text(mode, color, invert):
    pixels = np.random.default_rng(13).random((23, 37, 3))
    pixels[2:4, 7:9] = np.nan
    pixels[7, 2] = np.inf
    before = pixels.copy()
    renderer = TextRenderer(mode=mode, color=color, invert=invert)
    frame = renderer.render(pixels, columns=13, rows=7)
    np.testing.assert_array_equal(pixels, before)
    assert frame.background is None
    assert (frame.foreground is not None) == color
    assert '\x1b' not in frame.to_text()
    assert '48;2;' not in frame.to_text(ansi=True)
    repeated = renderer.render(pixels, columns=13, rows=7)
    assert frame.to_text(ansi=True) == repeated.to_text(ansi=True)


@pytest.mark.parametrize('mode', ['detail', 'shape'])
def test_glyphs_do_not_depend_on_color_enabled(mode):
    pixels = np.random.default_rng(6).random((35, 61, 3))
    colored = TextRenderer(mode=mode).render(pixels, columns=17, rows=9)
    mono = TextRenderer(mode=mode, color=False).render(pixels, columns=17, rows=9)
    np.testing.assert_array_equal(colored.characters, mono.characters)


@pytest.mark.parametrize('mode', ['detail', 'shape'])
@pytest.mark.parametrize('invert', [False, True])
def test_missing_samples_do_not_draw_phantom_edges(mode, invert):
    pixels = np.full((32, 32), np.nan)
    # A valid blank field surrounded by missing samples must remain blank.
    pixels[3:28, 5:19] = 1 if invert else 0
    renderer = TextRenderer(mode=mode, invert=invert)
    result = renderer.render(pixels, columns=8, rows=4)
    assert np.all(result.characters == ' ')
    assert np.all(renderer.render(np.full((8, 4), np.nan), rows=1, columns=1).characters == ' ')


def test_all_256_braille_patterns_independent_layout_oracle():
    # Use an independent literal dot order instead of trusting the production map.
    positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1)]
    samples = np.zeros((16 * 8, 16 * 4))
    for code in range(256):
        row, col = divmod(code, 16)
        for bit, (dy, dx) in enumerate(positions):
            if code & (1 << bit):
                samples[row * 8 + dy * 2:row * 8 + dy * 2 + 2,
                        col * 4 + dx * 2:col * 4 + dx * 2 + 2] = 1
    frame = TextRenderer(mode='detail', color=False).render(samples, rows=16, columns=16)
    expected = np.array([' '] + [chr(0x2800 + code) for code in range(1, 256)]).reshape(16, 16)
    np.testing.assert_array_equal(frame.characters, expected)


def test_detail_local_tone_budget_and_zero_positions():
    rows, cols = 17, 23
    random = np.random.default_rng(10)
    dots = random.random((rows, cols, 4, 2)).astype(np.float32)
    dots[..., 0, 1] = 0  # This addressed location must never acquire ink.
    samples = (dots.repeat(2, axis=2).repeat(2, axis=3)
               .transpose(0, 2, 1, 3).reshape(rows * 8, cols * 4))
    patches = cell_patches(samples, rows, cols)
    result = braille_cells(patches, np.ones_like(patches, bool), rows, cols)
    codes = np.array([0 if x == ' ' else ord(x) - 0x2800 for x in result.flat]).reshape(rows, cols)
    count = np.array([int(c).bit_count() for c in codes.flat]).reshape(rows, cols)
    assert np.max(np.abs(count - dots.sum(axis=(2, 3)))) < 1
    assert not np.any(codes & 8)


@pytest.mark.parametrize('mode', ['detail', 'shape'])
def test_equal_average_different_shape_is_not_lost(mode):
    slash = np.repeat(np.fliplr(np.eye(4)), 2, axis=0)
    backslash = slash[:, ::-1].copy()
    renderer = TextRenderer(mode=mode, color=False)
    a = renderer.render(slash, rows=1, columns=1).to_text()
    b = renderer.render(backslash, rows=1, columns=1).to_text()
    assert a != b and a != ' ' and b != ' '
    if mode == 'shape':
        assert a in ('/', '╱') and b in ('\\', '╲')
    simple = TextRenderer(mode='simple', color=False)
    assert simple.render(slash, rows=1, columns=1).to_text() == simple.render(
        backslash, rows=1, columns=1).to_text()


def test_atlas_readonly_single_column_and_finite():
    atlas = glyph_atlas()
    assert glyph_atlas() is atlas
    assert atlas.masks.shape[1:] == SAMPLE_YX
    assert len(set(atlas.characters)) == len(atlas.characters)
    assert all(wcwidth(char) == 1 and char.isprintable() for char in atlas.characters)
    for value in (atlas.characters, atlas.masks, atlas.features, atlas.tones):
        assert not value.flags.writeable
    assert np.isfinite(atlas.features).all()
    assert np.all((atlas.masks >= 0) & (atlas.masks <= 1))


def test_matching_batch_size_does_not_change_output(monkeypatch):
    import twhy.structured_rendering as structured

    image = np.random.default_rng(20).random((31, 49))
    image[1:6, 1:8] = np.nan
    renderer = TextRenderer(mode='shape')
    original = renderer.render(image, rows=9, columns=12).to_text(ansi=True)
    monkeypatch.setattr(structured, 'MATCH_BATCH', 3)
    assert original == renderer.render(image, rows=9, columns=12).to_text(ansi=True)


@pytest.mark.parametrize('mode', ['detail', 'shape'])
def test_structured_sample_cap_checked_before_resampling(mode, monkeypatch):
    import twhy.text_renderer as rendering

    def forbidden(*args, **kwargs):
        raise AssertionError('Resampling must not start')

    monkeypatch.setattr(rendering, 'resample_image', forbidden)
    with pytest.raises(ValueError, match='Structured preview'):
        TextRenderer(mode=mode).render([[0.0]], rows=1, columns=MAX_STRUCTURED_SAMPLES)


@pytest.mark.parametrize('mode', ['detail', 'shape'])
def test_new_modes_use_richer_regional_samples_and_share_preview_cache(mode, mif, monkeypatch):
    with OMETiffReader(mif[0]) as image, ViewSession(image) as session:
        calls = []
        original = image.read_region

        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(image, 'read_region', record)
        view = Viewport(0, 0, 97, 65)
        simple = TextRenderer(mode='simple')
        detail = TextRenderer(mode=mode)
        simple_plan = session.plan(view, rows=10, columns=20)
        sr, sc = detail.sample_shape(rows=10, columns=20)
        plan = session.plan(view, rows=sr, columns=sc)
        assert plan.source_samples > simple_plan.source_samples
        assert plan.source_samples <= session.max_read_pixels
        samples = session.sample(plan)
        styles = session.prepare_styles(samples)
        for rendering_mode in ('detail', 'shape'):
            other = TextRenderer(mode=rendering_mode)
            assert other.sample_shape(rows=10, columns=20) == (sr, sc)
            assert session.sample(plan) is samples
            frame = session.render(samples, styles=styles, renderer=other, rows=10, columns=20)
            assert frame.shape == (10, 20) and frame.background is None
        assert len(calls) == 1
        assert simple.sample_shape(rows=10, columns=20) == (10, 20)


@pytest.mark.parametrize('mode', ['detail', 'shape'])
def test_spatial_fit_still_obeys_read_budget(mode, mif):
    with OMETiffReader(mif[0]) as image, ViewSession(image, max_read_pixels=100) as session:
        renderer = TextRenderer(mode=mode)
        view, limited = session.fit(columns=20, rows=10, renderer=renderer, channels=(0, 2))
        assert limited
        sr, sc = renderer.sample_shape(rows=10, columns=20)
        plan = session.plan(view, rows=sr, columns=sc, channels=(0, 2))
        assert plan.source_samples <= 100
        result = session.sample(plan)
        assert result.values.shape == (2, sr, sc)


@pytest.mark.parametrize('mode', ['detail', 'shape', 'simple', 'ascii'])
@pytest.mark.parametrize('ansi', [False, True])
def test_cli_modes_real_tiff_snapshot(mode, ansi, mif):
    args = [str(mif[0]), '--snapshot', '--mode', mode, '--columns', '25', '--rows', '11']
    if ansi:
        args += ['--ansi']
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, (result.output, result.exception)
    if not ansi:
        assert '\x1b' not in result.stdout
        assert len(result.stdout.splitlines()) == 11
        assert all(len(line) == 25 for line in result.stdout.splitlines())
    assert '48;2;' not in result.stdout


def test_cli_default_is_plain_auto_and_detail_is_explicit(mif):
    args = [str(mif[0]), '--snapshot', '--columns', '25', '--rows', '11']
    runner = CliRunner()
    default = runner.invoke(main, args)
    simple = runner.invoke(main, [*args, '--mode', 'simple'])
    detail = runner.invoke(main, [*args, '--mode', 'detail'])
    assert default.exit_code == simple.exit_code == detail.exit_code == 0
    assert default.stdout == simple.stdout
    assert any(0x2801 <= ord(char) <= 0x28ff for char in detail.stdout)


def test_python311_grammar_not_a_runtime_substitute():
    root = Path(__file__).resolve().parents[1]
    for directory in ('src', 'tests', 'examples'):
        for path in (root / directory).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 11))
