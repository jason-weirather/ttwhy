"""Numerical reference, array ownership, and terminal-text contracts."""

import re

import numpy as np
import pytest

from twhy import (ChannelStyle, IntensityWindow, TextImage, TextRenderer, Viewport,
                   compose_channels, estimate_window, resample_image)


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.int16, np.float32, np.float64,
])
def test_window_nonmutating(dtype):
    values = np.array([0, 10, 20, 30, 40], dtype=dtype)
    original = values.copy()
    output = IntensityWindow(10, 30).apply(values)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(output, [0, 0, .5, 1, 1])
    assert output.dtype == np.float32
    assert not np.shares_memory(values, output)


def test_uint32_narrow_window():
    values = np.array([2**32 - 3, 2**32 - 2, 2**32 - 1], np.uint32)
    np.testing.assert_array_equal(IntensityWindow(2**32 - 3, 2**32 - 1).apply(values), [0, .5, 1])


@pytest.mark.parametrize("limits", [(1, 1, 1), (2, 1, 1), (0, 1, 0), (0, 1, -1),
                                    (np.nan, 1, 1), (0, np.inf, 1), (0, 1, np.nan)])
def test_bad_windows(limits):
    with pytest.raises(ValueError):
        IntensityWindow(*limits)


def test_gamma_and_nonfinite():
    result = IntensityWindow(0, 4, gamma=2).apply([0, 1, 4, np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(result[:3], [0, .5, 1])
    assert np.isnan(result[3:]).all()


@pytest.mark.parametrize("value,expected", [(0, (0, 1)), (5, (0, 5)), (-5, (-5, -4))])
def test_constant_window(value, expected):
    window = estimate_window(np.full((7, 11), value))
    assert (window.lower, window.upper) == expected


def test_estimate_sparse_and_deterministic():
    data = np.zeros(10000)
    data[-1] = 123
    assert estimate_window(data) == IntensityWindow(0, 123)
    assert estimate_window(np.arange(10000), max_samples=37) == estimate_window(
        np.arange(10000), max_samples=37)
    assert estimate_window([np.nan, 0, 10, np.inf], lower_percentile=0,
                           upper_percentile=100) == IntensityWindow(0, 10)


@pytest.mark.parametrize("data", [[], [np.nan], [np.inf, -np.inf]])
def test_no_finite_contrast(data):
    with pytest.raises(ValueError):
        estimate_window(data)


@pytest.mark.parametrize("bounds,shape", [((0, 0, 4, 3), (2, 3)),
                                         ((-.7, .25, 5, 2.9), (4, 7)),
                                         ((10, 10, 2, 3), (3, 4)),
                                         ((.5, .5, 2, 1), (7, 9))])
def test_area_average_against_independent_rectangle_oracle(bounds, shape):
    image = np.array([[0, 1, 2, 3], [4, np.nan, 6, 7], [8, 9, 10, 11]], float)
    x, y, width, height = bounds
    expected = np.full(shape, np.nan)
    for row in range(shape[0]):
        for col in range(shape[1]):
            total, weight = 0.0, 0.0
            left, right = x + col * width / shape[1], x + (col + 1) * width / shape[1]
            top, bottom = y + row * height / shape[0], y + (row + 1) * height / shape[0]
            for sy in range(image.shape[0]):
                for sx in range(image.shape[1]):
                    area = max(0, min(sx + 1, right) - max(sx, left))
                    area *= max(0, min(sy + 1, bottom) - max(sy, top))
                    if np.isfinite(image[sy, sx]):
                        total += image[sy, sx] * area
                        weight += area
            if weight:
                expected[row, col] = total / weight
    actual = resample_image(image, rows=shape[0], columns=shape[1], bounds=bounds)
    np.testing.assert_allclose(actual, expected, atol=2e-6, equal_nan=True)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (17, 13), (3, 4)])
def test_area_constant_and_rgb(shape):
    image = np.full((3, 4, 3), [0.1, 0.5, 0.8])
    original = image.copy()
    actual = resample_image(image, rows=shape[0], columns=shape[1])
    np.testing.assert_allclose(actual, np.broadcast_to([.1, .5, .8], (*shape, 3)))
    np.testing.assert_array_equal(image, original)


def test_ascii_exact_ramp_and_inversion():
    image = np.linspace(0, 1, 10)[None, :]
    assert TextRenderer(mode="simple", color=False).render(image, columns=10, rows=1).to_text() == " .:-=+*#%@"
    assert TextRenderer(mode="ascii", invert=True).render(image, columns=10, rows=1).to_text() == "@%#*+=-:. "
    result = TextRenderer(mode="simple", invert=True).render([[np.nan, 0, 1]], columns=3, rows=1)
    assert result.to_text() == " @ "


def test_aspect_and_no_input_mutation():
    image = np.eye(50, 100)
    result = TextRenderer(color=False).render(image, columns=80)
    assert result.shape == (20, 80)
    np.testing.assert_array_equal(image, np.eye(50, 100))


def test_blocks_preserve_upper_lower_colors():
    image = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]], float)
    result = TextRenderer(mode="blocks").render(image, columns=2, rows=1)
    assert result.to_text() == "▀▀"
    np.testing.assert_array_equal(result.foreground, image[::2] * 255)
    np.testing.assert_array_equal(result.background, image[1::2] * 255)
    assert "38;2;255;0;0" in result.to_text(ansi=True)
    assert "48;2;0;0;255" in result.to_text(ansi=True)


def test_colored_ascii_hue_and_missing_cells():
    image = np.array([[[.1, .2, .5], [np.nan, np.nan, np.nan]]])
    frame = TextRenderer(mode="simple").render(image, columns=2, rows=1)
    np.testing.assert_array_equal(frame.foreground[0, 0], [51, 102, 255])
    assert frame.to_text()[1] == " "
    blocks = TextRenderer(mode="blocks").render(np.full((2, 1), np.nan), columns=1, rows=1)
    assert blocks.to_text() == " "
    np.testing.assert_array_equal(blocks.background, 0)


def test_text_image_owns_arrays_and_run_serialization():
    chars = np.array([list("abba"), list(" q  ")])
    colors = np.full((2, 4, 3), 255, np.uint8)
    colors[0, 2:] = [30, 60, 90]
    frame = TextImage(chars, foreground=colors)
    chars[0, 0] = "x"
    colors[:] = 0
    assert str(frame) == "abba\n q  "
    assert len(list(frame.iter_runs(0))) == 2
    assert re.sub(r"\x1b\[[0-9;]+m", "", frame.to_text(ansi=True)) == frame.to_text()
    assert all(line.endswith("\x1b[0m") for line in frame.to_text(ansi=True).splitlines())
    with pytest.raises(ValueError):
        frame.characters[0, 0] = "x"
    assert "rows=2" in repr(frame)
    assert frame.nbytes == frame.characters.nbytes + frame.foreground.nbytes
    with pytest.raises(IndexError):
        list(frame.iter_runs(3))


@pytest.mark.parametrize("character", ["\x1b", "\n", "界", "ab", "", "\u0301"])
def test_terminal_control_and_width_rejection(character):
    with pytest.raises(ValueError):
        TextImage(np.array([[character]]))


@pytest.mark.parametrize("kwargs", [{"mode": "wrong"}, {"mode": "blocks", "color": False},
                                    {"ramp": "x"}, {"ramp": "x\x1b"},
                                    {"cell_aspect": 0}, {"cell_aspect": np.inf}])
def test_bad_renderer_options(kwargs):
    with pytest.raises(ValueError):
        TextRenderer(**kwargs)


@pytest.mark.parametrize("image", [np.empty((0, 5)), np.ones((3, 4, 4)), np.ones(10),
                                   np.array([["a"]]), np.array([[1j]])])
def test_unsupported_array_shapes_and_types(image):
    with pytest.raises((ValueError, TypeError)):
        TextRenderer().render(image)


def test_raw_values_require_explicit_window():
    with pytest.raises(ValueError, match="Normalize"):
        TextRenderer().render([[0, 255]])


def test_composite_independent_limits_clipping_and_nan():
    raw = np.array([[[5, 20, np.nan]], [[25, 50, np.nan]]], float)
    styles = [ChannelStyle(IntensityWindow(0, 10), (1, 0, 0)),
              ChannelStyle(IntensityWindow(0, 100), (0, 1, 0))]
    composed = compose_channels(raw, styles)
    np.testing.assert_allclose(composed[0, :2], [[.5, .25, 0], [1, .5, 0]])
    assert np.isnan(composed[0, 2]).all()
    assert raw[0, 0, 1] == 20
    with pytest.raises(ValueError):
        compose_channels(raw, styles[:1])


def test_style_owns_color_sequence():
    color = [1, 0, .5]
    style = ChannelStyle(IntensityWindow(0, 1), color)
    color[0] = 0
    assert style.color == (1, 0, .5)


def test_camera_anchor_pan_and_anisotropic_fit():
    view = Viewport(10, 20, 100, 200)
    zoom = view.zoom(2, anchor=(.2, .7))
    assert zoom.x + zoom.width * .2 == view.x + view.width * .2
    assert zoom.y + zoom.height * .7 == view.y + view.height * .7
    assert view.pan(5, -2) == Viewport(15, 18, 100, 200)
    assert view == Viewport(10, 20, 100, 200)
    fit = Viewport.fit((100, 200), columns=100, rows=25, cell_aspect=2, pixel_aspect=2)
    assert fit.width / fit.height == 4
    assert fit.width >= 200 and fit.height >= 100


@pytest.mark.parametrize("kwargs", [{"rows": 0, "columns": 2}, {"rows": True, "columns": 2},
                                    {"rows": 1, "columns": -1},
                                    {"rows": 1, "columns": 2, "bounds": (0, 0, -1, 2)}])
def test_invalid_resampling(kwargs):
    with pytest.raises((ValueError, TypeError)):
        resample_image(np.ones((3, 4)), **kwargs)
