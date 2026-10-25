"""Current Image -> plan -> sample -> render contract, using real Omeify sources."""

from dataclasses import replace

import numpy as np
import pytest
import tifffile
from omeify import ImageLevel, ImageMetadata, ImageSource, LabelImage, MultichannelImage
from omeify import OMETiffReader, PixelSize, RGBImage

from conftest import write_pyramid
from twhy import ChannelStyle, IntensityWindow, ReadBudgetExceeded, TextRenderer
from twhy import Viewport, ViewSession, resample_image


@pytest.mark.parametrize("fixture_name", ["mif", "rgb"])
def test_file_and_array_share_the_render_path(request, fixture_name):
    path, arrays = request.getfixturevalue(fixture_name)
    with OMETiffReader(path) as file_image:
        if fixture_name == "rgb":
            array_image = RGBImage.from_array(
                arrays[0], pixel_size=file_image.pixel_size,
                level_arrays=arrays[1:], levels=file_image.levels,
            )
        else:
            array_image = MultichannelImage.from_array(
                arrays[0], axes="CYX", channel_names=file_image.channel_names,
                pixel_size=file_image.pixel_size,
                level_arrays=arrays[1:], levels=file_image.levels,
            )
        with (array_image, ViewSession(file_image) as file_view,
              ViewSession(array_image) as array_view):
            renderer = TextRenderer(color=True)
            view, limited = file_view.fit(columns=25, rows=11, renderer=renderer)
            assert not limited
            plans = [s.plan(view, columns=25, rows=11) for s in (file_view, array_view)]
            samples = [s.sample(plan) for s, plan in zip((file_view, array_view), plans)]
            np.testing.assert_array_equal(samples[0].values, samples[1].values)
            frames = [s.render(v, styles=s.prepare_styles(v), renderer=renderer,
                               columns=25, rows=11)
                      for s, v in zip((file_view, array_view), samples)]
            assert frames[0].to_text(ansi=True) == frames[1].to_text(ansi=True)
            assert all(not p.level.approximate for p in plans)
        assert file_image.is_open


@pytest.mark.parametrize("dtype,tiled,byteorder", [
    (np.uint16, True, ">"), (np.int16, False, "<"),
    (np.float32, True, "<"), (np.float64, False, ">"),
])
def test_native_dtype_storage_and_raw_values(tmp_path, dtype, tiled, byteorder):
    data = np.arange(19 * 23).reshape(19, 23).astype(dtype) - 100
    path = tmp_path / "values.ome.tif"
    write_pyramid(path, data, levels=1, tiled=tiled, byteorder=byteorder, pixel_size=None)
    with OMETiffReader(path) as image, ViewSession(image) as session:
        samples = session.sample(session.plan(Viewport(2, 3, 10, 9), columns=10, rows=9))
        np.testing.assert_array_equal(samples.values[0], data[3:12, 2:12])
        assert session.info.pixel_size is None and session.info.pixel_aspect == 1


def test_raw_uint32_precision_and_rgb_logical_channel():
    raw = np.arange(100, dtype=np.uint32).reshape(10, 10) + 2**31
    with MultichannelImage.from_array(raw, axes="YX") as image, ViewSession(image) as session:
        samples = session.sample(session.plan(Viewport(0, 0, 10, 10), rows=10, columns=10))
        np.testing.assert_array_equal(samples.values[0], raw)
        frame = session.render(samples, styles=[ChannelStyle(IntensityWindow(2**31, 2**31+99))],
                               renderer=TextRenderer(color=False), rows=10, columns=10)
        assert len(set(frame.to_text())) > 5
    rgb = np.full((19, 29, 3), [1000, 32000, 65000], np.uint16)
    with RGBImage.from_array(rgb) as image, ViewSession(image) as session:
        samples = session.sample(session.plan(Viewport(0, 0, 29, 19), columns=29, rows=19))
        assert samples.is_rgb and len(session.info.channels) == 1
        styles = session.prepare_styles(samples)
        assert styles[0].window == IntensityWindow(0, 65535)
        with pytest.raises(IndexError):
            session.plan(Viewport(0, 0, 1, 1), rows=1, columns=1, channels=[1])


def test_metadata_and_region_never_materialize_whole_image(mif, monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("Whole-image materialization is forbidden")
    monkeypatch.setattr(tifffile.TiffPage, "asarray", prohibited)
    monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", prohibited)
    monkeypatch.setattr(OMETiffReader, "asarray", prohibited)
    with OMETiffReader(mif[0]) as image, ViewSession(image) as session:
        viewport, limited = session.fit(columns=24, rows=12, renderer=TextRenderer())
        samples = session.sample(session.plan(viewport, columns=24, rows=12))
        assert not limited and np.isfinite(samples.values).any()


def test_contrast_cache_lru_and_borrowed_lifetime(mif, monkeypatch):
    with OMETiffReader(mif[0]) as image, ViewSession(image, cache_mib=800/1024**2) as session:
        calls = []
        original = image.read_region
        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)
        monkeypatch.setattr(image, "read_region", record)
        plan = session.plan(Viewport(0, 0, 40, 40), rows=10, columns=10)
        samples = session.sample(plan)
        assert len(calls) == 1 and session.sample(plan) is samples
        styles = session.prepare_styles(samples)
        renderer = TextRenderer(color=False)
        first = session.render(samples, styles=styles, renderer=renderer, columns=10, rows=10)
        second = session.render(samples, styles=[ChannelStyle(IntensityWindow(0, 50000))],
                                renderer=renderer, columns=10, rows=10)
        assert first.to_text() != second.to_text() and len(calls) == 1
        next_plan = session.plan(Viewport(5, 0, 40, 40), rows=10, columns=10)
        session.sample(next_plan)
        assert session.cache_bytes == 800 and session.sample(plan) is not samples
        image.close()
        image.open()
        with pytest.raises(RuntimeError, match="lifetime"):
            session.sample(plan)
        with ViewSession(image) as fresh:
            fresh.sample(fresh.plan(Viewport(0, 0, 10, 10), rows=10, columns=10))
        assert image.is_open
    assert session.cache_bytes == 0 and not session.is_open
    assert samples.values.shape == (1, 10, 10) and not samples.values.flags.writeable


class Coordinates(ImageSource):
    def __init__(self):
        super().__init__(ImageMetadata(axes="CYX", shape=(2, 100_000, 120_000),
                                      dtype=np.dtype("float32"), channel_names=("A", "B")))
        self.requests = []
    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        self.requests.append((y0, y1, x0, x1, channels))
        y, x = np.ogrid[y0:y1, x0:x1]
        return np.stack([x + y * 2 + i * 100 for i in channels]).astype(np.float32)


def test_large_procedural_source_is_bounded_and_channel_aware():
    provider = Coordinates()
    with MultichannelImage(provider) as image, ViewSession(image, max_read_pixels=1024) as session:
        assert not provider.requests
        viewport = Viewport(50_000, 20_000, 16, 16)
        plan = session.plan(viewport, rows=16, columns=16, channels=[1, 0])
        values = session.sample(plan).values
        y, x = np.ogrid[20_000:20_016, 50_000:50_016]
        np.testing.assert_array_equal(values[0], x + y * 2 + 100)
        assert len(provider.requests) == 2
        with pytest.raises(ReadBudgetExceeded):
            session.plan(Viewport(0, 0, 1000, 1000), rows=10, columns=10)
        assert len(provider.requests) == 2
        _, limited = session.fit(rows=10, columns=20, renderer=TextRenderer())
        assert limited and len(provider.requests) == 2


def test_rgb_read_budget_counts_all_samples(rgb, monkeypatch):
    with OMETiffReader(rgb[0]) as image, ViewSession(image, max_read_pixels=299) as session:
        def forbidden(*args, **kwargs):
            raise AssertionError("Read must not start")
        monkeypatch.setattr(image, "read_region", forbidden)
        with pytest.raises(ReadBudgetExceeded):
            session.plan(Viewport(0, 0, 10, 10), rows=10, columns=10)


def test_fractional_viewport_and_letterbox(mif):
    with OMETiffReader(mif[0]) as image, ViewSession(image) as session:
        viewport = Viewport(4.2, 2.3, 23.6, 27.5)
        plan = session.plan(viewport, rows=40, columns=40, channels=[1])
        reference = resample_image(mif[1][0][1], rows=40, columns=40,
                                   bounds=(4.2, 2.3, 23.6, 27.5))
        np.testing.assert_allclose(session.sample(plan).values[0], reference, rtol=1e-12)
        with pytest.raises(ValueError, match="ReadPlan"):
            session.sample(replace(plan, source_samples=0))
        outside = session.sample(session.plan(Viewport(-100, -100, 20, 20), rows=4, columns=4))
        assert np.isnan(outside.values).all()


def test_declared_scales_do_not_use_rounded_extent_ratios():
    raw = np.arange(5 * 7).reshape(5, 7).astype(np.float32)
    size = PixelSize(.5, 1, "µm")
    levels = (ImageLevel(0, "YX", raw.shape, size),
              ImageLevel(1, "YX", (3, 4), size.scaled(2), (2, 2)))
    with (MultichannelImage.from_array(raw, axes="YX", pixel_size=size,
                                      level_arrays=[raw[::2, ::2]], levels=levels) as image,
          ViewSession(image) as session):
        assert session.info.pixel_aspect == 2
        plan = session.plan(Viewport(0, 0, 8, 6), rows=2, columns=2)
        assert plan.level.scale_x == 2 and not plan.level.approximate
        # Pixel value 4 covers base x=[4,6), while value 6 covers only x=[6,7).
        last = session.sample(plan).values[0, -1, -1]
        coarse = raw[::2, ::2]
        expected = (coarse[1, 2] * 2 + coarse[1, 3]
                    + coarse[2, 2] * 2 + coarse[2, 3]) / 6
        assert last == pytest.approx(expected)
        fully_outside = session.sample(session.plan(Viewport(7, 0, 2, 6), rows=3, columns=1))
        assert np.isnan(fully_outside.values).all()


def test_unknown_level_scale_is_explicit_display_approximation(tmp_path):
    path = tmp_path / "unknown-scale.ome.tif"
    write_pyramid(path, np.ones((65, 97), np.uint16), declare_scales=False)
    with OMETiffReader(path) as image, ViewSession(image) as session:
        assert image.levels[2].downsample_yx is None
        assert session.info.levels[2].approximate
        assert session.info.levels[2].scale_x == 97 / 25
        assert session.info.pixel_size == (.5, .5, "µm")


@pytest.mark.parametrize("selection,error", [([], ValueError), ([0, 0], ValueError),
                                             ([True], TypeError), ([1.2], TypeError),
                                             ([99], IndexError)])
def test_invalid_display_selection(mif, selection, error):
    with OMETiffReader(mif[0]) as image, ViewSession(image) as session, pytest.raises(error):
        session.plan(Viewport(0, 0, 10, 10), rows=10, columns=10, channels=selection)


def test_failure_does_not_cache_partial_results(mif, monkeypatch):
    with OMETiffReader(mif[0]) as image, ViewSession(image) as session:
        def fail(*args, **kwargs):
            raise OSError("intentional read failure")
        monkeypatch.setattr(image, "read_region", fail)
        with pytest.raises(OSError, match="intentional"):
            session.sample(session.plan(Viewport(0, 0, 10, 10), rows=10, columns=10))
        assert session.cache_bytes == 0


def test_no_silent_categorical_or_closed_image_reinterpretation():
    image = RGBImage.from_array(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(RuntimeError, match="open"):
        ViewSession(image)
    with LabelImage.from_array(np.zeros((10, 10), np.uint32)) as labels:
        with pytest.raises(ValueError, match="Categorical"):
            ViewSession(labels)
        assert labels.is_open
