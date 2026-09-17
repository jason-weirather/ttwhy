"""Real Omeify integration; source reads are never replaced by a fake backend."""

from dataclasses import replace

import numpy as np
import pytest
import tifffile
from omeify import OMETiffReader

from conftest import write_pyramid
from ttwhy import (ChannelStyle, IntensityWindow, OmeTiffSource, ReadBudgetExceeded,
                   TextRenderer, Viewport, ViewSession, resample_image)


@pytest.mark.parametrize("channel", [0, 1, 2])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_real_compressed_mif_regions(mif, channel, level):
    path, arrays = mif
    with OmeTiffSource(str(path)) as source:
        expected = arrays[level][channel, 2:13, 3:17]
        actual = source.read_region(2, 13, 3, 17, channel=channel, level=level)
        np.testing.assert_array_equal(actual, expected)
        assert source.info.channels == ("DAPI", "CD3", "panCK")
        assert source.info.pixel_size == (.5, .5, "µm")
        assert len(source.info.levels) == 3
        assert source.info.levels[2].scale_x == 97 / 25
        assert "open=True" in repr(source)
    assert not source.is_open
    assert source.info.shape == (65, 97)
    with pytest.raises(RuntimeError, match="closed"):
        source.read_region(0, 1, 0, 1)
    source.close()


@pytest.mark.parametrize("level", [0, 1, 2])
def test_real_rgb_kept_as_one_logical_channel(rgb, level):
    path, arrays = rgb
    with OmeTiffSource(path) as source:
        assert source.info.is_rgb
        assert len(source.info.channels) == 1
        np.testing.assert_array_equal(source.read_region(1, 11, 2, 16, level=level),
                                      arrays[level][1:11, 2:16])
        with pytest.raises(IndexError):
            source.select_channels([1])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
@pytest.mark.parametrize("tiled,compression,byteorder", [(True, None, ">"),
                                                        (False, "deflate", "<")])
def test_storage_and_dtype_variants(tmp_path, dtype, tiled, compression, byteorder):
    data = np.arange(32 * 47).reshape(32, 47).astype(dtype)
    path = tmp_path / "variant.ome.tif"
    write_pyramid(path, data, levels=1, tiled=tiled, compression=compression,
                  byteorder=byteorder, pixel_size=None)
    with OmeTiffSource(path) as source:
        np.testing.assert_array_equal(source.read_region(3, 27, 5, 41), data[3:27, 5:41])
        assert source.info.pixel_size is None
        assert source.info.pixel_aspect == 1
        assert not source.info.is_rgb


def test_real_rgb16(tmp_path):
    data = np.full((19, 29, 3), [1000, 32000, 65000], np.uint16)
    path = tmp_path / "rgb16.ome.tif"
    write_pyramid(path, data, rgb=True, levels=1)
    with OmeTiffSource(path) as source:
        session = ViewSession(source)
        samples = session.sample(session.plan(Viewport(0, 0, 29, 19), columns=29, rows=19))
        styles = session.prepare_styles(samples)
        assert styles[0].window == IntensityWindow(0, 65535)
        frame = session.render(samples, styles=styles, renderer=TextRenderer(mode="blocks"),
                               columns=29, rows=10)
        assert frame.shape == (10, 29)
        assert frame.foreground[0, 0, 2] == 253


def test_metadata_and_region_never_call_full_array(mif, monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("Full-image materialization is forbidden")
    monkeypatch.setattr(tifffile.TiffPage, "asarray", prohibited)
    monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", prohibited)
    monkeypatch.setattr(OMETiffReader, "asarray", prohibited, raising=False)
    with OmeTiffSource(mif[0]) as source:
        session = ViewSession(source)
        view, limited = session.fit(columns=24, rows=12, renderer=TextRenderer())
        samples = session.sample(session.plan(view, columns=24, rows=12))
        frame = session.render(samples, styles=session.prepare_styles(samples),
                               renderer=TextRenderer(), columns=24, rows=12)
        assert not limited and frame.shape == (12, 24)


def test_cache_reuse_no_extra_io_for_contrast(mif, monkeypatch):
    with OmeTiffSource(mif[0]) as source:
        calls = []
        original = source.read_region
        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)
        monkeypatch.setattr(source, "read_region", record)
        session = ViewSession(source)
        plan = session.plan(Viewport(0, 0, 97, 65), rows=20, columns=32, channels=[0, 2])
        samples = session.sample(plan)
        assert len(calls) == 2
        assert session.sample(plan) is samples
        styles = session.prepare_styles(samples)
        frame1 = session.render(samples, styles=styles, renderer=TextRenderer(),
                                columns=32, rows=20)
        adjusted = [ChannelStyle(IntensityWindow(0, 50000), style.color) for style in styles]
        frame2 = session.render(samples, styles=adjusted, renderer=TextRenderer(),
                                columns=32, rows=20)
        assert frame1.to_text() != frame2.to_text()
        assert len(calls) == 2
        assert session.cache_bytes == samples.values.nbytes
        session.clear_cache()
        assert session.cache_bytes == 0
        assert not np.shares_memory(samples.values, mif[1][0])
        with pytest.raises(ValueError):
            samples.values[:] = 0
        source.close()
        with pytest.raises(RuntimeError):
            session.sample(plan)


def test_preview_lru_and_disabled_cache(mif):
    with OmeTiffSource(mif[0]) as source:
        session = ViewSession(source, cache_mib=800 / 1024**2)
        plans = [session.plan(Viewport(x, 0, 40, 40), rows=10, columns=10) for x in (0, 5, 10)]
        first = session.sample(plans[0])
        session.sample(plans[1])
        assert session.cache_bytes == 800
        assert session.sample(plans[0]) is not first
        disabled = ViewSession(source, cache_mib=0)
        assert disabled.sample(plans[2]) is not disabled.sample(plans[2])
        assert disabled.cache_bytes == 0


def test_source_budget_checked_before_read(mif, monkeypatch):
    with OmeTiffSource(mif[0], max_read_pixels=100) as source:
        def prohibited(*args, **kwargs):
            raise AssertionError("Read must not start")
        monkeypatch.setattr(source._reader[0], "read_region", prohibited)
        with pytest.raises(ReadBudgetExceeded):
            source.read_region(0, 11, 0, 11)
        session = ViewSession(source)
        with pytest.raises(ReadBudgetExceeded):
            session.plan(Viewport(0, 0, 97, 65), rows=65, columns=97)
        view, limited = session.fit(columns=80, rows=24, renderer=TextRenderer())
        assert limited
        plan = session.plan(view, rows=24, columns=80)
        assert plan.source_samples <= 100


def test_rgb_sample_budget_counts_three(rgb):
    with OmeTiffSource(rgb[0], max_read_pixels=299) as source:
        with pytest.raises(ReadBudgetExceeded):
            source.read_region(0, 10, 0, 10)


def test_plan_forgery_and_blank_letterbox(mif):
    with OmeTiffSource(mif[0]) as source:
        session = ViewSession(source)
        plan = session.plan(Viewport(-50, -50, 100, 100), rows=10, columns=10)
        with pytest.raises(ValueError, match="ReadPlan"):
            session.sample(replace(plan, source_samples=0))
        samples = session.sample(plan)
        assert np.isnan(samples.values[0, :5]).all()
        outside = session.sample(session.plan(Viewport(-100, -100, 20, 20), rows=4, columns=4))
        assert np.isnan(outside.values).all()
        frame = session.render(outside, styles=session.prepare_styles(outside),
                               renderer=TextRenderer(invert=True), columns=4, rows=4)
        assert set(frame.to_text()) == {" ", "\n"}


def test_fractional_viewport_matches_direct_array(mif):
    with OmeTiffSource(mif[0]) as source:
        session = ViewSession(source)
        viewport = Viewport(4.2, 2.3, 23.6, 27.5)
        plan = session.plan(viewport, rows=40, columns=40, channels=(1,))
        assert plan.level.index == 0
        actual = session.sample(plan).values[0]
        reference = resample_image(mif[1][0][1], rows=40, columns=40,
                                   bounds=(4.2, 2.3, 23.6, 27.5))
        np.testing.assert_allclose(actual, reference, atol=1e-3)


@pytest.mark.parametrize("selection,error", [([], ValueError), ([0, 0], ValueError),
                                             ([3], IndexError), ([-1], IndexError),
                                             ([True], TypeError), ([.1], TypeError)])
def test_channel_validation(mif, selection, error):
    with OmeTiffSource(mif[0]) as source, pytest.raises(error):
        source.select_channels(selection)


@pytest.mark.parametrize("bounds", [(-1, 2, 0, 2), (0, 0, 0, 2), (0, 200, 0, 2),
                                   (0, 2, .1, 2)])
def test_invalid_region(mif, bounds):
    with OmeTiffSource(mif[0]) as source, pytest.raises((ValueError, TypeError)):
        source.read_region(*bounds)


def test_bad_files_and_series_close_cleanly(tmp_path, mif):
    missing = OmeTiffSource(tmp_path / "missing.tif")
    with pytest.raises((OSError, ValueError)):
        missing.open()
    assert not missing.is_open
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a TIFF")
    with pytest.raises((OSError, ValueError)):
        OmeTiffSource(bad).open()
    source = OmeTiffSource(mif[0], series=99)
    with pytest.raises((IndexError, ValueError)):
        source.open()
    assert not source.is_open
    with pytest.raises(RuntimeError):
        ViewSession(missing)


def test_multiseries_and_non_singleton_z(tmp_path):
    path = tmp_path / "series.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(np.ones((32, 32), np.uint16), metadata={"axes": "YX"}, tile=(16, 16))
        writer.write(np.full((17, 21, 3), 234, np.uint8), metadata={"axes": "YXS"},
                     photometric="rgb", tile=(16, 16))
    with OmeTiffSource(path, series=1) as source:
        assert source.info.series_count == 2
        assert source.info.is_rgb and source.info.shape == (17, 21)
        assert source.read_region(0, 1, 0, 1)[0, 0, 0] == 234
    zpath = tmp_path / "z.ome.tif"
    tifffile.imwrite(zpath, np.ones((2, 21, 32), np.uint16), metadata={"axes": "ZYX"},
                     photometric="minisblack", ome=True)
    with pytest.raises(NotImplementedError, match="singleton Z/T"):
        OmeTiffSource(zpath).open()


def test_decoder_failure_is_not_swallowed(mif, monkeypatch):
    with OmeTiffSource(mif[0]) as source:
        def fail(*args, **kwargs):
            raise OSError("fixture decode failed")
        monkeypatch.setattr(source._reader[0], "read_region", fail)
        session = ViewSession(source)
        with pytest.raises(OSError, match="fixture decode failed"):
            session.sample(session.plan(Viewport(0, 0, 10, 10), columns=10, rows=10))
        assert session.cache_bytes == 0


def test_uint32_window_precision_survives_the_entire_file_pipeline(tmp_path):
    data = np.tile(np.array([2**32 - 3, 2**32 - 2, 2**32 - 1], np.uint32), (16, 1))
    path = tmp_path / "uint32.ome.tif"
    write_pyramid(path, data, levels=1)
    with OmeTiffSource(path) as source:
        session = ViewSession(source)
        sample = session.sample(session.plan(Viewport(0, 0, 3, 16), rows=1, columns=3))
        np.testing.assert_array_equal(sample.values[0, 0], data[0])
        frame = session.render(sample, styles=[ChannelStyle(IntensityWindow(2**32 - 3, 2**32 - 1))],
                               renderer=TextRenderer(color=False, ramp=" .@"), rows=1, columns=3)
        assert frame.to_text() == " .@"


def test_ordinary_tiff_rejected_without_implicit_conversion(tmp_path):
    path = tmp_path / "not-ome.tif"
    tifffile.imwrite(path, np.ones((10, 10), np.uint8), metadata=None)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="OME metadata"):
        OmeTiffSource(path).open()
    assert path.read_bytes() == before


def test_active_decoder_pair_eviction_is_public_and_bounded(mif, monkeypatch):
    from omeify.io.channel import Channel
    cleared = []
    original = Channel.clear_cache
    def track(self):
        cleared.append(self)
        return original(self)
    monkeypatch.setattr(Channel, "clear_cache", track)
    with OmeTiffSource(mif[0]) as source:
        source.read_region(0, 2, 0, 2, channel=0)
        source.read_region(2, 4, 0, 2, channel=0)
        assert not cleared
        source.read_region(0, 2, 0, 2, channel=1)
        assert len(cleared) == 1
        source.read_region(0, 2, 0, 2, channel=1, level=1)
        assert len(cleared) == 2
        source.clear_cache()
        assert len(cleared) == 5


def test_anisotropic_physical_geometry(tmp_path):
    path = tmp_path / "anisotropic.ome.tif"
    write_pyramid(path, np.ones((40, 60), np.uint16), levels=1, pixel_size=(.5, 1.0))
    with OmeTiffSource(path) as source:
        assert source.info.pixel_aspect == 2
        view, limited = ViewSession(source).fit(columns=60, rows=40, renderer=TextRenderer())
        assert not limited
        assert view == Viewport(0, 0, 60, 40)


def test_session_lifetime_does_not_resurrect_on_source_reopen(mif):
    with OmeTiffSource(mif[0]) as source:
        session = ViewSession(source)
        plan = session.plan(Viewport(0, 0, 10, 10), columns=10, rows=10)
        session.sample(plan)
        source.close()
        source.open()
        with pytest.raises(RuntimeError, match="lifetime"):
            session.sample(plan)
        fresh = ViewSession(source)
        assert fresh.sample(fresh.plan(Viewport(0, 0, 10, 10), columns=10, rows=10)).values.size
