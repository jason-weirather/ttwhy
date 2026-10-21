"""Whole-slide zoom-out must not be trapped by an oversized stored pyramid level."""

import asyncio
import time
from dataclasses import replace

import numpy as np
import pytest
from omeify import ImageLevel, ImageMetadata, ImageSource, RGBImage, MultichannelImage
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from ttwhy import ReadBudgetExceeded, TextRenderer, Viewport, ViewSession, open_image
from ttwhy.viewer import Viewer
from ttwhy.viewer.rendering import FrameRequest, make_frame

from conftest import write_pyramid
from test_svs import synthetic_svs


class GradientSlide(ImageSource):
    """A gigantic portrait or landscape slide without allocating its pixels."""

    def __init__(self, height, width):
        self.shapes = ((height, width), (height // 16, width // 16))
        levels = tuple(ImageLevel(index, "YXS", (*shape, 3), downsample_yx=(16**index,) * 2)
                       for index, shape in enumerate(self.shapes))
        super().__init__(ImageMetadata(
            axes="YXS", shape=(*self.shapes[0], 3), dtype=np.dtype("uint8"),
            image_type="rgb", channel_names=("RGB",), levels=levels,
        ))
        self.reads = []

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        self.reads.append((level, y0, y1, x0, x1, channels))
        height, width = self.shapes[level]
        y, x = np.ogrid[y0:y1, x0:x1]
        rgb = np.empty((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        rgb[..., 0] = np.rint(x * 240 / width)
        rgb[..., 1] = np.rint(y * 240 / height)
        rgb[..., 2] = 100
        return rgb


@pytest.mark.parametrize("height,width", [(60_000, 10_000), (10_000, 60_000)])
def test_full_image_budgeted_overview_preserves_shape_and_letterbox(height, width):
    source = GradientSlide(height, width)
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=12_000) as session:
        renderer = TextRenderer(mode="blocks")
        initial, limited = session.fit(rows=26, columns=68, renderer=renderer)
        assert limited  # Original crop-on-open behavior stays the same.
        assert initial.width < width or initial.height < height
        whole = Viewport.fit((height, width), rows=26, columns=68,
                             cell_aspect=renderer.cell_aspect)
        shape = renderer.sample_shape(rows=26, columns=68)
        with pytest.raises(ReadBudgetExceeded):
            session.plan(whole, rows=shape[0], columns=shape[1])
        plan = session.plan(whole, rows=shape[0], columns=shape[1],
                            allow_sampled_overview=True)
        overview = plan.sampled_overview
        assert overview is not None and plan.level.index == 1
        assert plan.source_samples <= session.max_read_pixels
        assert 1 <= len(overview.regions) <= 256
        gy, gx = overview.grid_shape
        assert (gy > gx) if height > width else (gx > gy)
        assert not source.reads  # Planning never fetches pixels.

        samples = session.sample(plan)
        assert len(source.reads) == len(overview.regions)
        assert sum((y1 - y0) * (x1 - x0) * 3
                   for _, y0, y1, x0, x1, _ in source.reads) == plan.source_samples
        assert all(0 <= y0 < y1 <= source.shapes[level][0] and
                   0 <= x0 < x1 <= source.shapes[level][1]
                   for level, y0, y1, x0, x1, _ in source.reads)
        xs = whole.x + (np.arange(shape[1]) + .5) * whole.width / shape[1]
        ys = whole.y + (np.arange(shape[0]) + .5) * whole.height / shape[0]
        inside = (ys[:, None] >= 0) & (ys[:, None] < height) & (
            (xs[None, :] >= 0) & (xs[None, :] < width))
        actual_valid = np.isfinite(samples.values).all(axis=-1)
        np.testing.assert_array_equal(actual_valid, inside)
        assert 0 < inside.sum() < inside.size  # The slide has visible padding.
        assert np.nanmax(np.abs(samples.values[..., 2] - 100)) < 1e-10
        # Even the deliberately sparse navigational view retains orientation:
        # horizontal gradient increases rightwards; vertical gradient downwards.
        x_error = samples.values[..., 0] - 240 * xs[None, :] / width
        y_error = samples.values[..., 1] - 240 * ys[:, None] / height
        assert np.abs(x_error[inside]).mean() < 12
        assert np.abs(y_error[inside]).mean() < 12
        frame = session.render(samples, styles=session.prepare_styles(samples),
                               renderer=renderer, rows=26, columns=68)
        assert frame.shape == (26, 68)
        assert " " in frame.to_text() and "▀" in frame.to_text()
        assert session.sample(plan) is samples  # Cache hits don't reread every patch.
        with pytest.raises(ValueError, match="ReadPlan"):
            session.sample(replace(plan, source_samples=plan.source_samples - 1))


def test_precise_reading_unmodified_when_native_rectangle_fits():
    source = GradientSlide(60_000, 10_000)
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=20_000) as session:
        viewport = Viewport(100, 200, 50, 40)
        exact = session.plan(viewport, rows=20, columns=25)
        automatic = session.plan(viewport, rows=20, columns=25,
                                 allow_sampled_overview=True)
        assert exact == automatic and automatic.sampled_overview is None
        samples = session.sample(automatic)
        assert len(source.reads) == 1
        assert samples.plan == exact
        assert samples.values.shape == (20, 25, 3)
        assert np.isfinite(samples.values).all()


@pytest.mark.parametrize("suffix", ["svs", "ome.tif"])
def test_real_tiff_overview_fallback_uses_bounded_native_regions(tmp_path, suffix):
    y, x = np.mgrid[:128, :160]
    rgb = np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8)
    path = tmp_path / f"test.{suffix}"
    if suffix == "svs":
        synthetic_svs(path)
    else:
        write_pyramid(path, rgb, rgb=True)
    with open_image(path) as image, ViewSession(image, max_read_pixels=500) as session:
        renderer = TextRenderer(mode="blocks")
        full = Viewport.fit(session.info.shape, columns=24, rows=12)
        rows, columns = renderer.sample_shape(columns=24, rows=12)
        with pytest.raises(ReadBudgetExceeded):
            session.plan(full, rows=rows, columns=columns)
        plan = session.plan(full, rows=rows, columns=columns,
                            allow_sampled_overview=True)
        assert plan.sampled_overview is not None
        assert plan.source_samples <= 500
        frame = make_frame(session, FrameRequest(1, full, 24, 12, (0,), (None,), renderer))
        assert frame.samples.plan.sampled_overview is not None
        assert frame.frame.shape == (12, 24)
        assert np.isfinite(frame.samples.values).any()


def test_no_pyramid_and_multichannel_use_same_fallback():
    y, x = np.mgrid[:120, :50]
    channels = np.stack((y.astype(np.uint16), (x * 5).astype(np.uint16)))
    with (MultichannelImage.from_array(channels, axes="CYX") as image,
          ViewSession(image, max_read_pixels=128) as session):
        full = Viewport.fit((120, 50), columns=30, rows=10)
        with pytest.raises(ReadBudgetExceeded):
            session.plan(full, rows=10, columns=30, channels=[0, 1])
        plan = session.plan(full, rows=10, columns=30, channels=[0, 1],
                            allow_sampled_overview=True)
        assert plan.level.index == 0 and plan.sampled_overview is not None
        assert plan.source_samples <= 128
        values = session.sample(plan).values
        assert values.shape == (2, 10, 30)
        assert np.isfinite(values).any(axis=(1, 2)).all()
        assert (np.isnan(values[0]) == np.isnan(values[1])).all()


def test_one_rgb_sample_cannot_fit_below_three_component_budget():
    source = GradientSlide(60_000, 10_000)
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=2) as session:
        full = Viewport.fit((60_000, 10_000), columns=68, rows=26)
        with pytest.raises(ReadBudgetExceeded):
            session.plan(full, rows=52, columns=68, allow_sampled_overview=True)
        assert not source.reads


class SizedOutput(DummyOutput):
    def get_size(self):
        return Size(rows=30, columns=100)


@pytest.mark.asyncio
async def test_actual_viewer_minus_key_shows_entire_portrait_slide():
    source = GradientSlide(60_000, 10_000)
    with RGBImage(source) as image, ViewSession(
        image, overview_cache_mib=0,  # Exercise the fallback with cache preparation disabled.
    ) as session, create_pipe_input() as pipe:
        viewer = Viewer(session, renderer=TextRenderer(mode="blocks"))
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            start = time.monotonic()
            while viewer.frame is None or viewer.busy:
                if task.done():
                    await task
                    raise AssertionError("Viewer closed before the initial frame")
                assert time.monotonic() - start < 10
                await asyncio.sleep(.03)
            initial = viewer.viewport
            assert initial.height == session.info.shape[0]
            assert viewer.samples.plan.sampled_overview is not None
            # The opening view is already whole-image; '-' can zoom out farther.
            pipe.send_text("-" * 2)
            start = time.monotonic()
            while (viewer.busy or viewer.displayed_generation != viewer.generation
                   or viewer.viewport.height <= initial.height):
                if task.done():
                    await task
                    raise AssertionError("Zoom-out crashed the viewer")
                assert time.monotonic() - start < 12
                await asyncio.sleep(.04)
            assert viewer.viewport.x <= 0
            assert viewer.viewport.y <= 0
            assert viewer.viewport.x + viewer.viewport.width >= 10_000
            assert viewer.viewport.y + viewer.viewport.height >= 60_000
            assert viewer.samples.plan.sampled_overview is not None
            assert "sampled overview" in viewer._status()
            assert viewer.frame.shape == (26, 68)
            # Zooming back in recovers the ordinary faithful regional path.
            pipe.send_text("+" * 10)
            start = time.monotonic()
            while (viewer.busy or viewer.displayed_generation != viewer.generation
                   or viewer.samples.plan.sampled_overview is not None):
                if task.done():
                    await task
                    raise AssertionError("Zoom-in crashed the viewer")
                assert time.monotonic() - start < 12
                await asyncio.sleep(.04)
            assert "sampled overview" not in viewer._status()
        finally:
            pipe.send_bytes(b"\x11")  # Ctrl-Q
            await asyncio.wait_for(task, 12)


@pytest.mark.asyncio
async def test_fit_shortcut_restores_full_extent_after_zoom_in():
    source = GradientSlide(60_000, 10_000)
    with RGBImage(source) as image, ViewSession(
        image, overview_cache_mib=0,  # Exercise the fallback with cache preparation disabled.
    ) as session, create_pipe_input() as pipe:
        viewer = Viewer(session, renderer=TextRenderer(mode="blocks"))
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            start = time.monotonic()
            while viewer.frame is None or viewer.busy:
                if task.done():
                    await task
                    raise AssertionError("Viewer closed before initial display")
                assert time.monotonic() - start < 10
                await asyncio.sleep(.03)
            assert viewer.viewport.y <= 0
            assert viewer.viewport.y + viewer.viewport.height >= 60_000
            assert viewer.samples.plan.sampled_overview is not None
            pipe.send_text("+")
            start = time.monotonic()
            while (viewer.busy or viewer.displayed_generation != viewer.generation
                   or viewer.viewport.height >= 60_000):
                if task.done():
                    await task
                    raise AssertionError("Zoom-in crashed the viewer")
                assert time.monotonic() - start < 12
                await asyncio.sleep(.04)
            pipe.send_text("r")
            start = time.monotonic()
            while (viewer.busy or viewer.displayed_generation != viewer.generation
                   or viewer.viewport.y > 0):
                if task.done():
                    await task
                    raise AssertionError("Fit shortcut crashed the viewer")
                assert time.monotonic() - start < 12
                await asyncio.sleep(.04)
            assert viewer.viewport.y + viewer.viewport.height >= 60_000
            assert viewer.viewport.x <= 0
            assert viewer.viewport.x + viewer.viewport.width >= 10_000
            assert viewer.samples.plan.sampled_overview is not None
            assert "Fit slide" in viewer.message
        finally:
            pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 12)
