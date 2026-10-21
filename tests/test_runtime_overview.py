"""Complete native coverage, bounded memory/I/O, and foreground-first scheduling."""

import asyncio
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from omeify import ImageLevel, MultichannelImage, PixelSize, RGBImage
from prompt_toolkit.input import create_pipe_input

from ttwhy import ReadBudgetExceeded, TextRenderer, Viewport, ViewSession, open_image
from ttwhy.runtime_overview import OverviewBuilder, OverviewSpec
from ttwhy.view_info import PyramidLevel
from ttwhy.viewer import Viewer
from ttwhy.viewer.rendering import FrameRequest, make_frame
from conftest import write_pyramid
from test_sampled_overview import GradientSlide, SizedOutput
from test_svs import synthetic_svs


def drain(session, channels=(0,)):
    for _ in range(20_000):
        progress = session.step_overview(channels)
        if not progress.pending:
            return
    pytest.fail("Overview never completed")


def oracle(pixels, factor, coverage):
    """Independent per-pixel reduction, including partial native edge support."""
    h, w = pixels.shape[:2]
    shape = ((h + factor - 1) // factor, (w + factor - 1) // factor, *pixels.shape[2:])
    numerator = np.zeros(shape)
    support = np.zeros(shape)
    area = np.zeros(shape[:2])
    for y in range(h):
        for x in range(w):
            weight = min(1, max(0, coverage[0] - y)) * min(1, max(0, coverage[1] - x))
            target = y // factor, x // factor
            finite = np.isfinite(pixels[y, x])
            numerator[target] += np.where(finite, pixels[y, x], 0).astype(float) * weight
            support[target] += finite * weight
            area[target] += weight
    if pixels.ndim == 3:
        area = area[..., None]
    np.divide(numerator, area, out=numerator, where=area > 0)
    np.divide(support, area, out=support, where=area > 0)
    return numerator, support


@pytest.mark.parametrize("shape", [(9, 13), (1, 23), (17, 1)])
@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("budget", [3, 77, 600])
def test_chunked_reduction_matches_independent_weighted_oracle(shape, rgb, budget):
    components = 3 if rgb else 1
    pixels = np.arange(np.prod(shape) * components, dtype=np.float64).reshape(
        (*shape, 3) if rgb else shape)
    pixels.flat[0] = np.nan
    pixels.flat[-1] = np.inf
    factor = 4
    coverage = shape[0] - .25, shape[1] - .5
    spec = OverviewSpec(PyramidLevel(2, *shape, 2, 2), factor,
                        tuple((n + factor - 1) // factor for n in shape), coverage, components)
    builder = OverviewBuilder(spec, 0, budget)
    visits = np.zeros(shape, int)

    def read(channel, level, bounds):
        assert channel == 0 and level == 2
        y0, y1, x0, x1 = bounds
        assert (y1 - y0) * (x1 - x0) * components <= budget
        visits[y0:y1, x0:x1] += 1
        return pixels[y0:y1, x0:x1]

    with pytest.raises(RuntimeError, match="incomplete"):
        builder.finish()
    while not builder.done:
        builder.step(read)
    result = builder.finish()
    np.testing.assert_array_equal(visits, 1)
    weighted, support = oracle(pixels, factor, coverage)
    np.testing.assert_allclose(result.weighted, weighted, rtol=1e-14)
    np.testing.assert_allclose(result.support, support, rtol=1e-14)
    assert not result.weighted.flags.writeable and not result.support.flags.writeable
    # Combining complete coarse cells preserves the full finite native mean.
    whole = Viewport(0, 0, coverage[1] * 2, coverage[0] * 2)
    value = result.sample(whole, rows=1, columns=1)[0, 0]
    wy = np.clip(coverage[0] - np.arange(shape[0]), 0, 1)
    wx = np.clip(coverage[1] - np.arange(shape[1]), 0, 1)
    weight = wy[:, None] * wx[None, :]
    if rgb:
        weight = weight[..., None]
    valid = np.isfinite(pixels)
    expected = (np.where(valid, pixels, 0) * weight).sum(axis=(0, 1)) / (
        valid * weight).sum(axis=(0, 1))
    np.testing.assert_allclose(value, expected, rtol=1e-14)
    assert builder.processed == pixels.size


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float64])
def test_cache_raw_precision_and_no_rereads_across_cameras(dtype):
    pixels = np.arange(65 * 41).reshape(65, 41).astype(dtype)
    if dtype == np.uint32:
        pixels += 2**31
    with MultichannelImage.from_array(pixels, axes="YX") as image, ViewSession(
        image, max_read_pixels=128, overview_cache_mib=1,
    ) as session:
        view = Viewport(0, 0, 41, 65)
        plan = session.plan(view, rows=1, columns=1, allow_sampled_overview=True)
        initial = session.sample(plan)
        assert not initial.cached_overview
        drain(session)
        cached = session.sample(plan)
        assert cached is not initial and cached.cached_overview
        assert cached.values[0, 0, 0] == pytest.approx(pixels.mean(dtype=np.float64), abs=1e-7)
        assert session.sample(plan) is cached
        # Strict source budgeting is still explicit; cached display reductions
        # are not smuggled into Omeify's native level descriptors.
        with pytest.raises(ReadBudgetExceeded):
            session.plan(view, rows=1, columns=1)
        assert len(session.info.levels) == len(image.levels) == 1
        assert session.overview_cache_bytes <= 1024**2
        with pytest.MonkeyPatch.context() as patch:
            def forbidden(*args, **kwargs):
                pytest.fail("Cached overview should not access the source")
            patch.setattr(image, "read_region", forbidden)
            for camera in (view.zoom(.8), view.pan(2.25, -3.5), view):
                result = session.sample(session.plan(camera, rows=7, columns=9,
                                                     allow_sampled_overview=True))
                assert result.cached_overview
        # Native detail always wins once its actual source rectangle fits.
        close = session.sample(session.plan(Viewport(0, 0, 5, 5), rows=5, columns=5,
                                            allow_sampled_overview=True))
        assert not close.cached_overview and close.plan.sampled_overview is None
        np.testing.assert_array_equal(close.values[0], pixels[:5, :5])
        session.clear_cache()
        assert session.overview_cache_bytes == session.cache_bytes == 0


def test_sparse_missed_structure_contributes_to_complete_cache():
    pixels = np.zeros((128, 128), np.uint16)
    pixels[0, 0] = 16384
    with MultichannelImage.from_array(pixels, axes="YX") as image, ViewSession(
        image, max_read_pixels=256,
    ) as session:
        plan = session.plan(Viewport(0, 0, 128, 128), rows=1, columns=1,
                            allow_sampled_overview=True)
        assert session.sample(plan).values[0, 0, 0] == 0  # Sparse grid misses this corner.
        drain(session)
        assert session.sample(plan).values[0, 0, 0] == 1  # Complete box mean includes it.


def test_multi_channel_lru_cap_includes_builder_and_retains_overlap():
    pixels = np.stack([np.full((65, 47), c, np.uint16) for c in range(8)])
    cap = 32768
    with MultichannelImage.from_array(pixels, axes="CYX") as image, ViewSession(
        image, max_read_pixels=512, overview_cache_mib=cap / 1024**2,
    ) as session:
        for channels in ((0, 1), (1, 2), (4, 5), (6, 7), (0, 1)):
            for _ in range(1000):
                progress = session.step_overview(channels)
                assert session.overview_cache_bytes <= cap
                if not progress.pending:
                    break
            else:
                pytest.fail("Multichannel cache thrashed without completing")
            plan = session.plan(Viewport(0, 0, 47, 65), rows=3, columns=3,
                                channels=channels, allow_sampled_overview=True)
            result = session.sample(plan)
            assert result.cached_overview
            for index, c in enumerate(channels):
                np.testing.assert_allclose(result.values[index], c)


@pytest.mark.parametrize("options", [dict(overview_cache_mib=0),
                                     dict(overview_max_source_pixels=0),
                                     dict(overview_max_source_pixels=100)])
def test_disabled_or_excessive_scan_never_reads(options):
    source = GradientSlide(60_000, 10_000)
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=128, **options) as session:
        assert not session.step_overview().pending
        assert not source.reads
        assert session.overview_cache_bytes == 0


def test_no_build_when_native_coarsest_is_cheap_even_if_desired_detail_is_not():
    source = GradientSlide(1600, 1600)
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=30_000) as session:
        assert not session.step_overview().pending
        assert not source.reads
        full = Viewport(0, 0, 1600, 1600)
        plan = session.plan(full, rows=200, columns=200, allow_sampled_overview=True)
        assert plan.level.index == 1 and plan.sampled_overview is None
        assert not session.sample(plan).cached_overview


def test_cancel_discard_and_source_lifetime_invalidate_cache():
    source = GradientSlide(3200, 1600)
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=1000) as session:
        assert session.step_overview().pending
        assert len(source.reads) == 1
        session.cancel_overview()
        assert session.overview_cache_bytes == 0 and len(source.reads) == 1
        drain(session)
        assert session.overview_cache_bytes > 0
        full = Viewport(0, 0, 1600, 3200)
        plan = session.plan(full, rows=20, columns=10, allow_sampled_overview=True)
        assert session.sample(plan).cached_overview
        image.close()
        image.open()
        with pytest.raises(RuntimeError, match="lifetime"):
            session.sample(plan)
        with pytest.raises(RuntimeError, match="lifetime"):
            session.step_overview()


def test_failed_build_is_not_published_or_retried_automatically(monkeypatch):
    pixels = np.ones((60, 60), np.uint16)
    with MultichannelImage.from_array(pixels, axes="YX") as image, ViewSession(
        image, max_read_pixels=100,
    ) as session:
        calls = []
        original = image.read_region

        def failing(*args, **kwargs):
            calls.append(1)
            raise OSError("deliberate test read error")

        monkeypatch.setattr(image, "read_region", failing)
        progress = session.step_overview()
        assert not progress.pending and not progress.published
        assert "deliberate test read error" in session.overview_status
        assert session.overview_cache_bytes == 0
        assert not session.step_overview().pending and len(calls) == 1
        monkeypatch.setattr(image, "read_region", original)
        plan = session.plan(Viewport(0, 0, 60, 60), rows=4, columns=4,
                            allow_sampled_overview=True)
        assert not session.sample(plan).cached_overview
        session.clear_cache()
        drain(session)
        assert session.sample(plan).cached_overview


@pytest.mark.parametrize("suffix", ["svs", "ome.tif"])
def test_actual_tiff_reader_builds_cache_with_no_whole_level_materialization(
    tmp_path, monkeypatch, suffix,
):
    import tifffile
    path = tmp_path / f"overview.{suffix}"
    if suffix == "svs":
        synthetic_svs(path)
    else:
        rgb = np.full((97, 65, 3), (23, 101, 211), np.uint8)
        write_pyramid(path, rgb, rgb=True, declare_scales=False)
    with open_image(path) as image, ViewSession(image, max_read_pixels=500) as session:
        def forbidden(*args, **kwargs):
            pytest.fail("Must use regional Omeify reads")
        monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
        monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", forbidden)
        drain(session)
        renderer = TextRenderer(mode="blocks")
        whole = Viewport.fit(session.info.shape, rows=12, columns=24)
        result = make_frame(session, FrameRequest(1, whole, 24, 12, (0,), (None,), renderer))
        assert result.samples.cached_overview
        assert result.samples.plan.level.approximate == (suffix == "ome.tif")
        assert result.frame.shape == (12, 24)
        assert " " in result.frame.to_text()  # Aspect-preserving padding.
        assert session.info.pixel_size == image.pixel_size.to_tuple()


async def wait_for(predicate, task):
    for _ in range(1000):
        if task.done():
            await task
            pytest.fail("Viewer exited early")
        if predicate():
            return
        await asyncio.sleep(.01)
    pytest.fail("Viewer did not reach expected state")


@pytest.mark.asyncio
async def test_viewer_refines_same_camera_keeps_contrast_and_frozen_selection(monkeypatch):
    source = GradientSlide(3200, 1600)
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=12000) as session, \
            create_pipe_input() as pipe:
        viewer = Viewer(session, renderer=TextRenderer(mode="blocks"))
        whole = Viewport.fit(session.info.shape, rows=26, columns=68)
        viewer.viewport = whole
        original = session.step_overview
        allowed = Event()
        step_entered = Event()

        def gated(channels):
            step_entered.set()
            if not allowed.is_set():
                # Do not block the worker in this test; pause preparation so the
                # foreground frame is predictably available before publication.
                from ttwhy.runtime_overview import OverviewProgress
                return OverviewProgress()
            return original(channels)

        monkeypatch.setattr(session, "step_overview", gated)
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            await wait_for(lambda: viewer.frame is not None and not viewer.busy, task)
            assert not viewer.samples.cached_overview
            camera, windows = viewer.viewport, tuple(viewer.windows)
            # An inert frozen selection is sufficient: no naming/clipboard dependency.
            selection = SimpleNamespace(frame=viewer.frame, row=lambda _: [],
                                        kind="rectangle", closed=False, areas=[], active_index=0)
            monkeypatch.setattr(viewer.clipping, "status", lambda: "test frozen selection")
            viewer.clipping.selection = selection
            frozen = viewer.frame
            reads_before = len(source.reads)
            allowed.set()
            viewer._wake.set()
            await asyncio.sleep(.05)
            assert viewer.frame is frozen and not viewer.samples.cached_overview
            # One already-submitted worker step may complete after selection starts.
            assert len(source.reads) <= reads_before + 1
            stopped = len(source.reads)
            await asyncio.sleep(.05)
            assert len(source.reads) == stopped
            viewer.clipping.selection = None
            viewer.request_frame()
            await wait_for(lambda: viewer.samples.cached_overview and not viewer.busy, task)
            assert viewer.viewport == camera and tuple(viewer.windows) == windows
            assert "cached overview" in viewer._status()
            assert "sampled overview" not in viewer._status()
        finally:
            viewer.clipping.selection = None
            pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 10)
        assert session.overview_cache_bytes == 0


@pytest.mark.asyncio
async def test_foreground_preempts_build_and_quit_joins_only_one_chunk(monkeypatch):
    source = GradientSlide(3200, 1600)
    entered, release = Event(), Event()
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=12000) as session, \
            create_pipe_input() as pipe:
        original = session.step_overview
        step_calls = []

        def slow_step(channels):
            step_calls.append(len(source.reads))
            entered.set()
            assert release.wait(10), "Test did not release the in-flight chunk"
            return original(channels)

        monkeypatch.setattr(session, "step_overview", slow_step)
        viewer = Viewer(session, renderer=TextRenderer(mode="blocks"))
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            await wait_for(lambda: entered.is_set(), task)
            before = viewer.generation
            pipe.send_text("+")
            await wait_for(lambda: viewer.generation > before, task)
            # The UI processed zoom despite the native-worker step being occupied.
            release.set()
            await wait_for(lambda: not viewer.busy
                           and viewer.displayed_generation == viewer.generation, task)
            assert viewer.samples.plan.viewport == viewer.viewport
            assert viewer.viewport.height < 3200  # The user's zoom took effect.
            pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 10)
        finally:
            release.set()
            if not task.done():
                pipe.send_bytes(b"\x11")
                await asyncio.wait_for(task, 10)
        assert session.overview_cache_bytes == 0 and image.is_open
        assert step_calls


@pytest.mark.parametrize("keyword,value", [("overview_cache_mib", -1),
                                           ("overview_cache_mib", float("nan")),
                                           ("overview_max_source_pixels", -1),
                                           ("overview_max_source_pixels", 1.5)])
def test_configuration_validation(keyword, value):
    with MultichannelImage.from_array(np.ones((2, 2)), axes="YX") as image:
        with pytest.raises(ValueError, match="overview_"):
            ViewSession(image, **{keyword: value})
        assert image.is_open


def test_odd_declared_native_scale_and_anisotropy_are_not_reinferred():
    raw = np.arange(9 * 13, dtype=np.uint16).reshape(9, 13)
    size = PixelSize(.5, 1, "µm")
    levels = (ImageLevel(0, "YX", raw.shape, size),
              ImageLevel(1, "YX", (5, 7), size.scaled(2), (2, 2)))
    with MultichannelImage.from_array(raw, axes="YX", pixel_size=size, levels=levels,
                                      level_arrays=[raw[::2, ::2]]) as image, \
            ViewSession(image, max_read_pixels=8) as session:
        drain(session)
        plan = session.plan(Viewport(0, 0, 13, 9), rows=1, columns=1,
                            allow_sampled_overview=True)
        result = session.sample(plan)
        assert result.cached_overview and not plan.level.approximate
        assert session.info.pixel_aspect == 2 and session.info.pixel_size == (.5, 1, "µm")
        cached = session._overviews.get(session.info, (0,), 8)[0]
        assert cached.spec.level.scale_x * cached.spec.factor == 4
        weighted, support = oracle(raw[::2, ::2], 2, (4.5, 6.5))
        np.testing.assert_allclose(cached.weighted, weighted)
        np.testing.assert_allclose(cached.support, support)
        outside = cached.sample(Viewport(13, 0, 2, 9), rows=3, columns=1)
        assert np.isnan(outside).all()


@pytest.mark.asyncio
async def test_series_change_waits_for_chunk_then_drops_old_overview(tmp_path, monkeypatch):
    from ttwhy.image_document import ImageDocument
    from ttwhy.viewer.navigation import DocumentNavigation
    from test_document_navigation import write_series

    path = tmp_path / "series.ome.tif"
    write_series(path)
    document = ImageDocument(path)
    nav = DocumentNavigation(document, document.initialize(), max_read_pixels=300,
                             overview_cache_mib=1, overview_max_source_pixels=10_000)
    old = nav.session
    original = old.step_overview
    entered, release = Event(), Event()

    def gated(channels):
        entered.set()
        assert release.wait(10)
        assert old.is_open
        return original(channels)

    monkeypatch.setattr(old, "step_overview", gated)
    with create_pipe_input() as pipe:
        viewer = Viewer(old, navigation=nav, renderer=TextRenderer(mode="blocks"))
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            await wait_for(lambda: entered.is_set(), task)
            pipe.send_text("]")
            await wait_for(lambda: nav.target == 1 and nav.pending is not None, task)
            assert nav.active.index == 0 and old.is_open
            release.set()
            await wait_for(lambda: nav.active.index == 1 and not viewer.busy, task)
            assert not old.is_open and old.overview_cache_bytes == 0
            assert not viewer.session.info.is_rgb
            assert nav.session._overviews.budget == 1024**2
            assert nav.session._overviews.max_source_pixels == 10_000
            pipe.send_text("r")
            await wait_for(lambda: viewer.samples.cached_overview and not viewer.busy, task)
            finite = viewer.samples.values[np.isfinite(viewer.samples.values)]
            np.testing.assert_allclose(finite, 1000)
        finally:
            release.set()
            pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 10)
        assert nav.active.image is None
        assert nav.session.overview_cache_bytes == 0


@pytest.mark.asyncio
async def test_quit_during_chunk_finishes_no_later_chunks(monkeypatch):
    from threading import Timer

    source = GradientSlide(3200, 1600)
    entered, release = Event(), Event()
    with RGBImage(source) as image, ViewSession(image, max_read_pixels=12000) as session, \
            create_pipe_input() as pipe:
        original = session.step_overview
        calls = []

        def gated(channels):
            calls.append(1)
            entered.set()
            assert release.wait(10)
            assert session.is_open
            return original(channels)

        monkeypatch.setattr(session, "step_overview", gated)
        viewer = Viewer(session, renderer=TextRenderer(mode="blocks"))
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        timer = None
        try:
            await wait_for(lambda: entered.is_set(), task)
            # Shutdown joins the worker synchronously after leaving the terminal;
            # use a separate thread to release this deliberately blocked decode.
            timer = Timer(.15, release.set)
            timer.start()
            pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 10)
        finally:
            release.set()
            if timer is not None:
                timer.join()
            if not task.done():
                pipe.send_bytes(b"\x11")
                await asyncio.wait_for(task, 10)
        assert len(calls) == 1
        assert session.overview_cache_bytes == 0 and image.is_open


def test_cli_forwards_independent_overview_budgets(tmp_path, monkeypatch):
    import sys
    from click.testing import CliRunner
    from ttwhy.cli import main

    path = tmp_path / "image.png"
    path.write_bytes(b"not decoded by this forwarding test")
    received = {}

    async def fake_view_file(*args, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr("ttwhy.viewer.file_viewer.view_file", fake_view_file)
    monkeypatch.setattr(type(sys.stdout), "isatty", lambda self: True)
    # CliRunner replaces stdin/out, so it supplies the TTY seam explicitly here.
    import click.testing
    monkeypatch.setattr(click.testing._NamedTextIOWrapper, "isatty", lambda self: True)
    result = CliRunner().invoke(main, [str(path), "--mode", "simple",
                                     "--overview-cache-mib", "12",
                                     "--overview-max-source-pixels", "12345",
                                     "--startup-overview-seconds", "1.25"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert received["overview_cache_mib"] == 12
    assert received["overview_max_source_pixels"] == 12345
    assert received["startup_overview_seconds"] == 1.25


def test_snapshot_can_disable_quick_build_and_still_show_whole_image(mif, monkeypatch):
    from click.testing import CliRunner
    from ttwhy.cli import main

    def forbidden(*args, **kwargs):
        pytest.fail("Startup scan disabled; snapshot should use a sampled fallback")

    monkeypatch.setattr(ViewSession, "step_overview", forbidden)
    result = CliRunner().invoke(main, [str(mif[0]), "--snapshot", "--max-read-pixels", "100",
                                       "--startup-overview-seconds", "0"])
    assert result.exit_code == 0, result.output
    assert "Sampled overview" in result.output
    assert "bounded center crop" not in result.output


def test_svs_cached_overview_in_real_terminal(tmp_path):
    import io
    import os
    import sys

    if sys.platform == "win32":
        pytest.skip("POSIX pseudo-terminal test")
    pexpect = pytest.importorskip("pexpect")
    path = tmp_path / "portrait.svs"
    synthetic_svs(path)
    env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
    env.pop("NO_COLOR", None)
    env.pop("PROMPT_TOOLKIT_COLOR_DEPTH", None)
    child = pexpect.spawn(
        sys.executable, ["-m", "ttwhy", str(path), "--max-read-pixels", "500"],
        dimensions=(30, 120), encoding="utf-8", env=env, timeout=15,
    )
    transcript = io.StringIO()
    child.logfile_read = transcript
    try:
        child.expect(r"L[0-9]")
        child.send("r")
        child.expect("cached overview")
        child.send("q")
        child.expect(pexpect.EOF)
    finally:
        child.close(force=True)
    output = transcript.getvalue()
    assert child.exitstatus == 0 and "Traceback" not in output
    assert "\x1b[?1049h" in output and "\x1b[?1049l" in output
    assert "\x1b[?1006l" in output and "\x1b[?25h" in output
