"""Actual PDF processes, document ownership, and interactive page/series controls."""

import asyncio
import multiprocessing as mp
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from prompt_toolkit.input import create_pipe_input

from test_image_input import write_pdf
from test_viewer import SizedOutput, eventually, ready, send
from twhy import IntensityWindow, TextRenderer
from twhy.cli import main
from twhy.image_document import ImageDocument
from twhy.pdf_workers import PdfWorkers, pdf_dpi_steps, render_pdf_pyramid
from twhy.viewer import Viewer
from twhy.viewer.navigation import DocumentNavigation


@pytest.mark.parametrize("ceiling,steps", [
    (72, (72,)), (144, (144,)), (200, (144, 200)), (300, (144, 300)),
    (450, (144, 300, 450)), (600, (144, 300, 600)),
    (900, (144, 300, 600, 900)), (1200, (144, 300, 600, 1200)),
])
def test_resolution_schedule(ceiling, steps):
    assert pdf_dpi_steps(ceiling) == steps
    assert pdf_dpi_steps() == (144, 300, 600)


@pytest.mark.parametrize("ceiling", [0, -1, float("inf"), float("nan"), True])
def test_invalid_resolution_ceiling(ceiling):
    with pytest.raises(ValueError, match="positive and finite"):
        pdf_dpi_steps(ceiling)


def write_series(path):
    """Independent OME fixture with RGB, multichannel, then single-channel data."""
    arrays = [np.full((31, 43, 3), [255, 0, 0], np.uint8),
              np.stack([np.full((25, 37), 1000, np.uint16),
                        np.full((25, 37), 30000, np.uint16)]),
              np.full((17, 29), 5000, np.uint16)]
    with tifffile.TiffWriter(path, ome=True) as writer:
        for index, (array, axes, names) in enumerate(zip(
            arrays, ["YXS", "CYX", "YX"], [["RGB"], ["DAPI", "CD3"], ["Intensity"]],
            strict=True,
        )):
            writer.write(array, photometric="rgb" if index == 0 else "minisblack",
                         tile=(16, 16), metadata={"axes": axes, "Name": f"Series {index}",
                                                  "Channel": {"Name": names}})
    return arrays


@asynccontextmanager
async def browsing(path, **kwargs):
    doc = ImageDocument(path, **{key: value for key, value in kwargs.items()
                                if key in ("page", "series", "pdf_max_dpi")})
    initial = doc.initialize()
    nav = DocumentNavigation(doc, initial, cache_pages=kwargs.get("cache_pages", 1),
                             auto_invert=True)
    with create_pipe_input() as pipe:
        viewer = Viewer(nav.session, navigation=nav, renderer=TextRenderer(mode="blocks"))
        output = SizedOutput()
        task = asyncio.create_task(viewer.run(input=pipe, output=output))
        try:
            await ready(viewer, task)
            yield nav, viewer, pipe, output, task
        finally:
            if not task.done():
                pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 20)
            assert nav.active.image is None
            assert nav.cache_bytes == 0
            assert nav._workers is None or nav._workers.active_count == 0


async def test_pdf_next_previous_jump_invalid_and_cancel(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "book.pdf")
    async with browsing(path, pdf_max_dpi=144) as (nav, viewer, pipe, _, task):
        assert nav.active.index == 1 and nav.document.count == 2
        await send(pipe, "]", viewer, task)
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        assert "Page 2/2" in nav.status()
        np.testing.assert_allclose(viewer.samples.values[10, 10], [0, 0, 255], atol=1e-10)
        await send(pipe, "\x1b[5~", viewer, task)  # PageUp
        await eventually(lambda: nav.active.index == 1 and not viewer.busy, task)
        await send(pipe, "p\x01\x0b0\r", viewer, task)
        assert viewer.editing == "destination" and "Invalid page" in viewer.message
        assert nav.active.index == 1
        await send(pipe, "\x03", viewer, task)
        assert viewer.editing is None
        await send(pipe, "p\x01\x0b2\r", viewer, task)
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        assert viewer.editing is None
        await send(pipe, "]", viewer, task)
        assert nav.target == 2 and "No page beyond" in viewer.message
        await send(pipe, "[", viewer, task)
        await eventually(lambda: nav.active.index == 1 and not viewer.busy, task)
        assert nav.cache_pages == 1 and len(nav.cache) == 1


async def test_ome_series_shared_navigation_without_whole_image_reads(tmp_path, monkeypatch):
    path = tmp_path / "series.ome.tif"
    write_series(path)

    def prohibited(*args, **kwargs):
        raise AssertionError("Navigation must not materialize OME series")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", prohibited)
    async with browsing(path) as (nav, viewer, pipe, _, task):
        assert nav.active.index == 0 and nav.document.count == 3
        old = nav.active
        await send(pipe, "\x1b[6~", viewer, task)  # PageDown
        await eventually(lambda: nav.active.index == 1 and not viewer.busy, task)
        assert old.image is None
        assert viewer.session.info.channels == ("DAPI", "CD3")
        assert not viewer.renderer.invert and len(viewer.windows) == 2
        await send(pipe, "2", viewer, task)
        assert viewer.selected_channels == (1,)
        await send(pipe, "p\x01\x0b2\r", viewer, task)
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        assert viewer.selected_channels == (0,) and len(viewer.windows) == 1
        await send(pipe, "p\x01\x0b0\r", viewer, task)
        await eventually(lambda: nav.active.index == 0 and not viewer.busy, task)
        assert viewer.session.info.is_rgb and viewer.renderer.invert
        assert not nav.cache and nav._workers is None


async def test_real_pdf_processes_and_owned_rgb_pyramids(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "workers.pdf")
    before = set(tmp_path.iterdir())
    workers = PdfWorkers()
    try:
        a, b = await asyncio.gather(
            workers.render(path, 1, 300, (255, 255, 255)),
            workers.render(path, 2, 600, (255, 255, 255)),
        )
        assert a[0].shape == (200, 300, 3)
        assert b[0].shape == (400, 600, 3)
        np.testing.assert_array_equal(a[0][10, 10], [255, 0, 0])
        np.testing.assert_array_equal(b[0][10, 10], [0, 0, 255])
        assert all(not array.flags.writeable for array in (*a, *b))
        assert workers.active_count == 0
        assert set(tmp_path.iterdir()) == before
    finally:
        workers.close()


def slow_process(send, *args):
    time.sleep(60)
    send.close()


def crash_process(send, *args):
    os._exit(7)


async def test_cancellation_reaps_process_and_receiver(tmp_path, monkeypatch):
    monkeypatch.setattr("twhy.pdf_workers.pdf_process", slow_process)
    workers = PdfWorkers()
    before = {p.pid for p in mp.active_children()}
    task = asyncio.create_task(workers.render(tmp_path / "unused", 1, 600, (0, 0, 0)))
    await eventually(lambda: workers.active_count == 1, task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    workers.close()
    assert workers.active_count == 0
    assert {p.pid for p in mp.active_children()} == before


async def test_worker_death_is_reported_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr("twhy.pdf_workers.pdf_process", crash_process)
    workers = PdfWorkers()
    try:
        with pytest.raises(RuntimeError, match="without a complete result"):
            await workers.render(tmp_path / "unused", 1, 600, (0, 0, 0))
        assert workers.active_count == 0
    finally:
        workers.close()


class GatedWorkers:
    """Control completion order; use the real PDF decoder once a gate is released."""

    instances = []

    def __init__(self):
        self.calls = []
        self.gates = {}
        self.cancelled = []
        self.closed = False
        type(self).instances.append(self)

    @property
    def active_count(self):
        return 0

    async def render(self, path, page, dpi, background):
        key = page, dpi
        self.calls.append(key)
        gate = self.gates.setdefault(key, asyncio.Event())
        try:
            await gate.wait()
            return render_pdf_pyramid(str(path), page, dpi, background)
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise

    def close(self):
        self.closed = True


async def test_progressive_upgrade_keeps_camera_contrast_and_never_downgrades(
    tmp_path, monkeypatch
):
    pytest.importorskip("pypdfium2")
    monkeypatch.setattr("twhy.viewer.navigation.PdfWorkers", GatedWorkers)
    path = write_book(tmp_path / "progressive.pdf")
    async with browsing(path, pdf_max_dpi=600) as (nav, viewer, pipe, _, task):
        await eventually(lambda: nav._workers is not None and len(nav._workers.calls) == 2, task)
        assert nav.active.dpi == 144
        await send(pipe, "++\x1b[Cu\x01\x0b200\r", viewer, task)
        old = nav.active
        old_view, old_shape = viewer.viewport, viewer.session.info.shape
        expected_window = viewer.windows[0]
        nav._workers.gates[1, 600].set()  # Deliberately finish finer work first.
        await eventually(lambda: nav.active.dpi == 600 and not viewer.busy, task)
        new_shape = viewer.session.info.shape
        sy, sx = new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]
        assert viewer.viewport.x / sx == pytest.approx(old_view.x)
        assert viewer.viewport.y / sy == pytest.approx(old_view.y)
        assert viewer.viewport.width / sx == pytest.approx(old_view.width)
        assert viewer.viewport.height / sy == pytest.approx(old_view.height)
        assert viewer.windows[0] == expected_window == IntensityWindow(0, 200)
        assert old.image is None
        nav._workers.gates[1, 300].set()
        await eventually(lambda: nav._refine_task.done(), task)
        assert nav.active.dpi == 600 and nav.cache[1].dpi == 600
        assert len(nav.cache) == 1


async def test_neighbor_prefetch_waits_for_current_and_cached_flip_is_reused(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    monkeypatch.setattr("twhy.viewer.navigation.PdfWorkers", GatedWorkers)
    path = write_pdf(tmp_path / "neighbors.pdf")
    async with browsing(path, pdf_max_dpi=600, cache_pages=3) as (nav, viewer, pipe, _, task):
        await eventually(lambda: nav._workers is not None and len(nav._workers.calls) == 2, task)
        worker = nav._workers
        worker.gates[1, 600].set()
        await eventually(lambda: nav.active.dpi == 600, task)
        assert all(page == 1 for page, _ in worker.calls)
        worker.gates[1, 300].set()
        for dpi in (144, 300, 600):
            await eventually(lambda dpi=dpi: (2, dpi) in worker.gates, task)
            worker.gates[2, dpi].set()
        await eventually(lambda: nav._refine_task.done(), task)
        assert nav.cache[2].dpi == 600

        def prohibited(*args):
            raise AssertionError("Cached page must not be decoded again")

        monkeypatch.setattr(nav.document, "load", prohibited)
        await send(pipe, "]", viewer, task)
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        assert nav.active.dpi == 600
        assert viewer.viewport.width >= 600  # New page fits rather than keeping old crop.
        await send(pipe, "[", viewer, task)
        await eventually(lambda: nav.active.index == 1 and not viewer.busy, task)
        assert nav.active.dpi == 600


async def test_failed_detail_keeps_usable_preview(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")

    class FailingWorkers(GatedWorkers):
        async def render(self, path, page, dpi, background):
            raise MemoryError("deliberate allocation failure")

    monkeypatch.setattr("twhy.viewer.navigation.PdfWorkers", FailingWorkers)
    path = write_pdf(tmp_path / "fail.pdf")
    async with browsing(path, pdf_max_dpi=600) as (nav, viewer, pipe, _, task):
        await eventually(lambda: nav._refine_task is not None and nav._refine_task.done(), task)
        assert nav.active.dpi == 144 and "allocation failure" in nav.error
        assert viewer.frame is not None and nav.session.is_open
        await send(pipe, "+", viewer, task)
        assert nav.session.is_open


async def test_initial_load_is_coalesced_and_not_on_ui_thread(tmp_path, monkeypatch):
    path = tmp_path / "load.ome.tif"
    write_series(path)
    async with browsing(path) as (nav, viewer, pipe, _, task):
        started, release = threading.Event(), threading.Event()
        original = nav.document.load
        calls, items = [], []

        def delayed(index):
            calls.append(index)
            if index == 1:
                started.set()
                assert release.wait(4)
            item = original(index)
            items.append(item)
            return item

        monkeypatch.setattr(nav.document, "load", delayed)
        try:
            nav.go(1)
            await eventually(started.is_set, task)
            before = viewer.viewport
            await send(pipe, "+", viewer, task)
            assert viewer.viewport.width < before.width
            nav.go(2)
            release.set()
            await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
            assert calls == [1, 2]
            assert items[0].image is None
        finally:
            release.set()


def test_old_cli_flag_removed_and_snapshot_uses_maximum(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "snapshot.pdf")
    runner = CliRunner()
    assert "--pdf-max-dpi" in runner.invoke(main, ["--help"]).output
    assert "--pdf-dpi" not in runner.invoke(main, ["--help"]).output
    result = runner.invoke(main, [str(path), "--snapshot", "--pdf-dpi", "144"])
    assert result.exit_code != 0
    from twhy import image_input
    original = image_input._pdf_pixels
    resolutions = []

    def record(path, page, dpi, background):
        resolutions.append(dpi)
        return original(path, page, dpi, background)

    monkeypatch.setattr(image_input, "_pdf_pixels", record)
    for args, dpi in [([], 600), (["--pdf-max-dpi", "144"], 144),
                      (["--pdf-max-dpi", "200"], 200)]:
        result = runner.invoke(main, [str(path), "--snapshot", *args])
        assert result.exit_code == 0, result.output
        assert resolutions[-1] == dpi
    assert resolutions == [600, 144, 200]


def write_book(path, *, count=5):
    """Several pages, fractional media sizes and different colors; stdlib only."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               (f"<< /Type /Pages /Count {count} /Kids [" +
                " ".join(f"{3 + i*2} 0 R" for i in range(count)) + "] >>").encode()]
    colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (.6, 0, .8), (1, .8, 0)]
    for i in range(count):
        width, height = 73.3 + i * 3, 49.2 + i * 2
        r, g, b = colors[i % len(colors)]
        content = (f"{r} {g} {b} rg 0 0 {width} {height} re f "
                   f"0 0 0 RG 0.2 w 3 3 {width-6} {height-6} re S").encode()
        rotation = " /Rotate 90" if i % 2 else ""
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
                        f"/Resources << >> /Contents {4+i*2} 0 R{rotation} >>").encode())
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)
    return path


async def test_moving_three_page_cache_releases_distant_images(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    monkeypatch.setattr("twhy.viewer.navigation.PdfWorkers", GatedWorkers)
    path = write_book(tmp_path / "five.pdf")
    async with browsing(
        path, page=3, pdf_max_dpi=144, cache_pages=3
    ) as (nav, viewer, pipe, _, task):
        await eventually(lambda: nav._workers is not None and len(nav._workers.calls) == 2, task)
        worker = nav._workers
        assert set(worker.calls) == {(2, 144), (4, 144)}
        for gate in worker.gates.values():
            gate.set()
        await eventually(lambda: nav._refine_task.done(), task)
        assert set(nav.cache) == {2, 3, 4}
        second = nav.cache[2]
        fourth = nav.cache[4]
        await send(pipe, "]", viewer, task)
        await eventually(lambda: nav.active.index == 4 and not viewer.busy, task)
        assert nav.active is fourth and second.image is None
        await eventually(lambda: (5, 144) in worker.gates, task)
        worker.gates[5, 144].set()
        await eventually(lambda: nav._refine_task.done(), task)
        assert set(nav.cache) == {3, 4, 5}
        old = list(nav.cache.values())
        await send(pipe, "p1\r", viewer, task)
        await eventually(lambda: nav.active.index == 1 and not viewer.busy, task)
        assert all(item.image is None for item in old)
        assert len(nav.cache) <= 3


async def test_source_handoff_waits_for_inflight_frame(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "handoff.pdf")
    async with browsing(path, pdf_max_dpi=144) as (nav, viewer, pipe, _, task):
        started, release = threading.Event(), threading.Event()
        old = nav.active
        original = nav.session.sample

        def slow(plan):
            started.set()
            assert release.wait(4)
            assert old.image is not None and old.image.is_open
            return original(plan)

        monkeypatch.setattr(nav.session, "sample", slow)
        try:
            viewer.zoom(2)
            await eventually(started.is_set, task)
            nav.go(2)
            await eventually(lambda: nav.pending is not None, task)
            assert nav.active.index == 1 and old.image.is_open
            release.set()
            await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
            assert old.image is None
            finite = viewer.samples.values[np.isfinite(viewer.samples.values).all(axis=-1)]
            np.testing.assert_allclose(
                finite, np.broadcast_to([0, 0, 255], finite.shape), atol=1e-9
            )
        finally:
            release.set()


async def test_failed_series_open_keeps_current_source_and_navigation(tmp_path, monkeypatch):
    path = tmp_path / "failed-series.ome.tif"
    write_series(path)
    async with browsing(path) as (nav, viewer, pipe, _, task):
        original = nav.document.load
        active = nav.active

        def failing(index):
            if index == 1:
                raise ValueError("unsupported Z stack")
            return original(index)

        monkeypatch.setattr(nav.document, "load", failing)
        nav.go(1)
        await eventually(lambda: "unsupported Z stack" in nav.error, task)
        assert nav.active is active and nav.target == 0 and active.image.is_open
        assert "unsupported Z stack" in viewer.message
        await send(pipe, "p2\r", viewer, task)
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        assert active.image is None


async def test_read_budget_cannot_move_camera_during_upgrade(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    monkeypatch.setattr("twhy.viewer.navigation.PdfWorkers", GatedWorkers)
    path = write_pdf(tmp_path / "budget.pdf")
    async with browsing(path, pdf_max_dpi=600) as (nav, viewer, _, _, task):
        await eventually(lambda: nav._workers is not None and len(nav._workers.calls) == 2, task)
        nav.max_read_pixels = nav.session.max_read_pixels = 11000
        before = viewer.viewport
        for gate in nav._workers.gates.values():
            gate.set()
        await eventually(lambda: nav._refine_task.done() and not viewer.busy, task)
        assert nav.active.dpi == 144 and nav.cache[1].dpi == 600
        assert viewer.viewport == before and "read budget" in nav.error
        viewer.zoom(10)
        await eventually(lambda: nav.active.dpi == 600 and not viewer.busy, task)
        assert nav.session.max_read_pixels == 11000
        assert not nav.error


async def test_real_file_viewer_refines_and_quits_with_jobs_active(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    from twhy.viewer import file_viewer

    created = []

    class ObservedViewer(Viewer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(file_viewer, "Viewer", ObservedViewer)
    path = write_book(tmp_path / "real-book.pdf")
    children_before = {p.pid for p in mp.active_children()}
    with create_pipe_input() as pipe:
        task = asyncio.create_task(file_viewer.view_file(path, input=pipe, output=SizedOutput()))
        try:
            await eventually(lambda: bool(created), task)
            viewer = created[0]
            nav = viewer.navigation
            await eventually(lambda: nav.active.dpi == 600 and not viewer.busy, task, timeout=20)
            assert nav.active.index == 1 and "600/600 DPI" in nav.status()
            pipe.send_text("]")
            await eventually(lambda: nav.active.index == 2 and not viewer.busy, task, timeout=20)
            pipe.send_text("p4\r")
            await eventually(lambda: nav.active.index == 4 and not viewer.busy, task, timeout=20)
            pipe.send_text("][[q")  # Rapid retargets followed immediately by quit.
            await asyncio.wait_for(task, 20)
        finally:
            if not task.done():
                pipe.send_bytes(b"\x11")
                await asyncio.wait_for(task, 20)
        assert nav.cache_bytes == 0 and nav.active.image is None
    assert {p.pid for p in mp.active_children()} == children_before


async def test_clickable_navigation_and_destination_entry(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "buttons.pdf")
    async with browsing(path, pdf_max_dpi=144) as (nav, viewer, pipe, _, task):
        # The navigation bar is terminal row 2, with the next button at column 7.
        await send(pipe, "\x1b[<0;7;2M\x1b[<0;7;2m", viewer, task)
        await eventually(lambda: nav.active.index == 2 and not viewer.busy, task)
        await send(pipe, "\x1b[<0;12;2M\x1b[<0;12;2m", viewer, task)
        assert viewer.editing == "destination" and viewer.entry.text == ""
        await send(pipe, "1\r", viewer, task)
        await eventually(lambda: nav.active.index == 1 and not viewer.busy, task)


async def test_shutdown_awaits_both_child_finalizers(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    class CleanupWorkers(GatedWorkers):
        def __init__(self):
            super().__init__()
            self.live = set()

        async def render(self, path, page, dpi, background):
            self.calls.append((page, dpi))
            self.live.add(dpi)
            try:
                await asyncio.Event().wait()
            finally:
                if dpi == 600:
                    cleanup_started.set()
                    await cleanup_release.wait()
                self.live.remove(dpi)

        def close(self):
            assert not self.live, "A sibling PDF finalizer was abandoned"
            super().close()

    monkeypatch.setattr("twhy.viewer.navigation.PdfWorkers", CleanupWorkers)
    path = write_pdf(tmp_path / "cleanup.pdf")
    async with browsing(path, pdf_max_dpi=600) as (nav, _, pipe, _, task):
        await eventually(lambda: nav._workers is not None and len(nav._workers.calls) == 2, task)
        pipe.send_text("q")
        await eventually(cleanup_started.is_set, task)
        await asyncio.sleep(.05)
        assert not task.done()
        cleanup_release.set()
        await asyncio.wait_for(task, 5)
        assert nav._workers.closed


def test_snapshot_region_uses_initial_preview_coordinates(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    from twhy import ViewSession
    from twhy.image_input import pdf_raster_shape

    path = write_book(tmp_path / "coordinates.pdf")
    plans = []
    original = ViewSession.plan

    def record(self, viewport, **kwargs):
        plans.append((self.info.shape, viewport))
        return original(self, viewport, **kwargs)

    monkeypatch.setattr(ViewSession, "plan", record)
    result = CliRunner().invoke(main, [str(path), "--snapshot", "--region", "10", "12", "20", "18"])
    assert result.exit_code == 0, result.output
    initial_y, initial_x = pdf_raster_shape(path, 1, 144)
    shape, view = plans[0]
    assert view.x == pytest.approx(10 * shape[1] / initial_x)
    assert view.y == pytest.approx(12 * shape[0] / initial_y)
    assert view.width == pytest.approx(20 * shape[1] / initial_x)
    assert view.height == pytest.approx(18 * shape[0] / initial_y)
