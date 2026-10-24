"""Privacy, union exports, polygon masks, long channel lists and ASCII presets."""

import asyncio
import base64
import json
import re
import sys
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from PIL import Image
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from conftest import write_pyramid
from test_viewer import eventually, ready, running, send
from ttwhy import TextImage, TextRenderer, Viewport
from ttwhy import annotation_identity
from ttwhy.annotation_identity import IdentityFactory
from ttwhy.geojson_clip import geometry_feature
from ttwhy.png_clip import render_clip_png
from ttwhy.selection_geometry import SelectionGeometry, union_mask
from ttwhy.viewer import Viewer
from ttwhy.viewer.controls import clip_fragments
from ttwhy.viewer.rendering import terminal_lines
from ttwhy.viewer.selection_set import SelectionSet

REAL_RANDOM_NAME = annotation_identity.random_name


def rect(x0, y0, x1, y1):
    return SelectionGeometry("rectangle", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


@pytest.fixture
def cells():
    y, x = np.mgrid[:12, :18]
    colors = np.stack([x*9, y*13, (x+y)*7], -1).astype(np.uint8)
    chars = np.array(list("ABCDEFGHIJKLMNOPQR") * 12).reshape(12, 18)
    return TextImage(chars, foreground=colors, background=255-colors)


@pytest.fixture
def fake_viewer(cells, tmp_path):
    source = tmp_path / "PRIVATE_USERNAME_PATIENT_RECORD.png"
    source.write_bytes(b"source unchanged")
    session = SimpleNamespace(info=SimpleNamespace(shape=cells.shape, channels=("secret marker",),
                                                   is_rgb=True),
                              _select_channels=lambda values: tuple(values))
    viewer = Viewer(session, source_path=source, clipboard="osc52",
                    renderer=TextRenderer(mode="blocks"))
    viewer.frame = cells
    viewer.cell_lines = terminal_lines(cells)
    viewer.canvas_size = (18, 12)
    viewer.viewport = Viewport(0, 0, 18, 12)
    viewer.generation = viewer.displayed_generation = 1
    written = []
    viewer.application = SimpleNamespace(
        layout=SimpleNamespace(focus=lambda _: None), invalidate=lambda: None,
        create_background_task=asyncio.create_task, is_running=True,
        output=SimpleNamespace(write_raw=written.append, flush=lambda: None,
                               get_size=lambda: Size(30, 100)))
    viewer.test_writes = written
    viewer.clipping.start()
    try:
        yield viewer
    finally:
        viewer.clipping.close()


def draw_rect(selection, a=(1, 1), b=(3, 3)):
    selection.start_drag(*a)
    selection.drag(*b, finish=True)


def draw_poly(selection, points=((8, 2), (14, 2), (8, 8))):
    for point in points:
        selection.start_drag(*point)
        selection.drag(*point, finish=True)
    selection.start_drag(*points[0])
    assert selection.closed


def payload(viewer):
    osc = [s for s in viewer.test_writes if s.startswith("\x1b]52;")][-1]
    return base64.b64decode(osc[7:-1]).decode("utf-8")


def test_coolname_adapter_requests_three_words(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "coolname", SimpleNamespace(
        generate_slug=lambda n: calls.append(n) or "birch-paper-finch"))
    assert REAL_RANDOM_NAME() == "birch-paper-finch"
    assert calls == [3]


def test_real_coolname_distribution_and_vocabulary(monkeypatch):
    module = pytest.importorskip("coolname")
    monkeypatch.setattr(annotation_identity, "random_name", lambda: module.generate_slug(3))
    factory = IdentityFactory()
    records = [factory.create() for _ in range(30)]
    assert len({record.name for record in records}) == 30
    assert all(len(record.name.split("-")) == 3 for record in records)
    assert all(UUID(record.id).version == 4 for record in records)


def test_identity_collision_retry_rename_and_stable_uuid(monkeypatch):
    words = iter(["birch-paper-finch", "birch-paper-finch", "small-copper-otter"])
    monkeypatch.setattr(annotation_identity, "random_name", lambda: next(words))
    factory = IdentityFactory()
    first, second = factory.create(), factory.create()
    assert first.id != second.id and first.name != second.name
    assert UUID(first.id).version == UUID(second.id).version == 4
    assert first.name_generated
    changed = first.renamed("Interesting area")
    assert changed.id == first.id and not changed.name_generated
    assert first.renamed(first.name) is first
    with pytest.raises(ValueError):
        first.renamed("bad\x1bname")


def test_name_retry_is_bounded(monkeypatch):
    monkeypatch.setattr(annotation_identity, "random_name", lambda: "not-three")
    with pytest.raises(RuntimeError, match="fresh"):
        IdentityFactory().create()


@pytest.mark.parametrize("points", [((0, 0), (1, 1)), ((0, 0), (1, 1), (2, 2)),
                                      ((0, 0), (2, 2), (0, 2), (2, 0)),
                                      ((0, 0), (3, 0), (2, 0), (2, 3)),
                                      ((0, 0), (2, 0), (float("nan"), 2))])
def test_reject_degenerate_and_crossed_polygons(points):
    with pytest.raises(ValueError):
        SelectionGeometry("polygon", points)


def test_concave_mask_and_reversed_winding_agree():
    points = ((.5, .5), (5.5, .5), (5.5, 2.5), (2.5, 2.5), (2.5, 5.5), (.5, 5.5))
    forward = SelectionGeometry("polygon", points)
    reverse = SelectionGeometry("polygon", tuple(reversed(points)))
    mask = union_mask((forward,), shape=(7, 7))
    np.testing.assert_array_equal(mask, union_mask((reverse,), shape=(7, 7)))
    assert mask[0, 0] and mask[2, 5] and mask[5, 2]
    assert not mask[4, 4] and not mask[6].any()
    assert not mask.flags.writeable


def test_union_preserves_gaps_and_overlaps(cells):
    geometries = (rect(1, 1, 4, 4), rect(9, 6, 12, 9), rect(2, 2, 5, 5))
    mask = union_mask(geometries, shape=cells.shape)
    assert mask.sum() == 9 + 9 + 9 - 4
    assert not mask[4, 8]
    clipped = TextImage(cells.characters, foreground=cells.foreground,
                        background=cells.background, mask=mask)
    assert np.all(clipped.characters[~mask] == " ")
    assert np.all(clipped.foreground[~mask] == 0)
    assert np.all(clipped.background[~mask] == 0)
    ansi = clipped.to_text(ansi=True)
    assert re.sub(r"\x1b\[[0-9;]+m", "", ansi) == clipped.to_text()
    # Missing cells carry reset/default style rather than hidden image RGB.
    assert any(run.foreground is None and run.background is None for run in clipped.iter_runs(1))


def test_rgba_polygon_edges_are_subcell_and_hidden_rgb_zero():
    fg = np.zeros((5, 5, 3), np.uint8)
    bg = np.full_like(fg, (200, 90, 30))
    frame = TextImage(np.full((5, 5), "▀"), foreground=fg, background=bg)
    geom = SelectionGeometry("polygon", ((.5, .5), (4.5, .5), (.5, 4.5)))
    data = render_clip_png(frame, regions=(geom,))
    with Image.open(BytesIO(data)) as image:
        assert image.mode == "RGBA"
        assert not image.info
        pixels = np.array(image)
    assert pixels[0, 0, 3] == 0
    assert pixels[30, 14, 3] == 255  # A genuinely black foreground remains opaque.
    np.testing.assert_array_equal(pixels[30, 14, :3], [0, 0, 0])
    assert set(np.unique(pixels[48:72, 24:36, 3])) == {0, 255}
    assert np.all(pixels[pixels[..., 3] == 0, :3] == 0)


def test_geojson_clips_real_edges_and_rejects_empty_image_intersection():
    identity = IdentityFactory().create()
    geom = SelectionGeometry("polygon", ((0, 0), (8, 0), (0, 8)))
    feature = geometry_feature(geom, viewport=Viewport(-2, -2, 10, 10),
                               frame_shape=(10, 10), raster_shape=(10, 10), identity=identity)
    ring = feature["geometry"]["coordinates"][0]
    assert ring == [[0., 0.], [4., 0.], [0., 4.], [0., 0.]]
    with pytest.raises(ValueError, match="no area"):
        geometry_feature(geom, viewport=Viewport(-100, -100, 10, 10),
                         frame_shape=(10, 10), raster_shape=(10, 10), identity=identity)


async def test_multiple_geojson_features_minimal_private_and_stable(fake_viewer, tmp_path):
    viewer = fake_viewer
    clip, selection = viewer.clipping, viewer.clipping.selection
    draw_rect(selection)
    first_id = selection.active.identity.id
    clip.add_area("polygon")
    draw_poly(selection)
    second_id = selection.active.identity.id
    clip.accept_name("Area B")
    clip.submit_geojson()
    await clip._task
    value = json.loads(payload(viewer))
    assert len(value["features"]) == 2
    assert [v["id"] for v in value["features"]] == [first_id, second_id]
    assert value["features"][1]["properties"]["name"] == "Area B"
    assert value["features"][1]["properties"]["name_generated"] is False
    for feature in value["features"]:
        assert set(feature) == {"id", "type", "geometry", "properties"}
        assert set(feature["properties"]) == {"name", "name_generated", "selection_type"}
        assert UUID(feature["id"]).version == 4
    serialized = json.dumps(value)
    for forbidden in ("PRIVATE_", "source_path", "image_width", "series", "page", "secret marker"):
        assert forbidden not in serialized
    out = tmp_path / "export.geojson"
    clip.submit_geojson(copy=False, path=out)
    await clip._task
    assert json.loads(out.read_text()) == value
    selection.cycle(-1)
    clip.accept_name("Area A")
    clip.submit_geojson()
    await clip._task
    renamed = json.loads(payload(viewer))
    assert [v["id"] for v in renamed["features"]] == [first_id, second_id]
    assert renamed["features"][0]["properties"]["name"] == "Area A"


async def test_union_text_ansi_png_copy_save_and_empty_draft(fake_viewer, tmp_path, monkeypatch):
    viewer = fake_viewer
    clip, selection = viewer.clipping, viewer.clipping.selection
    draw_rect(selection)
    clip.add_area("polygon")
    draw_poly(selection)
    expected = selection.clip()
    clip.add_area()  # Empty draft is not a third region or an export error.
    assert len(selection.snapshots()) == 2
    for ansi in (False, True):
        out = tmp_path / ("selected.ansi" if ansi else "selected.txt")
        clip.submit(ansi=ansi, path=out)
        await clip._task
        text = expected.to_text(ansi=ansi)
        assert payload(viewer) == text
        assert out.read_text() == text + "\n"
    captured = []
    from ttwhy.text_clip import ClipboardDelivery
    monkeypatch.setattr("ttwhy.viewer.clipping.clipboard_png",
                        lambda data, mode: captured.append(data) or ClipboardDelivery("PNG request"))
    out = tmp_path / "selected.png"
    clip.submit_image(path=out)
    await clip._task
    assert out.read_bytes() == captured[0]
    with Image.open(out) as image:
        pixels = np.array(image)
    assert pixels.shape[2] == 4
    assert np.all(pixels[pixels[..., 3] == 0] == 0)
    assert pixels[36, 12, 3] == 255
    assert pixels[36, 72, 3] == 0  # Gap between the two shapes.


async def test_unfinished_polygon_does_not_silently_export_other_areas(fake_viewer, tmp_path):
    clip = fake_viewer.clipping
    draw_rect(clip.selection)
    clip.add_area("polygon")
    clip.selection.start_drag(9, 8)
    out = tmp_path / "no.geojson"
    clip.submit_geojson(copy=False, path=out)
    assert "Close the polygon" in fake_viewer.message
    assert clip._task is None and not out.exists()
    clip.remove_area()
    assert len(clip.selection.snapshots()) == 1
    clip.submit_geojson(copy=False, path=out)
    await clip._task
    assert len(json.loads(out.read_text())["features"]) == 1


def test_neutral_filenames_and_short_toolbar(fake_viewer):
    clip = fake_viewer.clipping
    draw_rect(clip.selection)
    for options, expected in (({}, "selection.txt"), ({"ansi": True}, "selection.ansi"),
                              ({"image": True}, "selection.png"),
                              ({"geojson": True}, "regions.geojson")):
        clip.save(**options)
        assert fake_viewer.entry.text == expected
        fake_viewer.cancel_edit()
    for width in (80, 60, 45):
        fake_viewer.application.output.get_size = lambda: Size(30, width)
        fragments = clip_fragments(fake_viewer)
        assert len("".join(f[1] for f in fragments)) <= width
        assert {f[1] for f in fragments if len(f) == 3} >= {"y", "Y", "i", "I", "t", "T", "j", "J"}


async def test_ascii_and_color_cycles_preserve_camera_ids_masks_and_use_no_new_read(
    rgb, tmp_path, monkeypatch,
):
    out = tmp_path / "plain.ansi"
    async with running(rgb[0], output_ansi_clip=out, clipboard="off") as (viewer, pipe, _, task):
        before, windows = viewer.renderer, tuple(viewer.windows)
        await send(pipe, "M", viewer, task)
        assert viewer.renderer.mode == "simple" and not viewer.renderer.color
        assert viewer.mode_navigation.label == "ASCII classic"
        assert viewer.frame.to_text(ansi=True).isascii()
        assert "\x1b" not in viewer.frame.to_text(ansi=True)
        await send(pipe, "M", viewer, task)
        assert viewer.mode_navigation.label == "ASCII fine"
        await send(pipe, "m", viewer, task)
        assert viewer.renderer.mode == before.mode and viewer.renderer.color == before.color
        assert tuple(viewer.windows) == windows
        await send(pipe, "s", viewer, task)
        draw_rect(viewer.clipping.selection)
        viewer.clipping.add_area("polygon")
        draw_poly(viewer.clipping.selection)
        ids = [a.identity.id for a in viewer.clipping.selection.snapshots()]
        view, size, original_frame = viewer.viewport, viewer.canvas_size, viewer.frame

        def forbidden(*args, **kwargs):
            raise AssertionError("Selection style conversion must not read source pixels")
        monkeypatch.setattr(viewer.session, "sample", forbidden)
        await send(pipe, "M", viewer, task)
        assert viewer.clipping.ascii_preview
        assert viewer.clipping.preview_modes.label == "ASCII fine"
        await send(pipe, "M", viewer, task)
        assert viewer.clipping.preview_modes.label == "ASCII bold"
        await send(pipe, "T", viewer, task)
        await eventually(lambda: out.exists() and not viewer.clipping.busy, task)
        assert out.read_text().isascii() and "\x1b" not in out.read_text()
        assert [a.identity.id for a in viewer.clipping.selection.snapshots()] == ids
        assert viewer.viewport == view and viewer.canvas_size == size
        await send(pipe, "m", viewer, task)
        assert not viewer.clipping.ascii_preview
        assert viewer.clipping.selection.frame is original_frame
        await send(pipe, "m", viewer, task)
        assert viewer.clipping.preview_modes.color_mode == "shape"
        await send(pipe, "M", viewer, task)
        assert viewer.clipping.preview_modes.label == "ASCII bold"
        assert viewer.viewport == view and viewer.canvas_size == size
        assert [a.identity.id for a in viewer.clipping.selection.snapshots()] == ids


async def test_full_terminal_rename_and_save_keep_all_areas(rgb, tmp_path):
    out = tmp_path / "full.geojson"
    async with running(rgb[0], clipboard="off") as (viewer, pipe, _, task):
        await send(pipe, "f", viewer, task)
        await send(pipe, "s", viewer, task)
        draw_rect(viewer.clipping.selection, (20, 10), (25, 13))
        viewer.clipping.add_area("polygon")
        draw_poly(viewer.clipping.selection, ((30, 10), (40, 10), (30, 18)))
        ids = [a.identity.id for a in viewer.clipping.selection.snapshots()]
        size = viewer.canvas_size
        await send(pipe, "n", viewer, task)
        await send(pipe, "\x01\x0bUser area\r", viewer, task)
        assert viewer.image_only and viewer.selecting and viewer.canvas_size == size
        await send(pipe, "J", viewer, task)
        assert viewer.editing == "save-geojson" and viewer.canvas_size == size
        await send(pipe, "\x01\x0b"+str(out)+"\r", viewer, task)
        await eventually(lambda: out.exists() and not viewer.clipping.busy, task)
        assert [v["id"] for v in json.loads(out.read_text())["features"]] == ids
        assert viewer.canvas_size == size == (100, 30)


async def test_128_channel_paging_endpoints_and_mouse_selection(tmp_path):
    path = tmp_path / "many.ome.tif"
    raw = np.arange(128*32*48, dtype=np.uint32).reshape(128, 32, 48).astype(np.uint16)
    write_pyramid(path, raw, levels=2)
    async with running(path) as (viewer, pipe, output, task):
        assert len(viewer.windows) == 128
        original_samples, camera = viewer.samples, viewer.viewport
        await send(pipe, "\t\x1b[6~", viewer, task)
        assert viewer.focused_channel == viewer.channel_control.visible_channels
        assert viewer.samples is original_samples and viewer.viewport == camera
        await send(pipe, "\x1b[F", viewer, task)
        assert viewer.focused_channel == 127
        content = viewer.channel_control.create_content(31, viewer.canvas_size[1])
        text = "\n".join("".join(f[1] for f in content.get_line(i)) for i in range(content.line_count))
        assert "127" in text and "Marker-127" in text
        await send(pipe, "\r", viewer, task)
        assert viewer.selected_channels == (127,)
        await send(pipe, "\x1b[H", viewer, task)
        assert viewer.focused_channel == 0
        await send(pipe, "\x1b[6~\x1b[6~\x1b[5~", viewer, task)
        assert viewer.focused_channel == viewer.channel_control.visible_channels
        # Mouse hit testing uses the scrolled window, not absolute channel zero.
        first = viewer.channel_control.first_channel
        viewer.channel_control.mouse_handler(MouseEvent(Point(6, 1), MouseEventType.MOUSE_UP,
                                                        MouseButton.LEFT, frozenset()))
        await ready(viewer, task)
        assert viewer.focused_channel == first
        assert first in viewer.selected_channels
        # Shorter panes keep a focused row visible rather than hiding it behind instructions.
        output.size = Size(12, 80)
        viewer.application._on_resize()
        await ready(viewer, task)
        await send(pipe, "\x1b[F", viewer, task)
        assert viewer.focused_channel == 127
        content = viewer.channel_control.create_content(31, viewer.canvas_size[1])
        assert content.line_count <= viewer.canvas_size[1]
        assert viewer.channel_control.first_channel <= 127 < (
            viewer.channel_control.first_channel + viewer.channel_control.visible_channels)


@pytest.mark.parametrize("key", ["y", "Y", "i", "I", "t", "T", "j", "J"])
async def test_polygon_has_all_eight_export_bindings(rgb, tmp_path, monkeypatch, key):
    paths = {False: tmp_path / "out.txt", True: tmp_path / "out.ansi",
             "png": tmp_path / "out.png", "geojson": tmp_path / "out.geojson"}
    from ttwhy.text_clip import ClipboardDelivery
    images = []
    monkeypatch.setattr("ttwhy.viewer.clipping.clipboard_png",
                        lambda data, mode: images.append(data) or ClipboardDelivery("PNG request"))
    async with running(rgb[0], renderer=TextRenderer(mode="blocks"), clipboard="osc52",
                       output_text_clip=paths[False], output_ansi_clip=paths[True],
                       output_png_clip=paths["png"], output_geojson_clip=paths["geojson"]) as (
        viewer, pipe, output, task,
    ):
        writes = []
        monkeypatch.setattr(output, "write_raw", writes.append)
        await send(pipe, "v", viewer, task)
        draw_poly(viewer.clipping.selection, ((12, 5), (23, 5), (12, 14)))
        expected = viewer.clipping.selection.clip()
        await send(pipe, key, viewer, task)
        await eventually(lambda: not viewer.clipping.busy, task)
        typ = False if key in "yY" else True if key in "tT" else "png" if key in "iI" else "geojson"
        assert paths[typ].exists(), viewer.message
        if typ in (False, True):
            assert paths[typ].read_text() == expected.to_text(ansi=typ) + "\n"
        elif typ == "geojson":
            assert len(json.loads(paths[typ].read_text())["features"]) == 1
        else:
            with Image.open(paths[typ]) as image:
                assert image.mode == "RGBA"
        osc = [w for w in writes if w.startswith("\x1b]52;")]
        assert bool(osc or images) == (key in ("y", "i", "t", "j"))


async def test_polygon_enter_closes_then_allows_text_copy_and_keyboard_add(rgb, tmp_path):
    path = tmp_path / "keys.txt"
    async with running(rgb[0], output_text_clip=path, clipboard="off") as (viewer, pipe, _, task):
        await send(pipe, "v\x1b[H \x1b[C\x1b[C \x1b[B\x1b[B ", viewer, task)
        assert len(viewer.clipping.selection.vertices()) == 3
        assert not viewer.clipping.selection.closed
        await send(pipe, "\r", viewer, task)
        assert viewer.clipping.selection.closed and not path.exists()
        await send(pipe, "ya", viewer, task)
        await eventually(lambda: path.exists() and not viewer.clipping.busy, task)
        assert len(viewer.clipping.selection.areas) == 2
        await send(pipe, "[n\x01\x0bFirst area\r", viewer, task)
        assert viewer.clipping.selection.active_index == 0
        assert viewer.clipping.geojson_name == "First area"
        await send(pipe, "]\x1b[3~", viewer, task)
        assert len(viewer.clipping.selection.areas) == 1


async def test_pdf_multiple_areas_use_masked_screen_not_a_leaking_bounding_box(tmp_path, monkeypatch):
    from test_clipping import write_text_pdf
    from test_document_navigation import browsing
    path = write_text_pdf(tmp_path / "text.pdf")
    out = tmp_path / "masked.txt"
    async with browsing(path, pdf_max_dpi=144) as (_, viewer, pipe, _, task):
        clip = viewer.clipping
        clip.clipboard = "off"
        clip.paths[False] = out
        await send(pipe, "s", viewer, task)
        draw_rect(clip.selection, (15, 5), (22, 8))
        assert clip.source_label == "PDF text"
        clip.add_area()
        draw_rect(clip.selection, (30, 10), (35, 15))
        assert clip.source_label == "screen text"
        def prohibited(*args, **kwargs):
            raise AssertionError("Do not include PDF text between separate selected areas")
        monkeypatch.setattr("ttwhy.viewer.clipping.extract_pdf_text", prohibited)
        expected = clip.selection.clip().to_text()
        await send(pipe, "y", viewer, task)
        await eventually(lambda: out.exists() and not clip.busy, task)
        assert out.read_text() == expected + "\n"


async def test_region_snapshot_pins_names_while_saving(fake_viewer, tmp_path, monkeypatch):
    import threading
    import ttwhy.viewer.clipping as module
    clip = fake_viewer.clipping
    draw_rect(clip.selection)
    name, identity = clip.geojson_name, clip.selection.active.identity.id
    started, release = threading.Event(), threading.Event()
    original = module.geometry_feature
    def slow(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "geometry_feature", slow)
    out = tmp_path / "pinned.geojson"
    clip.submit_geojson(copy=False, path=out)
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(.01)
        assert started.is_set()
        clip.accept_name("Different later name")
        clip.add_area()
    finally:
        release.set()
    await clip._task
    feature = json.loads(out.read_text())["features"][0]
    assert feature["id"] == identity and feature["properties"]["name"] == name
