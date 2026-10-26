"""GeoJSON geometry, terminal selection, and explicit export without an image decoder."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from twhy import TextImage, TextRenderer, Viewport
from twhy.geojson_clip import dumps, map_cell_point, polygon_feature, rectangle_feature
from twhy.viewer.clipping import clip_options
from twhy.viewer.rendering import terminal_lines
from twhy.viewer.selection import PolygonSelection, RectangleSelection
from twhy.viewer.viewer import Viewer


@pytest.fixture
def frame():
    return TextImage(np.full((4, 6), "▀", dtype="U1"))


def test_rectangle_uses_pixel_edges_and_closes_ring(frame):
    selection = RectangleSelection(frame, terminal_lines(frame))
    selection.start_drag(4, 2)
    selection.drag(1, 0, finish=True)
    assert selection.bounds == (0, 3, 1, 5)
    feature = rectangle_feature(bounds=selection.bounds, viewport=Viewport(100, 200, 60, 40),
                                frame_shape=frame.shape, raster_shape=(400, 800), name="A box")
    ring = feature["geometry"]["coordinates"][0]
    assert ring == [[110., 200.], [150., 200.], [150., 230.], [110., 230.], [110., 200.]]
    assert feature["properties"]["name"] == "A box"
    assert feature["properties"]["selection_type"] == "rectangle"
    assert json.loads(dumps(feature))["type"] == "FeatureCollection"


def test_polygon_clicks_arrows_undo_and_export(frame):
    selection = PolygonSelection(frame, terminal_lines(frame))
    selection.start_drag(0, 0)
    selection.drag(0, 0, finish=True)
    selection.move(5, 0)
    selection.mark()
    selection.move(0, 3)
    selection.mark()
    assert selection.vertices() == [(0, 0), (5, 0), (5, 3)]
    assert len(selection.row(0)) > 0
    feature = polygon_feature(vertices=selection.vertices(), viewport=Viewport(10, 20, 60, 40),
                              frame_shape=frame.shape, raster_shape=(400, 800), name="region")
    ring = feature["geometry"]["coordinates"][0]
    assert ring == [[15., 25.], [65., 25.], [65., 55.], [15., 25.]]
    assert feature["properties"]["name"] == "region"
    selection.swap_corner()
    assert selection.vertices() == [(0, 0), (5, 0)]
    with pytest.raises(ValueError, match="Close the polygon"):
        selection.clip()
    with pytest.raises(ValueError, match="at least three"):
        polygon_feature(vertices=selection.vertices(), viewport=Viewport(0, 0, 6, 4),
                        frame_shape=frame.shape, raster_shape=(4, 6))


def test_letterboxing_clamps_to_image_without_using_terminal_pixels(frame):
    view = Viewport(-20, -30, 100, 100)
    assert map_cell_point(0, 0, viewport=view, frame_shape=frame.shape,
                          raster_shape=(40, 60)) == (0.0, 0.0)
    assert map_cell_point(6, 4, viewport=view, frame_shape=frame.shape,
                          raster_shape=(40, 60)) == (60.0, 40.0)


def test_geojson_destination_preflight_and_collision(tmp_path):
    source = tmp_path / "slide.ome.tiff"
    source.write_bytes(b"image")
    out = tmp_path / "clip.geojson"
    clip_options(source_path=source, output_geojson_clip=out)
    with pytest.raises(ValueError, match="different"):
        clip_options(output_geojson_clip=out, output_text_clip=out)
    with pytest.raises(ValueError, match="input image"):
        clip_options(source_path=source, output_geojson_clip=source)


@pytest.mark.asyncio
async def test_rectangle_save_and_named_polygon_copy(frame, tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    fake_image = SimpleNamespace(
        info=SimpleNamespace(shape=(400, 800), channels=("RGB",), is_rgb=True),
        _select_channels=lambda channels: tuple(channels),
    )
    viewer = Viewer(fake_image, source_path=source, renderer=TextRenderer(mode="blocks"),
                    clipboard="osc52")
    viewer.frame = frame
    viewer.cell_lines = terminal_lines(frame)
    viewer.canvas_size = (frame.shape[1], frame.shape[0])
    viewer.viewport = Viewport(100, 200, 60, 40)
    viewer.generation = viewer.displayed_generation = 1
    written = []
    viewer.application = SimpleNamespace(
        layout=SimpleNamespace(focus=lambda _: None),
        invalidate=lambda: None,
        create_background_task=asyncio.create_task,
        is_running=True,
        output=SimpleNamespace(write_raw=written.append, flush=lambda: None),
    )
    try:
        viewer.clipping.start("rectangle")
        rect = viewer.clipping.selection
        rect.start_drag(1, 0)
        rect.drag(4, 2, finish=True)
        path = tmp_path / "out.geojson"
        viewer.clipping.submit_geojson(copy=False, path=path)
        await viewer.clipping._task
        value = json.loads(path.read_text())
        assert value["features"][0]["geometry"]["coordinates"][0][0] == [110., 200.]
        assert value["features"][0]["properties"]["name"] == viewer.clipping.geojson_name
        assert "image" not in value["features"][0]["properties"]["name"]
        assert value["features"][0]["properties"]["name_generated"] is True
        viewer.clipping.start_polygon()
        polygon = viewer.clipping.selection
        for cell in [(0, 0), (3, 0), (3, 3)]:
            polygon.start_drag(*cell)
            polygon.drag(*cell, finish=True)
        polygon.finish()
        viewer.clipping.geojson_name = "Nucleus A"
        viewer.clipping.submit_geojson()
        await viewer.clipping._task
        assert written and written[-1].startswith("\x1b]52;c;")
        import base64
        raw = base64.b64decode(written[-1].split(";")[-1].rstrip("\x07")).decode("utf8")
        geo = json.loads(raw)
        assert len(geo["features"]) == 2
        assert geo["features"][1]["properties"]["name"] == "Nucleus A"
        assert len(geo["features"][1]["geometry"]["coordinates"][0]) == 4
    finally:
        viewer.clipping.close()
