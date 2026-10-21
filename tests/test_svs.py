"""Aperio SVS ingress uses Omeify's regional RGB reader, never an eager raster."""

import sys
from types import ModuleType

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from PIL import Image as PILImage

from ttwhy import TextRenderer, ViewSession, open_image
from ttwhy.cli import main
from ttwhy.image_document import ImageDocument
from ttwhy.image_input import image_kind


def synthetic_svs(path):
    """Independently write a small, TIFF-valid Aperio-style RGB pyramid."""
    y, x = np.mgrid[:128, :160]
    base = np.stack([x % 256, y * 2, (x + 2 * y) % 256], axis=-1).astype(np.uint8)
    arrays = [base, base[::2, ::2], base[::4, ::4]]
    description = (
        "Aperio Image Library v10.0.51 160x128 "
        "[0,0 160x128] (16x16) Deflate/RGB"
        "|AppMag = 20|MPP = 0.5000|Filename = private-slide"
    )
    with tifffile.TiffWriter(path) as writer:
        # An Aperio SVS uses top-level TIFF pages, NOT SubIFD pyramids.
        # IFD 0 is the base; IFD 1 is the associated thumbnail; subsequent
        # tiled non-reduced IFDs describe the baseline's downsample levels.
        writer.write(
            arrays[0], photometric="rgb", tile=(16, 16), compression="deflate",
            description=description, metadata=None,
            resolution=(20000, 20000), resolutionunit="CENTIMETER",
        )
        writer.write(
            arrays[-1], photometric="rgb", compression="deflate", metadata=None,
        )
        for level, pixels in enumerate(arrays[1:], 1):
            writer.write(
                pixels, photometric="rgb", tile=(16, 16), compression="deflate",
                metadata=None,
                resolution=(20000 / 2**level, 20000 / 2**level),
                resolutionunit="CENTIMETER",
            )
    with tifffile.TiffFile(path) as tiff:
        assert tiff.is_svs, "Fixture must be recognized as an Aperio SVS TIFF"
        assert len(tiff.series[0].levels) == len(arrays)
    return arrays


def test_independent_aperio_fixture_has_stored_levels(tmp_path):
    path = tmp_path / "pyramid.svs"
    arrays = synthetic_svs(path)
    with tifffile.TiffFile(path) as tiff:
        assert tiff.is_svs and len(tiff.series) == 2
        assert len(tiff.series[0].levels) == len(arrays)
        for level, expected in zip(tiff.series[0].levels, arrays, strict=True):
            np.testing.assert_array_equal(level.asarray(), expected)


@pytest.mark.parametrize("suffix", [".svs", ".SVS"])
def test_svs_tiff_magic_selects_aperio_not_ome_or_pillow(tmp_path, monkeypatch, suffix):
    """Routing works without depending on an actual decoder or full-slide input."""
    path = tmp_path / ("synthetic" + suffix)
    path.write_bytes(b"II*\x00some synthetic bytes")
    assert image_kind(path) == "tiff"
    calls = []
    sentinel = object()

    class Aperio:
        def __init__(self, path_arg, *, series):
            calls.append((path_arg, series))

        def __enter__(self):
            return sentinel

        def __exit__(self, *args):
            calls.append("closed")

    omeify = ModuleType("omeify")
    omeify.AperioSVSReader = Aperio
    monkeypatch.setitem(sys.modules, "omeify", omeify)

    def no_eager_decode(*args, **kwargs):
        raise AssertionError("A TIFF must not be eagerly decoded as an ordinary raster")

    monkeypatch.setattr("ttwhy.image_input.image_pyramid", no_eager_decode)
    monkeypatch.setattr(PILImage, "open", no_eager_decode)
    with open_image(path, series=2) as image:
        assert image is sentinel
    assert calls == [(path, 2), "closed"]


def test_svs_suffix_does_not_override_non_tiff_content(tmp_path):
    path = tmp_path / "actually-a-png.svs"
    pixels = np.full((4, 5, 3), (18, 67, 142), dtype=np.uint8)
    PILImage.fromarray(pixels).save(path, format="PNG")
    assert image_kind(path) == "raster"
    # This test checks the signature before the vendor reader is ever selected.
    assert path.read_bytes().startswith(b"\x89PNG")


def test_real_svs_regional_pyramid_and_rgb_semantics(tmp_path, monkeypatch):
    pytest.importorskip("omeify")
    from omeify import AperioSVSReader

    path = tmp_path / "pyramid.SVS"
    levels = synthetic_svs(path)
    before = path.read_bytes()

    def no_eager_decode(*args, **kwargs):
        raise AssertionError("Do not build a RAM pyramid or use Pillow for an SVS")

    monkeypatch.setattr("ttwhy.image_input.image_pyramid", no_eager_decode)
    monkeypatch.setattr(PILImage, "open", no_eager_decode)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", no_eager_decode)
    with open_image(path) as image, ViewSession(image, max_read_pixels=10_000) as session:
        assert isinstance(image, AperioSVSReader)
        assert image.image_type == "rgb" and image.channel_names == ("RGB",)
        assert image.dtype == np.dtype("uint8")
        assert image.pixel_size.x == pytest.approx(0.5)
        assert image.pixel_size.y == pytest.approx(0.5)
        assert len(image.levels) == 3
        for index, pixels in enumerate(levels):
            np.testing.assert_array_equal(
                image[0].read_region(3, 11, 4, 13, level=index), pixels[3:11, 4:13],
            )
            assert image.levels[index].downsample_yx == (2**index, 2**index)

        # With a stored coarse level, fitting never allocates the entire base RGB slide.
        renderer = TextRenderer(mode="blocks")
        viewport, limited = session.fit(rows=12, columns=24, renderer=renderer)
        assert not limited
        shape = renderer.sample_shape(rows=12, columns=24)
        plan = session.plan(viewport, rows=shape[0], columns=shape[1])
        assert plan.level.index > 0 and not plan.level.approximate
        assert plan.source_samples <= session.max_read_pixels
        samples = session.sample(plan)
        assert samples.values.shape == (*shape, 3)
        assert session.prepare_styles(samples)[0].window.upper == 255
        assert len(session.render(samples, styles=session.prepare_styles(samples),
                                  renderer=renderer, rows=12, columns=24).characters) == 12
    assert not image.is_open and path.read_bytes() == before


def test_svs_snapshot_and_tiff_series_navigation(tmp_path):
    pytest.importorskip("omeify")
    path = tmp_path / "pyramid.svs"
    synthetic_svs(path)
    document = ImageDocument(path)
    assert document.kind == "tiff" and document.label == "series"
    initial = document.initialize()
    try:
        assert document.first_index == 0 and document.count >= 1
        assert initial.image.image_type == "rgb"
        assert initial.image.series_names[0] is None
        assert document.count == 2  # Baseline slide and associated thumbnail.
        thumbnail = document.load(1)
        try:
            assert thumbnail.image.levels[0].spatial_shape == (32, 40)
            assert thumbnail.image.image_type == "rgb"
        finally:
            thumbnail.release()
    finally:
        initial.release()
    assert initial.image is None

    result = CliRunner().invoke(main, [str(path), "--snapshot", "--mode", "blocks",
                                       "--ansi", "--columns", "20", "--rows", "10"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "\x1b[" in result.stdout and "▀" in result.stdout
    assert len(result.stdout.splitlines()) == 10
    assert "Pyramid scale unknown" not in result.output


@pytest.mark.parametrize("options", [{"page": 1}, {"frame": 0}, {"pdf_dpi": 144},
                                     {"background": (255, 255, 255)}])
def test_svs_rejects_non_tiff_options(tmp_path, options):
    path = tmp_path / "anything.svs"
    path.write_bytes(b"II*\x00invalid tiff payload")
    with pytest.raises(ValueError, match="do not apply"):
        with open_image(path, **options):
            pass


def test_non_aperio_tiff_renamed_svs_fails_instead_of_falling_back(tmp_path):
    pytest.importorskip("omeify")
    path = tmp_path / "not-really-aperio.svs"
    tifffile.imwrite(path, np.zeros((32, 32, 3), np.uint8), photometric="rgb",
                     metadata=None)
    with pytest.raises(ValueError, match="Aperio SVS source"):
        with open_image(path):
            pass
