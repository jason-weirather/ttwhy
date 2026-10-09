"""Real codecs and the same source/session/CLI path as OME-TIFF, without converters."""

import builtins
import os
import subprocess
import sys

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image as PILImage
from PIL import features

from ttwhy import IntensityWindow, TextRenderer, Viewport, ViewSession, open_image
from ttwhy.cli import main
from ttwhy.image_input import image_kind
from ttwhy.image_pyramid import image_pyramid


def write_pdf(path, *, rotated=False):
    """Independent tiny vector PDF fixture; no PDF-writing test dependency."""
    commands = [b"1 0 0 rg 0 0 36 48 re f", b"0 0 1 rg 0 0 72 48 re f"]
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Count 2 /Kids [3 0 R 5 0 R] >>"]
    for index, content in enumerate(commands):
        rotation = b" /Rotate 90" if rotated else b""
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 48] "
                       b"/Resources << >> /Contents " + str(4 + index * 2).encode()
                       + b" 0 R" + rotation + b" >>")
        objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
                       + content + b"\nendstream")
    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)
    return path


@pytest.fixture
def raster(tmp_path):
    y, x = np.mgrid[:65, :97]
    pixels = np.stack([x * 2, y * 3, (x + y) % 256], axis=-1).astype(np.uint8)
    path = tmp_path / "ordinary.png"
    PILImage.fromarray(pixels).save(path)
    return path, pixels


@pytest.mark.parametrize("shape", [(5, 7), (1, 17), (19, 1), (1, 1), (8, 16), (7, 11, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int16, np.float32])
def test_pyramid_against_independent_base_rectangle_means(shape, dtype):
    base = (np.arange(np.prod(shape)).reshape(shape) % 239).astype(dtype)
    if dtype == np.float32:
        base.flat[::11] = np.nan
    original = base.copy()
    levels = image_pyramid(base)
    np.testing.assert_array_equal(base, original)
    assert not np.shares_memory(base, levels[0]) and base.flags.writeable
    assert levels[-1].shape[:2] == (1, 1)
    for index, level in enumerate(levels):
        scale = 2**index
        assert level.dtype == base.dtype and not level.flags.writeable
        assert level.shape[:2] == ((shape[0] + scale - 1) // scale,
                                  (shape[1] + scale - 1) // scale)
        for y in range(level.shape[0]):
            for x in range(level.shape[1]):
                part = base[y*scale:(y+1)*scale, x*scale:(x+1)*scale].astype(float)
                count = np.isfinite(part).sum(axis=(0, 1))
                expected = np.divide(np.nansum(part, axis=(0, 1)), count,
                                     out=np.full(np.shape(count), np.nan), where=count > 0)
                if base.dtype.kind in "ui":
                    expected = np.rint(expected)
                np.testing.assert_allclose(level[y, x], expected, rtol=1e-6, equal_nan=True)


def test_pyramid_avoids_cumulative_rounding_and_odd_edge_weighting():
    raw = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 255]], np.uint8)
    levels = image_pyramid(raw)
    # Equal weighting of the four L1 pixels would incorrectly give 64, not 28.
    assert levels[2][0, 0] == 28
    raw = np.zeros((4, 4), np.uint8)
    raw[0, 0] = raw[0, 2] = raw[2, 0] = raw[2, 2] = 2
    raw[0, 1] = 1
    levels = image_pyramid(raw)
    assert levels[2][0, 0] == round(raw.mean()) == 1


@pytest.mark.parametrize("format,extension", [
    ("PNG", ".PNG"), ("JPEG", ".jpeg"), ("WEBP", ".webp"), ("BMP", ".bmp"),
    ("GIF", ".gif"), ("PPM", ".ppm"), ("TGA", ".tga"), ("ICO", ".ico"),
])
def test_real_raster_formats_and_native_rgb(tmp_path, format, extension):
    if format == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow build has no WebP codec")
    path = tmp_path / ("input" + extension)
    y, x = np.mgrid[:32, :32]
    original = np.stack([x * 8, y * 8, (x + y) * 4], -1).astype(np.uint8)
    options = {"lossless": True} if format == "WEBP" else {}
    PILImage.fromarray(original).save(path, format=format, **options)
    with PILImage.open(path) as decoded:
        expected = np.array(decoded.convert("RGB"))
    before = set(tmp_path.iterdir())
    with open_image(path) as image, ViewSession(image) as session:
        assert image.image_type == "rgb" and image.channel_names == ("RGB",)
        assert image.pixel_size is None
        np.testing.assert_array_equal(image.asarray(), expected)
        assert len(image.levels) > 1
        assert all(level.downsample_yx == (2**i, 2**i) for i, level in enumerate(image.levels))
        samples = session.sample(session.plan(Viewport(0, 0, 32, 32), rows=32, columns=32))
        assert session.prepare_styles(samples)[0].window == IntensityWindow(0, 255)
    assert not image.is_open and not session.is_open
    assert set(tmp_path.iterdir()) == before  # No intermediate files, even while opening.


@pytest.mark.parametrize("background,expected", [
    (None, [[255, 255, 255], [255, 127, 127], [0, 255, 0]]),
    ((0, 0, 0), [[0, 0, 0], [128, 0, 0], [0, 255, 0]]),
])
def test_rgba_matte_not_hidden_rgb(tmp_path, background, expected):
    raw = np.array([[[0, 0, 255, 0], [255, 0, 0, 128], [0, 255, 0, 255]]], np.uint8)
    path = tmp_path / "alpha.png"
    PILImage.fromarray(raw).save(path)
    with open_image(path, background=background) as image:
        np.testing.assert_array_equal(image.asarray(), [expected])


def test_palette_and_grayscale_transparency(tmp_path):
    path = tmp_path / "palette.png"
    indexed = PILImage.fromarray(np.array([[0, 1]], np.uint8)).convert("P")
    indexed.putpalette([0, 0, 255, 255, 0, 0] + [0] * 762)
    indexed.save(path, transparency=0)
    with open_image(path) as image:
        np.testing.assert_array_equal(image.asarray(), [[[255, 255, 255], [255, 0, 0]]])
    gray = tmp_path / "gray.png"
    PILImage.fromarray(np.array([[0, 128]], np.uint8)).save(gray, transparency=0)
    with open_image(gray) as image:
        np.testing.assert_array_equal(image.asarray(), [[[255, 255, 255], [128, 128, 128]]])


@pytest.mark.parametrize("mode", ["L", "1", "CMYK"])
def test_other_color_modes_are_display_rgb(tmp_path, mode):
    original = PILImage.new("RGB", (5, 3), (60, 120, 180)).convert(mode)
    path = tmp_path / ("mode.jpg" if mode == "CMYK" else "mode.png")
    original.save(path)
    with PILImage.open(path) as decoded:
        expected = np.array(decoded.convert("RGB"))
    with open_image(path) as image:
        assert image.image_type == "rgb"
        np.testing.assert_array_equal(image.asarray(), expected)


@pytest.mark.parametrize("extension", [".png", ".jpg"])
def test_exif_orientation_before_pyramid(tmp_path, extension):
    path = tmp_path / ("rotated" + extension)
    pixels = np.arange(4*7*3, dtype=np.uint8).reshape(4, 7, 3)
    exif = PILImage.Exif()
    exif[274] = 6
    PILImage.fromarray(pixels).save(path, exif=exif)
    original_bytes = path.read_bytes()
    with PILImage.open(path) as decoded:
        expected = np.rot90(np.array(decoded.convert("RGB")), k=3)
    with open_image(path) as image:
        np.testing.assert_array_equal(image.asarray(), expected)
        assert image.levels[0].spatial_shape == (7, 4)
    assert path.read_bytes() == original_bytes


def test_uint16_grayscale_keeps_all_native_values(tmp_path):
    raw = (np.arange(35).reshape(5, 7) * 1800).astype(np.uint16)
    path = tmp_path / "16bit.png"
    PILImage.fromarray(raw).save(path)
    with open_image(path) as image, ViewSession(image) as session:
        assert image.dtype == np.dtype("uint16") and image.image_type == "multichannel"
        np.testing.assert_array_equal(image.asarray(), raw)
        sample = session.sample(session.plan(Viewport(0, 0, 7, 5), rows=5, columns=7))
        np.testing.assert_array_equal(sample.values[0], raw)
        assert session.prepare_styles(sample)[0].window.upper > 255


@pytest.mark.parametrize("format,extension", [("GIF", ".gif"), ("PNG", ".png"), ("WEBP", ".webp")])
def test_frames_are_selected_not_played(tmp_path, format, extension):
    if format == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow build has no WebP codec")
    path = tmp_path / ("animation" + extension)
    frames = [PILImage.new("RGB", (8, 6), color) for color in ("red", "blue")]
    frames[0].save(path, format=format, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, **({"lossless": True} if format == "WEBP" else {}))
    for index, color in enumerate(([255, 0, 0], [0, 0, 255])):
        with open_image(path, frame=index) as image:
            np.testing.assert_array_equal(image.asarray(), np.full((6, 8, 3), color, np.uint8))
    with pytest.raises(IndexError, match="Frame 2"):
        with open_image(path, frame=2):
            pass


def test_magic_detection_extensionless_and_renamed(raster, tmp_path):
    alias = tmp_path / "not-an-extension"
    alias.write_bytes(raster[0].read_bytes())
    assert image_kind(alias) == "raster"
    with open_image(alias) as image:
        np.testing.assert_array_equal(image.asarray(), raster[1])


def test_tiff_keeps_original_reader_and_never_constructs_ram_pyramid(rgb, monkeypatch):
    from omeify import OMETiffReader

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary-image ingress must not run on a TIFF")

    monkeypatch.setattr("ttwhy.image_input.image_pyramid", forbidden)
    monkeypatch.setattr(PILImage, "open", forbidden)
    with open_image(rgb[0]) as image:
        assert isinstance(image, OMETiffReader)
        np.testing.assert_array_equal(image.asarray(), rgb[1][0])


def test_plain_tiff_does_not_fall_through_to_pillow(tmp_path, monkeypatch):
    import tifffile

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((3, 4, 3), np.uint8), photometric="rgb", metadata=None)

    def forbidden(*args, **kwargs):
        raise AssertionError("TIFF must not silently switch decoders")

    monkeypatch.setattr(PILImage, "open", forbidden)
    with pytest.raises(ValueError, match="OME metadata"):
        with open_image(path):
            pass


def test_large_image_fits_and_zoom_recovers_original_pixels(tmp_path):
    y, x = np.mgrid[:1601, :2403]
    raw = np.stack([x % 256, y % 256, (x + y) % 256], -1).astype(np.uint8)
    path = tmp_path / "large.png"
    PILImage.fromarray(raw).save(path)
    with open_image(path) as image, ViewSession(image, max_read_pixels=60_000) as session:
        renderer = TextRenderer(mode="blocks")
        fit, limited = session.fit(rows=30, columns=80, renderer=renderer)
        assert not limited
        sr, sc = renderer.sample_shape(rows=30, columns=80)
        plan = session.plan(fit, rows=sr, columns=sc)
        assert plan.level.index > 0 and plan.source_samples <= 60_000
        assert not plan.level.approximate
        far = session.sample(plan)
        assert np.isfinite(far.values).any()
        close_plan = session.plan(Viewport(1001, 702, 80, 60), rows=60, columns=80)
        assert close_plan.level.index == 0
        np.testing.assert_array_equal(session.sample(close_plan).values, raw[702:762, 1001:1081])
        # A partial final coarse pixel is correctly clipped at the base edge.
        outside = session.plan(Viewport(2403, 0, 64, 1601), rows=10, columns=1)
        assert np.isnan(session.sample(outside).values).all()


@pytest.mark.parametrize("mode,ansi", [("auto", False), ("auto", True), ("blocks", True),
                                      ("simple", False), ("detail", False), ("shape", False)])
def test_raster_cli_modes(raster, mode, ansi):
    args = [str(raster[0]), "--snapshot", "--mode", mode, "--columns", "31", "--rows", "13"]
    if ansi:
        args.append("--ansi")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "bounded center crop" not in result.output
    assert "Pyramid scale unknown" not in result.output
    if ansi:
        assert "▀" in result.stdout and "38;2;" in result.stdout and "48;2;" in result.stdout
    else:
        assert "\x1b" not in result.stdout
        assert all(len(row) == 31 for row in result.stdout.splitlines())
    assert len(result.stdout.splitlines()) == 13


@pytest.mark.parametrize("options", [["--page", "1"], ["--pdf-max-dpi", "144"],
                                      ["--series", "1"], ["--frame", "9"],
                                      ["--channel", "1"], ["--pdf-max-dpi", "nan"],
                                      ["--pdf-max-dpi", "inf"]])
def test_wrong_raster_options_are_clean_cli_errors(raster, options):
    result = CliRunner().invoke(main, [str(raster[0]), "--snapshot", *options])
    assert result.exit_code != 0 and "Error:" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("options", [{"frame": -1}, {"frame": True}, {"page": 0},
                                      {"page": 1.2}, {"series": True}, {"pdf_dpi": float("nan")},
                                      {"background": (256, 0, 0)}])
def test_invalid_library_parameters(raster, options):
    with pytest.raises(ValueError):
        with open_image(raster[0], **options):
            pass


def test_non_ome_options_rejected_for_tiff(rgb):
    for flag, value in [("--page", "1"), ("--frame", "0"), ("--pdf-max-dpi", "144"),
                        ("--background", "white")]:
        result = CliRunner().invoke(main, [str(rgb[0]), "--snapshot", flag, value])
        assert result.exit_code != 0 and "do not apply" in result.output


@pytest.mark.parametrize("filename,content", [
    ("vector.svg", b'<svg xmlns="http://www.w3.org/2000/svg"/>'),
    ("script.eps", b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n"),
    ("broken.png", b"not an image"),
])
def test_unsupported_formats_cannot_launch_external_decoder(
    tmp_path, monkeypatch, filename, content,
):
    path = tmp_path / filename
    path.write_bytes(content)

    def forbidden(*args, **kwargs):
        raise AssertionError("No external converter may run")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = CliRunner().invoke(main, [str(path), "--snapshot"])
    assert result.exit_code != 0 and "Unsupported or unrecognized image" in result.output


def test_raster_snapshot_preflight_and_input_alias_protection(raster, tmp_path, monkeypatch):
    path = raster[0]
    alias = tmp_path / "alias.txt"
    os.link(path, alias)
    existing = tmp_path / "keep.txt"
    existing.write_text("untouched")

    def forbidden(*args, **kwargs):
        raise AssertionError("Pixel decoding must happen after output preflight")

    monkeypatch.setattr(PILImage, "open", forbidden)
    runner = CliRunner()
    for target in (path, alias):
        result = runner.invoke(main, [str(path), "--snapshot", "-o", str(target)])
        assert result.exit_code != 0 and "must not be the input" in result.output
    result = runner.invoke(main, [str(path), "--snapshot", "-o", str(existing), "--no-overwrite"])
    assert result.exit_code != 0 and "already exists" in result.output
    assert existing.read_text() == "untouched"


def test_array_import_does_not_import_file_decoders_or_frontend():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split(".")[0] not in ("PIL", "pypdfium2", "omeify", "prompt_toolkit"), name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ttwhy import TextRenderer, open_image
assert callable(open_image)
TextRenderer(color=False).render([[0.0, 1.0]])
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pdf_missing_extra_is_actionable_but_rasters_still_work(raster, tmp_path, monkeypatch):
    path = write_pdf(tmp_path / "optional.pdf")
    original = builtins.__import__

    def without_pdf(name, *args, **kwargs):
        if name == "pypdfium2":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pdf)
    with open_image(raster[0]) as image:
        assert image.is_open
    result = CliRunner().invoke(main, [str(path), "--snapshot"])
    assert result.exit_code != 0 and 'pip install "ttwhy[pdf]"' in result.output


@pytest.mark.parametrize("dpi", [72, 144, 300])
def test_pdf_actual_pixels_and_fixed_resolution(tmp_path, dpi):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "pages.pdf")
    before = path.read_bytes()
    with open_image(path, page=2, pdf_dpi=dpi) as image, ViewSession(image) as session:
        assert image.image_type == "rgb" and image.dtype == np.uint8
        assert image.pixel_size is None
        assert image.levels[0].spatial_shape == (int(np.ceil(48*dpi/72)), dpi)
        values = image.read_region(2, 4, 2, 4)
        np.testing.assert_array_equal(values, np.full((2, 2, 3), [0, 0, 255], np.uint8))
        # The native PDF bitmap has already been closed. Its pixels remain owned.
        samples = session.sample(session.plan(Viewport(2, 2, 2, 2), rows=2, columns=2))
        np.testing.assert_array_equal(samples.values, values)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_pdf_default_page_matte_rotation_and_renamed_file(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "no_extension")
    assert image_kind(path) == "pdf"
    with open_image(path) as image:
        data = image.asarray()
        np.testing.assert_array_equal(data[10, 10], [255, 0, 0])
        np.testing.assert_array_equal(data[10, 100], [255, 255, 255])
    with open_image(path, background=(0, 0, 0)) as image:
        np.testing.assert_array_equal(image.asarray()[10, 100], [0, 0, 0])
    rotated = write_pdf(tmp_path / "rotated.pdf", rotated=True)
    with open_image(rotated) as image:
        assert image.levels[0].spatial_shape == (144, 96)


def test_pdf_cli_and_bad_page_do_not_write_output(tmp_path):
    pytest.importorskip("pypdfium2")
    path = write_pdf(tmp_path / "pages.pdf")
    out = tmp_path / "preview.ansi"
    runner = CliRunner()
    result = runner.invoke(main, [str(path), "--snapshot", "--page", "2", "--pdf-max-dpi", "144",
                                  "--ansi", "--columns", "24", "--rows", "8", "-o", str(out)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "38;2;0;0;255" in out.read_text() and "▀" in out.read_text()
    original = out.read_bytes()
    result = runner.invoke(main, [str(path), "--snapshot", "--page", "3", "-o", str(out)])
    assert result.exit_code != 0 and "choose 1..2" in result.output
    assert out.read_bytes() == original
    result = runner.invoke(main, [str(path), "--snapshot", "--frame", "0"])
    assert result.exit_code != 0 and "PDF uses --page" in result.output


def test_corrupt_pdf_and_huge_dpi_are_clean_errors(tmp_path):
    pytest.importorskip("pypdfium2")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nnot a real pdf")
    result = CliRunner().invoke(main, [str(broken), "--snapshot"])
    assert result.exit_code != 0 and "Error:" in result.output
    path = write_pdf(tmp_path / "real.pdf")
    result = CliRunner().invoke(main, [str(path), "--snapshot", "--pdf-max-dpi", "1e300"])
    assert result.exit_code != 0 and "too large" in result.output


def test_pillow_large_image_guard_remains_enabled(raster, monkeypatch):
    original = PILImage.MAX_IMAGE_PIXELS
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 10)
    result = CliRunner().invoke(main, [str(raster[0]), "--snapshot"])
    assert result.exit_code != 0 and "Pillow rejected" in result.output
    assert PILImage.MAX_IMAGE_PIXELS == 10
    assert original is not None
