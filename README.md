# ttwhy

**Whole-slide microscopy, ordinary images, and scientific PDFs, right in your terminal.**

`ttwhy` is a zoomable, full-screen image viewer made of *actual terminal characters*. Open an OME-TIFF, SVS, or OME-Zarr slide, explore multichannel fluorescence, inspect a PNG, or page through a PDF. No browser, image server, X11, graphics protocol, or GPU required.

I had a use case for this on a remote cluster with thousands of images. Sometimes
I just needed the tissue's shape, a rough count of regions of interest, or to tell
a big chunk of tissue from a few tiny core needle biopsies. An ASCII-art
representation of a preview from the image pyramid seemed like it would do the job.

Then working in tmux, switching the active area of the terminal with either the
mouse or keyboard, got me wondering... just how far can we push a text-based image
viewer? And thus began `ttwhy`. I've been having a little fun with AI slop along
the way, and now it opens PDFs and lets me draw polygons on pathology images.
It's ridiculous, but I'm actually going to use it.

<p align="center">
  <a href="https://github.com/jason-weirather/ttwhy/blob/main/docs/media/pathology.gif"><img src="https://raw.githubusercontent.com/jason-weirather/ttwhy/main/docs/media/pathology.gif" width="900" alt="Real recording of ttwhy zooming through multiplex immunofluorescence and revealing CD8 signal in cyan on blue DAPI nuclei"></a>
</p>
<p align="center"><em>Real terminal footage: 59-channel multiplex microscopy, DAPI and CD8. Channel selection and zoom, no mock interface.</em></p>

## See it in action

These are short, automatically looping recordings of `ttwhy` itself. Click any clip to view it at full GIF size.

<table>
<tr>
<td width="50%" valign="top">
<strong>Whole-slide H&amp;E-style imagery</strong><br>
<a href="https://github.com/jason-weirather/ttwhy/blob/main/docs/media/he.gif"><img src="https://raw.githubusercontent.com/jason-weirather/ttwhy/main/docs/media/he.gif" width="440" alt="ttwhy zooming from a tissue overview to individual nuclei in a pseudo-H&E slide"></a><br>
<small>Zoom from tissue-scale context into individual nuclei.</small>
</td>
<td width="50%" valign="top">
<strong>An ordinary PNG. Featuring a cat.</strong><br>
<a href="https://github.com/jason-weirather/ttwhy/blob/main/docs/media/cat.gif"><img src="https://raw.githubusercontent.com/jason-weirather/ttwhy/main/docs/media/cat.gif" width="440" alt="ttwhy zooming into an orange-and-white cat in a sunlit windowsill illustration"></a><br>
<small>PNG, JPEG, WebP: zoom and pan familiar images, too.</small>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>A scientific PDF</strong><br>
<a href="https://github.com/jason-weirather/ttwhy/blob/main/docs/media/pdf.gif"><img src="https://raw.githubusercontent.com/jason-weirather/ttwhy/main/docs/media/pdf.gif" width="440" alt="ttwhy switching pages in a scientific PDF and zooming into a transformer architecture diagram"></a><br>
<small>Turn pages, then zoom into figures. PDF support is standard.</small>
</td>
<td width="50%" valign="top">
<strong>Polygon selections</strong><br>
<a href="https://github.com/jason-weirather/ttwhy/blob/main/docs/media/selection.gif"><img src="https://raw.githubusercontent.com/jason-weirather/ttwhy/main/docs/media/selection.gif" width="440" alt="ttwhy drawing a yellow polygon selection around a tissue core and preparing it for GeoJSON export"></a><br>
<small>Select rectangles or polygons; export GeoJSON, text, ANSI art, or PNG.</small>
</td>
</tr>
</table>

## Install

Python **3.11+**. The **distribution name** is `ttwhy-viewer`; the executable and Python import are both `ttwhy`. (The unrelated package named `ttwhy` on PyPI is **not** this project.)

Install the standalone command without changing packages in a scientific Python environment:

```bash
uv tool install ttwhy-viewer
# or: pipx install ttwhy-viewer
# or: python -m pip install ttwhy-viewer
```

From a cloned checkout, for development:

```bash
python -m pip install -e '.[dev]'
```

PDF support is **included** in the standard install. No special extra or system PDF converter is needed.

## Quickstart

```bash
ttwhy photo.png                         # PNG, JPEG, WebP, and more
ttwhy paper.pdf                         # page navigation and progressive detail
ttwhy slide.svs                         # Aperio whole-slide RGB
ttwhy slide.ome.tif                     # OME-TIFF, grayscale or multichannel
ttwhy /data/sample.ome.zarr             # local OME-Zarr

# Compose two fluorescence channels with chosen colors:
ttwhy slide.ome.tif --channel 0 --channel 2 \
  --channel-color 0 blue --channel-color 2 cyan
```

The initial view fits the image. Wheel zoom is anchored to the pointer; pan with the mouse or arrow keys. The terminal sidebar keeps channels, colors, and contrast controls at hand.

## Why a terminal viewer?

- **Whole-slide without whole-slide RAM use:** regional TIFF/SVS reads and pyramids, with bounded previews and useful zoomed-out overviews.
- **Multichannel, not just RGB:** select channels, change pseudocolors, and adjust each marker's display window and gamma without modifying source pixels.
- **PDFs and everyday images:** ordinary images use in-memory pyramids; PDFs refine progressively from a quick preview to higher-detail pages.
- **Real characters:** colored `▀` half-blocks, Braille detail, shape-oriented Unicode art, and portable ASCII modes.
- **Selection and export:** copy/save terminal text, colored ANSI, shareable PNGs, and GeoJSON polygons/rectangles.
- **Reusable Python library:** sampling, intensity normalization, composition, and rendering are available without opening the interactive terminal UI.

`ttwhy` works over ordinary terminal sessions, including environments where a desktop image viewer is inconvenient. It is an exploratory display, **not a color-managed diagnostic viewer** and not a substitute for measuring the original pixels.

## The keys you'll actually use

| Action | Key or gesture |
|---|---|
| Zoom; pan; fit image | Wheel or `+`/`-`; arrows or drag; `r`/Home |
| Next/previous PDF page or OME series | `]` / `[` or PageDown / PageUp; `p` to jump |
| Toggle a channel; focus another | Click channel/Space; `,` and `.` |
| Adjust channel color or contrast | `h` (hue); `l`/`u` (limits); `g` (gamma); `a` (auto window) |
| Switch artwork / strict ASCII | `m` / `M`; `t` toggles artwork color; `f` toggles fullscreen |
| Select region | `s` rectangle; `v` polygon |
| Export a selection | `y/Y` text, `i/I` PNG, `t/T` ANSI, `j/J` GeoJSON (copy/save) |
| Help or quit | `?` / `q` |

## From terminal annotations to full-resolution ROIs

The useful part of the joke: draw regions in `ttwhy`, save their coordinates as
GeoJSON, and let [Omeify](https://github.com/jason-weirather/omeify) extract them
from the original image. This example uses a calibrated OME-TIFF, series 0.
You also need the `omeify` command on your PATH; with isolated tool installs,
install it separately using `uv tool install omeify`.

```bash
# Open the source and choose where to save the annotations.
ttwhy slide.ome.tif --series 0 --output-geojson-clip regions.geojson
```

Zoom to the view you need, then press **`s`** for a rectangle or **`v`** for a
polygon and draw. Close a polygon with **Enter**; **`a`** adds another region in
the same view. Press **Shift+J** to save all the regions before quitting.

```bash
# Export one full-resolution image per region, with lossless compression.
omeify crop slide.ome.tif --series 0 \
  --geojson regions.geojson -o regions.ome.tif \
  --shatter by_index --compression Deflate --no-overwrite
```

This writes `regions-01.ome.tif`, `regions-02.ome.tif`, and so on, retaining all
channels from the selected series. Omit `--shatter by_index` to put the regions
into separate image series in one OME-TIFF instead.

**Full-resolution pixels, not a screenshot:** Omeify reads the original image,
not the terminal preview. Polygon crops use their **bounding rectangles**, not
polygon masks; keep the GeoJSON for the outlines. Use the same source image,
series, and orientation in both steps. Annotation placement follows the terminal
grid, so zoom in before drawing for finer boundaries.
[More crop options](https://github.com/jason-weirather/omeify/blob/main/docs/regions.md).

## Supported files

| Format | What ttwhy does |
|---|---|
| OME-TIFF | Regional, pyramidal microscopy viewing and channel composition |
| Aperio SVS | RGB whole-slide viewing using available stored levels |
| OME-Zarr | Local and HTTP(S) stores, regional chunk reads, supported NGFF multiscales |
| PNG, JPEG, WebP, BMP, TGA, PPM | Ordinary raster viewing with zoom and pan |
| GIF, APNG, animated WebP | View a selected frame (not animation playback) |
| PDF | Navigate pages; progressively improve raster detail; copy embedded text from a rectangular selection |

For advanced flags, channel/series semantics, color modes, cache limits, clipboard behavior, and format caveats, use the **[full reference](https://github.com/jason-weirather/ttwhy/blob/main/docs/reference.md)**. Architecture notes are in [docs/architecture.md](https://github.com/jason-weirather/ttwhy/blob/main/docs/architecture.md).

## Try a synthetic microscopy example

The repo includes a deterministic fixture generator that creates public-safe, synthetic tissue-like RGB and multichannel OME-TIFFs. It is a renderer demonstration, not a biological simulation.

```bash
python examples/make_demo.py --output-directory /tmp/ttwhy-demo
ttwhy /tmp/ttwhy-demo/synthetic-mif.ome.tif --channel 0 --channel 2
```

## Python API

```python
import numpy as np
from ttwhy import TextRenderer

pixels = np.linspace(0, 1, 256**2).reshape(256, 256)
frame = TextRenderer(mode="detail", color=False).render(pixels, columns=80)
print(frame.to_text())
```

## Optional remote sources

Local files are the main workflow. `ttwhy` also supports HTTP(S) images/PDFs (bounded download) and HTTP(S) OME-Zarr (regional chunk reads). For native S3 or S3-compatible stores, install `ttwhy-viewer[s3]` and use your normal AWS credentials or an endpoint override. No S3 setup is needed for local images, slides, or PDFs. [Remote input reference](https://github.com/jason-weirather/ttwhy/blob/main/docs/reference.md#local-and-remote-inputs).

## Release media and provenance

These GIFs are edited recordings of the real `ttwhy` terminal viewer. Showcase sources:

- **mIF and pseudo-H&E:** Tan, Y. & Kempchen, T. (2024), [*SPACEc demo dataset*](https://doi.org/10.5061/dryad.brv15dvj1), Dryad. The pseudo-H&E is computationally derived, not a conventional H&E stain.
- **PDF:** Vaswani, A. et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NeurIPS.
- **Cat:** Illustration generated with ChatGPT (OpenAI).

## License

Apache-2.0. See [LICENSE](https://github.com/jason-weirather/ttwhy/blob/main/LICENSE).
