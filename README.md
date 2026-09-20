# ttwhy

A whole slide pathology image viewer in text; because sometimes the answer to “why is this in the terminal?” is “yes.”

`ttwhy` is a library-first image-to-text renderer and a full-screen OME-TIFF viewer.
The default is actual ASCII density art. Optional Unicode half-blocks carry two
RGB samples per terminal cell. No terminal bitmap protocol, browser, image server,
X11, or GPU is required.

This is an early exploratory viewer, not a diagnostic display or a color-managed
replacement for a microscopy workstation. It is useful for navigation and visual
checks, not pixel measurements from the rendered text.

## Install

Python 3.13 or newer is required. Start in an environment containing a compatible
Omeify release. ttwhy 0.2 uses **Omeify 0.18** and requires
`omeify>=0.18,<0.19`; it does not vendor its code.

```bash
mamba env create -f environment.yml
mamba activate ttwhy_env

# Install your Omeify checkout when it is not available from your package index.
python -m pip install -e /path/to/omeify
python -m pip install -e '.[dev]'

ttwhy --version
ttwhy slide.ome.tif
```

Only NumPy, SciPy, Click, Omeify, prompt_toolkit, and wcwidth are direct runtime
dependencies. The library's array-rendering operations do not import the terminal
frontend or open any files. File opening does not trigger conversion, model
inference, whole-slide pixel scans, or pyramid construction.

## Viewer

```bash
ttwhy slide.ome.tif

# Channel indices are zero-based. Repeating the option selects a composite.
ttwhy multiplex.ome.tif --channel 0 --channel 2

# Raw intensity limits, not percentiles. These initialize all selected channels.
ttwhy multiplex.ome.tif --channel 0 --lower 250 --upper 12000

# RGB remains one logical channel containing its three color samples.
ttwhy brightfield.ome.tif --mode blocks

# Select one OME image series, or start at a particular base-pixel field of view.
ttwhy slide.ome.tif --series 1
ttwhy slide.ome.tif --region 10000 12000 2500 1800
```

The image occupies the main pane. The sidebar shows channel selection and the
focused channel's lower limit, upper limit, and gamma. It can be hidden with `c`,
and automatically hides below 70 terminal columns. The terminal's own pane can
still be split by tmux; ttwhy does not implement a shell or manage tmux sessions.

| Control | Action |
|---|---|
| Mouse wheel over the image; `+` / `-` | Zoom, anchored to the pointer for the wheel |
| Arrow keys; left-button drag | Pan the image |
| `r` or Home | Fit the image, subject to the read budget |
| `c`; Tab | Toggle the sidebar; move focus between image and channels |
| `,` / `.`; sidebar wheel | Focus the previous / next channel |
| Space; click a sidebar channel | Toggle the focused channel in the composite |
| `1` through `9` | Solo channels 0 through 8 |
| `l` / `u` / `g` | Enter the focused channel's lower / upper limit / gamma |
| Enter; Escape or Ctrl-C while editing | Apply; cancel numeric entry |
| `a` | Estimate and lock contrast from the focused channel's visible preview |
| `t`; `m`; `i` | Toggle image color; ASCII / half-block mode; reverse ASCII density |
| `?`; `q` or Ctrl-Q | Toggle key help; quit |

When the sidebar has keyboard focus, Up/Down selects its focused channel. At least
one channel must remain selected. The number shortcuts do not change the zero-based
indices shown in metadata or accepted by `--channel`.

### Contrast and color

Windowing is `clip((value - lower) / (upper - lower), 0, 1) ** (1 / gamma)`.
Larger gamma brightens midtones. The source file and decoded source intensities
are never modified. Lower and upper must be finite, upper must exceed lower, and
gamma must be finite and positive. Invalid numeric edits stay in the editor rather
than crashing the viewer or silently accepting a different value.

Unsigned RGB starts with its dtype's full intensity range and a shared window for
all three RGB samples, preserving their relative balance. Other image channels
start with the 1st and 99.5th percentiles of the first finite preview, with explicit
constant/sparse-field handling. These are **preview-based estimates**, not a
whole-slide stain-normalization procedure. Zeros participate. Initial windows
remain fixed as the camera moves. Pressing `a` deliberately re-estimates the focused
channel from the current visible preview; manual `l/u` values remain in raw units. Gamma is dimensionless and is preserved
when `a` re-estimates the lower and upper limits.

Multiplex composition windows each selected channel separately, adds a deterministic
pseudocolor, and clips the composite. Colors are assigned by channel index, not by
inferred marker identity. There is no color picker in this first version; the library
accepts explicit RGB colors through `ChannelStyle`.

ASCII uses ` .:-=+*#%@` by default. Colored ASCII uses character density for brightness
and peak-normalized foreground color for hue. RGB defaults to reversed density so
white background becomes empty space. `i` changes that behavior. Half-block mode
uses actual windowed RGB as the upper/lower half-cell colors; density inversion has
no effect on half-block colors. `--no-color` uses the strongest selected signal for
a monochrome ASCII view.

Truecolor is enabled when `COLORTERM` is `truecolor` or `24bit`. An explicit
`PROMPT_TOOLKIT_COLOR_DEPTH` takes precedence; otherwise prompt_toolkit selects its
terminal fallback, usually 256 colors. `--ansi` snapshots always encode 24-bit SGR.
Some terminals or SSH/tmux configurations need their mouse/color forwarding enabled.
For tmux, enable its mouse handling with `set -g mouse on`; key controls remain
available when wheel or drag events are intercepted elsewhere. tmux itself is not
a runtime dependency, and automated tests do not imply every terminal is validated.

## Text snapshots and Python use

The noninteractive path uses the same planning, sampling, composition, and rendering
operations as the viewer. It never enters the alternate screen.

```bash
ttwhy slide.ome.tif --snapshot --columns 120 --rows 40

ttwhy slide.ome.tif --snapshot -o preview.txt

ttwhy slide.ome.tif --snapshot --mode blocks --ansi -o preview.ansi
```

Plain snapshots contain no escape sequences. `--color` and half-block snapshots
require explicit `--ansi`. Existing output files are replaced by default; use
`--no-overwrite` to refuse replacement before reading pixels. File output is
published only after the entire text frame is ready. The input image cannot be
used as an output, including through a symlink or hard link. No snapshot is written
implicitly when the viewer exits.

### Render an array without opening a file or terminal

```python
import numpy as np
from ttwhy import IntensityWindow, TextRenderer

pixels = np.arange(4096, dtype=np.uint16).reshape(64, 64)
normalized = IntensityWindow(lower=100, upper=3500, gamma=1.2).apply(pixels)
frame = TextRenderer(color=False).render(normalized, columns=64)
print(frame.to_text())
```

Input arrays are YX grayscale or YXS RGB. `TextRenderer` takes normalized values
in `[0, 1]`; raw intensity conversion is an explicit separate operation. Omitted
`rows` preserves square source-pixel shape using the configured terminal-cell
height/width ratio, which defaults to 2.0. Explicit rows selects the output geometry.
Nonfinite samples remain missing, not fabricated zero-intensity image data.

A `TextImage` owns read-only `characters`, optional `foreground`, and optional
`background` arrays. `to_text(ansi=False)` returns plain text, `to_text(ansi=True)`
adds SGR colors, and `iter_runs(row)` exposes framework-independent styled runs for
other applications. No serialization includes cursor movement. `resample_image`
and `compose_channels` are independently usable public functions.

### Read and render a field of view

```python
from omeify import OMETiffReader
from ttwhy import TextRenderer, ViewSession

with (
    OMETiffReader("multiplex.ome.tif") as image,
    ViewSession(image, name="Multiplex", cache_mib=32) as session,
):
    renderer = TextRenderer(mode="ascii", color=True)
    viewport, limited = session.fit(
        columns=120, rows=40, renderer=renderer, channels=[0, 2],
    )
    sample_rows, sample_columns = renderer.sample_shape(columns=120, rows=40)
    plan = session.plan(
        viewport, columns=sample_columns, rows=sample_rows, channels=[0, 2],
    )
    samples = session.sample(plan)
    styles = session.prepare_styles(samples)
    frame = session.render(samples, styles=styles, renderer=renderer, columns=120, rows=40)
    print(frame.to_text(ansi=True))
    print(f"Bounded crop instead of whole image: {limited}")
```

A `Viewport(x, y, width, height)` uses **base-image pixel-edge coordinates**.
`pan()` and `zoom()` return new viewports. A source's `read_region()` instead uses
coordinates in the requested pyramid level. `ViewSession.plan()` uses declared
`ImageLevel.downsample_yx` when known, including odd-size edge footprints. For an
unknown reduced-level scale it retains the former extent-fit display approximation,
flagged by `plan.level.approximate`, the live status, and snapshot stderr. That
fallback is for browsing, not exact registration or a physical-calibration estimate. Physical pixel anisotropy participates in fit geometry;
unknown calibration means square pixels, not an invented microns-per-pixel value.

To change an individual sampled channel's style:

```python
from ttwhy import ChannelStyle, IntensityWindow

# Run inside the source context above; sampled channel order is [0, 2].
styles = [
    ChannelStyle(IntensityWindow(100, 12000), color=(0.35, 0.55, 1.0)),
    ChannelStyle(IntensityWindow(300, 24000, gamma=1.4), color=(1.0, 0.3, 0.5)),
]
frame = session.render(samples, styles=styles, renderer=renderer, columns=120, rows=40)
```

`ViewSession` borrows an already-open Omeify Image and owns its display caches.
Closing the session does not close the image. Closing/reopening the image ends the
session's usable lifetime, including cache hits. Independent `ViewSamples` and
`TextImage` values remain inspectable afterward.
For embedding a frontend, `from ttwhy.viewer import Viewer` provides
`await Viewer(session).run()`; the caller retains the source context while it runs.

## Image-backed Python use (0.2)

The former ttwhy `OmeTiffSource` class is removed. Omeify owns image origins;
`ViewSession(image)` owns bounded display. There is one sampling/rendering path,
not a file path and a separate simulator path. `max_read_pixels=` now belongs to
the session. The CLI retains its options and controls.

```python
# With an already rendered Mocktome mIF product:
from ttwhy import TextRenderer, ViewSession

with mif.as_image() as image, ViewSession(image, name="Mocktome") as session:
    renderer = TextRenderer(color=False)
    view, limited = session.fit(columns=100, rows=35, renderer=renderer)
    rows, columns = renderer.sample_shape(columns=100, rows=35)
    samples = session.sample(session.plan(view, rows=rows, columns=columns))
    frame = session.render(samples, styles=session.prepare_styles(samples),
                           renderer=renderer, columns=100, rows=35)
    print(frame.to_text())
```

Mocktome 0.5 supplies `as_image()` for its existing eager outputs. It is not a
new random exposure per read. Other providers can implement Omeify ImageSource
without a dependency on Mocktome or ttwhy. There is no TIFF serialization in this
Python workflow. Sources with one level still obey the same read budget; the
session does not fabricate a cheap whole-image overview.

A session may be used explicitly with `close()` rather than `with`, but it cannot
be reopened. On a real terminal, `await Viewer(session).run()` borrows the session;
it waits for its worker before returning. Notebook text snapshots use `print()`;
this release does not add a graphical notebook widget.

## Bounded reads and caching

The source boundary is Omeify. ttwhy never reads TIFF pages through a second decoder,
parses competing OME metadata, or runs Omeify's conversion/mutation commands.
Stored pyramid levels are selected according to the display sampling footprint.
Only a clipped, intersecting source region is requested. Those source tiles are
decoded by Omeify. Separable area averaging reduces the bounded rectangle to the
terminal sampling grid, including fractional camera positions and letterboxing.
Raw preview means use float64 so narrow windows on uint32 data survive resampling.

There are two deliberately different caches:

* The provider-owned cache, evicted through Omeify's public `clear_cache()` when
  switching channel/level pairs. ttwhy does not create a second raw tile cache or
  assume one separate decoder cache exists for each Channel.
* ttwhy's byte-bounded LRU of **raw resampled previews**, 32 MiB by default. Contrast,
  gamma, coloring, and density can reuse these samples without reading TIFF data.
  The current rendered frame and its style runs are retained for cheap terminal repaint.

One background worker performs sampling and rendering. New camera requests replace
pending older ones; an in-flight decode completes safely but its stale result is
not displayed. Quitting joins the worker before returning control to the source's
context manager. Decode errors exit with their cause rather than leaving a stale
frame presented as a successful new view.

`--max-read-pixels` defaults to 4,194,304 source samples **per request**, counting
selected channels and all RGB samples. Requests exceeding it are rejected before
pixel reads. A large file with no useful coarse pyramid cannot provide a cheap,
faithful overview. Initial fit then shows an explicitly labeled bounded center crop;
zooming farther out is rejected rather than silently reading a huge image.
No automatic pyramid construction or sparse fake overview is substituted.

This sample-count budget and the preview cache size are **not a process-RSS cap**.
Decoder buffers, masks, area-averaging intermediates, metadata, and terminal output
also use memory. Omeify's per-plane decoded cache targets 64 MiB, but one oversized
source strip or tile can exceed that target. A poorly tiled source can still be slow.

## Scope of this initial version

The CLI supports tiled or striped **OME-TIFF**, including stored SubIFD pyramids,
grayscale, RGB, and planar multi-marker images. Python sessions accept an open
Omeify intensity or RGB Image, including array and procedural backends. One OME
series is viewed per CLI application;
choose another at startup with `--series`. Non-singleton Z/T stacks, label-specific
categorical coloring, in-app series switching, raw vendor formats, OME-Zarr, remote
URLs, annotations, segmentation overlays, saved view settings, custom colors in
the sidebar, and ICC-managed color are not implemented. Explicit categorical LabelImages are rejected rather than assigning continuous
brightness to arbitrary object IDs. OME-TIFF alone does not declare label semantics;
CLI users must not interpret numeric label brightness as expression. The reader fails clearly
on unsupported inputs. Font-dependent density and terminal color approximation
remain limitations of text rendering.

## Demo and tests

```bash
python examples/make_demo.py --output-directory /tmp/ttwhy-demo

ttwhy /tmp/ttwhy-demo/synthetic-rgb.ome.tif
ttwhy /tmp/ttwhy-demo/synthetic-mif.ome.tif --channel 0 --channel 2

python -m pytest --cov=ttwhy --cov-report=term-missing
ruff check .
python -m build
```

The demo creates deterministic tissue-like rings, nuclei, and marker signals. It
contains no patient data and is not a biological simulation. Tests independently
write real TIFFs with tifffile, read them through the real Omeify dependency, compare
resampling with a small rectangle-intersection oracle, and exercise the actual
prompt_toolkit application with keyboard and SGR mouse events. POSIX PTY tests also
check alternate-screen, mouse, and cursor restoration. Some codec/terminal/platform
combinations require separate validation; see [the validation record](docs/validation.md) for
what was actually run for this patch.

## License

Apache-2.0. The original repository license is unchanged.
