# Rendering boundary and extension notes

## Public workflow

```
Omeify Image + Viewport
    -> ViewSession.plan()
    -> ViewSession.sample()
    -> ViewSession.prepare_styles() or explicit ChannelStyle values
    -> ViewSession.render()
    -> TextImage
```

The file-free operations `IntensityWindow.apply`, `resample_image`,
`compose_channels`, and `TextRenderer.render` are separate public stages, not
private functionality only reachable through the viewer. `TextImage.iter_runs`
provides structured presentation for integrations without depending on ANSI or
prompt_toolkit.

## Modules

The CLI calls `image_input.open_image()` and passes its open Omeify Image to
`ViewSession`. OME-TIFF still uses `OMETiffReader` directly, with its original
regional reads and stored levels. Ordinary raster files and one selected PDF page
are decoded once and wrapped through Omeify's public array API. There is no
second TIFF decoder or second renderer. `view_info.py` derives display geometry
from the public image descriptor.
`ViewSession` uses a session-bound borrowed Omeify image view, generic channels,
regional reads, and source cache eviction. It is the same path for a TIFF,
array, or externally implemented procedural image. There is no ttwhy backend
registry, TIFF decoder, or dependency on Tilework or Mocktome. The existing
Image/ImageSource contract is reused rather than replaced with a viewer adapter.

`viewport.py` owns camera geometry. `view_session.py` owns planning, read budgeting,
preview sampling, preview cache, and orchestration of normalization/rendering.
`resampling.py` performs separable area integration with sparse overlap matrices.
Partial edge cells average covered finite pixels only; uncovered cells are NaN.
Known pyramid scale factors come from Omeify ImageLevel.downsample_yx. An
odd-sized base can truncate the final coarse pixel's support; the resampler clips
that support at the true image edge. Unknown scale uses explicitly approximate
extent-fit display mapping. This fallback neither changes image calibration nor
claims knowledge of a vendor resize kernel.

`intensity_window.py`, `compositing.py`, `text_renderer.py`, and `text_image.py`
separate raw intensities, normalized RGB, character selection, and serialization.
Inputs are unchanged and retained outputs own their arrays. ASCII colored hue
normalization is a display operation, not source color normalization.

`viewer/` contains terminal controls, bindings, worker requests, and application
lifecycle. UI painting reuses prepared cell/style runs; it never decodes tiles.
The event loop owns UI state. A single worker owns expensive sampling/rendering,
with a latest-pending-request slot. There is no unbounded queue of old wheel events
waiting to become images. Accepted completed frames carry their generation number;
a stale generation cannot replace the current request's result.

## Ordinary-file ingress

`image_input.py` is deliberately a small, explicit front door, not a plugin
registry. TIFF and PDF signatures select their paths; the remaining path uses
Pillow with a fixed raster-format allowlist. No decoder exception triggers a
fallback, and formats such as EPS cannot invoke converter subprocesses. The
Pillow dependency is lazy-imported, while PDFium is an optional `pdf` extra.
Even `ttwhy.open_image` remains importable without loading those decoders.

One selected raster frame is loaded, orientation-corrected, and composited over
a chosen opaque matte before reduction. Common display modes become RGB8;
16-bit scalar PNG retains uint16 intensity. ICC color conversion is out of scope.
PDF pages render once at the requested opening DPI. RGB byte order is explicit,
and the native bitmap is copied before bitmap/page/document cleanup. All ttwhy
PDFium work, including import and cleanup, is serialized under one lock; it
finishes before the viewer worker starts. Unrelated code using PDFium directly
in the same process must coordinate its own calls with this constraint.

`image_pyramid.py` performs file-free dyadic reduction through 1x1. Level zero is
an independent unchanged copy, and all retained levels are immutable. Float64
sums and finite counts propagate to the next reduction before dtype rounding;
odd edges are neither padded nor equally weighted against larger supports.
Explicit `ImageLevel.downsample_yx` values are powers of two, not ratios of rounded
array dimensions. Omeify borrows these owned arrays through `from_array(copy=False)`.
The generic ViewSession does not acquire pyramid-building behavior.

This ingress intentionally pays the complete decode/build cost in RAM. Its
retained pyramid and transient sums/copies are outside the session preview-cache
and per-read budgets. No temporary TIFF or persistent cache exists. Array-image
references retain their arrays after close, following Omeify's standard ownership
contract; releasing the references releases this storage. CLI process exit also
releases it. OME-TIFF's existing read/cache/lifetime policy is unchanged.

`--series` remains zero-based for OME-TIFF; `--page` is one-based for PDF;
`--frame` is zero-based for ordinary raster frames. Invalid cross-kind selections
fail explicitly. PDF zoom enlarges the selected raster, not live vectors; page
switching and animation playback are deliberately not added in this pass.

## Structured text rendering

`TextRenderer` defaults to `detail`; `shape` and `simple` are the other text modes.
`ascii` is an input alias for `simple`. `blocks` retains the original two-color
half-block path. Simple and blocks retain their previous numerical algorithms.

`renderer.sample_shape()` is the planning boundary: detail and shape ask for
8-by-4 display samples per terminal cell. The CLI and worker already use that
public contract, so they preserve subcell structure through regional reading and
raw resampling, before windowing and composition. This may select a finer pyramid
level; it does not bypass the source-sample budget or create a second decoder.
Library callers must use that contract too, rather than passing cell dimensions
as if all renderers used one sample per cell.

`structured_rendering.py` consumes normalized display samples and emits a normal
`TextImage`. Detail addresses a 4-by-2 Braille dot grid from the 8-by-4 samples.
It uses mild local sharpening, an ordered rounding rule for each cell's total dot
count, and ranks addressed intensities to select dot positions. A small fixed tie
preference balances nearly equal samples. Zero/absent positions cannot receive
ink solely through the tie preference. The bit order follows Unicode's dot
numbering, including the bottom-left/bottom-right bits 6 and 7.

Shape uses a relief-style intensity transform and compares centered, normalized
patches to the hand-authored stroke descriptors in `glyph_atlas.py`, balancing
spatial correspondence against tonal error. Low-variation cells use the tonal
anchors instead of arbitrary letters. These masks are geometric approximations,
not terminal-font rasters or extracted font data. There is no font dependency.

Normalized convolution excludes missing coverage from neighborhood statistics;
missing samples do not become false dark boundaries. Candidate descriptors are
also masked and renormalized on partial cells. Shape matching uses batches of at
most 256 cells; it never creates a whole-slide candidate tensor. Structured output
grids have a separate 4,194,304-position cap checked before resampling. Existing
sample-count and cache budgets remain distinct from a process-RSS limit.

Both structured modes use foreground-only color. Their UTF-8 characters still
carry the structure in plain snapshots; independent background colors remain
exclusive to explicit blocks mode. All modes share `TextImage`, terminal runs,
atomic snapshot publication, and worker lifetime handling. There is no browser,
model inference, semantic description, random rendering, or persistent state.
The ordered pattern is screen-grid anchored, not motion-compensated.

The UI's `m` cycles detail/shape/simple/blocks (omitting blocks without color).
`b` toggles the original half-block preview and restores the previous text/color
settings. Color toggling preserves a text mode. Switching modes never requests an
automatic contrast reset: existing per-channel windows stay locked. Detail and
shape request identical sampling grids and share cached previews for a view.

## Terminal capability policy

The CLI's `--mode auto` is deliberately a *frontend policy*, not another glyph
algorithm. `terminal_capabilities.py` chooses the original blocks renderer for
interactive Unicode-capable terminals whose prompt_toolkit color depth is at
least 256 colors, and simple for limited terminal/encoding capabilities or
explicit no-color. The selector respects the same color-depth precedence as the
viewer (`NO_COLOR`, `PROMPT_TOOLKIT_COLOR_DEPTH`, `COLORTERM`, toolkit default).
No terminal escape sequence is sent to probe capability. Font coverage itself
cannot be verified; users can always force a mode. The viewer status shows the
resolved renderer rather than claiming a font was tested.

Snapshots have a different output contract: auto + no ANSI emits portable simple
text, while auto + explicit ANSI emits truecolor blocks. The array-level
`TextRenderer` retains its context-independent `detail` default. Explicit modes,
`sample_shape()`, and their numerical rendering are unaffected by auto selection.

## Thread and resource contract

Keep the Omeify image and `ViewSession` contexts open for `Viewer.run()`. The viewer
waits for an in-flight read on exit; cancellation does not pretend to interrupt a
native TIFF codec. Session methods serialize reads and session close through an RLock.
A session borrows a stable image: do not mutate its pixels or concurrently close
it. A new image lifetime requires a new ViewSession, whose preview cache belongs
to the original observation. Closing the display does not close the caller's image.
A session rejects access after its source is closed and reopened, including cache hits.

The provider's cache is cleared on channel/level switches. No independent channel
cache is assumed in Omeify 0.21.0; providers own their cache implementation. ttwhy's own preview LRU can retain multiple selections and viewports.
Changing intensity windows does not change preview-cache keys.

There is no on-disk cache or implicitly written user configuration. Errors do not
fall back to another decoder. Snapshot output is independent of terminal state;
file publication is atomic within the destination filesystem, with exclusive
publication for `--no-overwrite`.

## Deliberate future work, not hidden partial implementations

A truly shared Omeify decoded-cache budget could improve multi-channel panning
without one-cache-per-channel memory growth. Further profiling should precede
prefetch or another raw tile cache. Render-time optimizations should be compared
with the exact rectangle-area oracle in tests.

More glyph modes, overlays, and other terminal frontends should share `TextImage`
and the normalized image pipeline, not add independent contrast semantics. Vendor
formats should only be added after their reader-level region/pyramid contracts are
verified. Input conversion support alone is not evidence of interactive read support.
