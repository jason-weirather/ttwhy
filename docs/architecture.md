# Rendering boundary and extension notes

## Public workflow

```
OmeTiffSource.info + Viewport
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

`ome_tiff_source.py` is the only file boundary. It uses Omeify's public reader,
logical channels, level metadata, regional reads, and channel cache clearing.
There is no general image-backend registry and no dependency on Tilework.

`viewport.py` owns camera geometry. `view_session.py` owns planning, read budgeting,
preview sampling, preview cache, and orchestration of normalization/rendering.
`resampling.py` performs separable area integration with sparse overlap matrices.
Partial edge cells average covered finite pixels only; uncovered cells are NaN.
Pyramid scale factors are ratios of stored raster extents for display alignment,
not invented physical calibration or a guarantee about a vendor's resize kernel.

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

## Thread and resource contract

Keep the `OmeTiffSource` context open for the full `Viewer.run()` call. The viewer
waits for an in-flight read on exit; cancellation does not pretend to interrupt a
native TIFF codec. Source methods serialize reads and close through an RLock.
A session borrows a stable source: do not change the file while it is open or close
and reopen the same source underneath an existing session. A new source session
requires a new ViewSession, whose preview cache belongs to that file lifetime.
A session rejects access after its source is closed and reopened, including cache hits.

Only one source channel/level decoder cache remains active. This deliberately
trades cross-channel decoded-tile reuse for bounded cache multiplicity in Omeify
0.16.1. ttwhy's own preview LRU can retain multiple selections and viewports.
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
