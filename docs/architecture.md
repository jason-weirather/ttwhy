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

The CLI opens an Omeify `OMETiffReader` and passes it to `ViewSession`.
`view_info.py` derives display geometry from the public image descriptor.
`ViewSession` uses a session-bound borrowed Omeify image view, generic channels,
regional reads, and source cache eviction. It is the same path for a TIFF,
array, or externally implemented procedural image. There is no ttwhy backend
registry, TIFF decoder, or dependency on Tilework or Mocktome.

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
