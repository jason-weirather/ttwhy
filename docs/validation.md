# ttwhy 0.2: Omeify 0.18 migration validation

This update targets the supplied `2269e1c1db738e83e36ab9a120980dd61ed3ef9e`
snapshot. It replaces the old file-only adapter with a ViewSession borrowing an
ordinary Omeify Image. No second TIFF decoder or pixel-normalization path is added.

## Executed checks

**108 tests passed** against the actual Omeify 0.18 source reconstructed from the
provided snapshot and consolidation patch. These include independent TIFF
fixtures, numerical rendering, Click snapshots, the actual prompt_toolkit viewer,
and two POSIX pseudo-terminal tests. Two Python 3.13 forkpty deprecation warnings
came from the terminal harness; neither was a test failure.

The replaced file-adapter tests were consolidated around current behavior:
file/array equivalence, regional reads without whole-level materialization,
RGB logical-channel semantics, uint32 display precision, explicit dtype/byte
order handling, anisotropy, fractional viewports, empty coverage, current read
budgets, cache eviction, source failure, and source close/reopen invalidation.
A huge procedural image is described without allocating its complete raster.
Its requested channels and absolute-coordinate values are checked independently.

Declared pyramid scales and odd-sized boundary coverage have dedicated numerical
checks. Unknown scales retain a visibly labelled extent-fit display approximation;
this is not inferred physical calibration. Categorical LabelImages are explicitly
rejected because object-ID brightness would not be categorical visualization.

Against the untouched ttwhy/Omeify 0.16.1 baseline, CLI help and four real snapshot
workflows matched byte-for-byte in stdout and stderr: grayscale ASCII,
multichannel ANSI, RGB half-blocks, and a read-budget-limited crop. Those fixtures
have declared scales and dimensions divisible through their pyramids. This does
not claim that corrected odd-sized mapping or newly labelled unknown scales
produce unchanged text.

Actual Mocktome mIF and H&E image handles were sampled and rendered without an
intermediate TIFF. The exact upper/lower half-block color arrays were inspected
as enlarged PNGs; these are representations of TextImage cells, not screenshots
of a terminal emulator. The same open mIF image passed through Tilework 0.9 and
Cadastre 0.15, with no input TIFF staged. Cadastre used a small CPU test segmenter,
not Mesmer or StarDist.

## Packaging and environment

Python 3.13.5 on Linux; NumPy 2.3.5, SciPy 1.17.0, tifffile 2026.5.15,
prompt_toolkit 3.0.52, and pytest 9.0.2. The complete real dependency source was
used; the Image contract and TIFF reader were not replaced by mocks.

Wheel and source-distribution construction uses the installed setuptools build
backend. No font, external image, model weight, or simulator source is bundled.
The pure-array renderer remains importable without importing Omeify or the
terminal frontend. The package retains its Python 3.13 minimum.

## Limits

Ruff and the `build` command-line frontend were unavailable. The local OME XSD
and imagecodecs were also unavailable. Viewer tests write TIFF fixtures
independently and do not require an OME writer-validation bypass. Deflate and
uncompressed reads were exercised; JPEG/JPEG 2000/LZW codecs were not.

No real patient slide, multi-gigabyte production slide, cluster deployment,
macOS/Windows terminal, or actual SSH/tmux combination was tested. Source-sample
and preview-cache budgets are not a process-RSS limit. A generic provider can
perform expensive work internally, and decoded caching remains provider-owned.
This is not a color-managed or diagnostic microscopy display.
