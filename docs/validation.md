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

## Omeify 0.21.0 / Python 3.11 baseline (October 2026)

The preceding validation record describes the historical Omeify 0.18 migration and
its then-current Python 3.13 minimum. The packaging baseline is now
`omeify>=0.21.0` and `requires-python = ">=3.11"`; the default Conda environment
uses Python 3.11. No image sampling, rendering, or terminal interaction logic was
changed for this transition. In particular, ttwhy still consumes Omeify's public
`Image` and `OMETiffReader` APIs, without a second TIFF decoder.

The complete ttwhy test suite was run against the **Omeify 0.21.0 source snapshot**
(commit `6d9f5ff`) under Python 3.13.5: **108 passed, 2 warnings**. All ttwhy
source, demo, and test modules also parse with Python 3.11 grammar. This is a
source-level integration test, not verification of the separately published PyPI
wheel or an executed Python 3.11 runtime test. A Python 3.11 interpreter was not
available in the test environment.

To validate the newly supported interpreter using the published dependency:

```bash
mamba env create -f environment.yml
mamba activate ttwhy_env
python -m pip install -e '.[dev]'
python -m pip check
python -c 'import sys, omeify; print(sys.version); print(omeify.__version__)'
python -m pytest -q
```

Also smoke-test a real RGB and multiplex OME-TIFF, especially when the downstream
TIFF codec and terminal combination differs from the synthetic fixtures. Python
3.11 compatibility should not be called fully verified until this lane passes.

## Foreground-only terminal render modes (October 2026)

This patch targets the supplied ttwhy `aed55e5` snapshot. It adds default `detail`,
artistic `shape`, and legacy `simple` modes; `ascii` remains an alias for simple
and the original `blocks` renderer remains available. No dependency or Python
minimum was added/raised; the package version is unchanged at 0.2.0.

### Executed checks for this patch

**163 tests passed**, with **94% reported statement coverage**, on Linux with
Python 3.13.5 and the actual reconstructed Omeify 0.21.0 source (`6d9f5ff`). NumPy
2.3.5, SciPy 1.17.0, tifffile 2026.5.15, prompt_toolkit 3.0.52, and pytest 9.0.2
were used. Four `forkpty` deprecation warnings came from the POSIX terminal tests.
They were not failures. This is not a separately installed PyPI-wheel test.

The checks include all 256 Braille patterns against independent literal dot
positions; within-cell geometry that has the same mean but different orientation;
foreground-only and plain-text output; NaN/partial coverage; deterministic batch
matching; richer source sampling; retained read/cache budgets; source lifetimes;
mode cycling; locked windows; monochrome/blocks round-trips; custom legacy ramps;
and actual keyboard, mouse, resize, and terminal-restoration behavior. All three
text modes were opened and interrupted through real POSIX pseudo-terminals.

Eight byte-level goldens were captured from the untouched aed55e5 renderer across
grayscale/RGB, color/no-color, and inversion settings. Simple matches them exactly.
Four complete old/new CLI workflows also had identical stdout and stderr when
the new invocation selected the equivalent legacy mode: grayscale snapshot,
multichannel ANSI snapshot, RGB half-block snapshot, and read-budget-limited crop.

Wheel and source-distribution builds succeeded through the installed setuptools
backend. The wheel includes both new renderer modules and no font assets.
Pure-array rendering was checked without importing Omeify or prompt_toolkit.
All Python source, examples, and tests parse with Python 3.11 grammar.

### Visual inspection and performance scope

Actual `TextImage` results were inspected for the supplied reaction-image crop
(with its letterboxing removed) and deterministic synthetic brightfield, single
marker, and multichannel fields from `examples/make_demo.py`. The comparisons
used equal 80-column text budgets. Inspection images were rasterizations of actual
characters using local fonts, not generated UI mockups or patient-slide results.
Those local fonts are not bundled with ttwhy. Raw text and ANSI were also retained.

On a 512-by-512 synthetic RGB array rendered to 120 columns by 40 rows, median
render-only times over five warm runs were 5.4 ms for simple, 37.1 ms for shape,
and 35.6 ms for detail. These measurements include array resampling but exclude
TIFF decoding, terminal repaint, and SSH transport. They are not an interactive
frame-rate guarantee or a benchmark against other character-art packages.

### Remaining limits

A Python 3.11 interpreter could not be installed in this sandbox because network
DNS resolution failed. Actual Python 3.11 runtime validation therefore remains
outstanding; grammar checks are not a substitute. JPEG/LZW/production-slide and
real macOS/Windows/SSH/tmux validation were not performed in this pass.

Detail's dot-count fit is not a claim of perceptual optimality. Shape's geometric
stencils are not a measured terminal font. Artistic sharpening/relief can alter
appearance and must not be interpreted as segmentation or quantitative evidence.
The ordered pattern is deterministic but screen-grid anchored, so panning can
still change its texture. Both structured modes may need finer pyramid reads and
therefore a tighter initial crop under the same source-read budget.

## Auto terminal renderer selection (October 2026)

Against the supplied ttwhy `1679265` snapshot, this change makes **auto** the CLI
mode default without editing the original rendering algorithms. Interactive
truecolor/256-color Unicode terminals select blocks. Low-depth, dumb, or
unsupported-encoding terminals select simple. Explicit modes remain available;
`NO_COLOR` and prompt_toolkit's color-depth override are honored. Plain auto
snapshots select simple, while `--ansi` auto snapshots select blocks.

The selection is based on advertised terminal capabilities, not a font probe.
Quarter/block-character experiments remain future work. The Python library's
`TextRenderer()` default is still detail, unaffected by environment variables.

## Ordinary raster and optional PDF ingress (October 2026)

This patch targets the supplied ttwhy `aec704f348cfe2b8a418f627a14e3e1ddf11a819`
snapshot. Python >=3.11, Omeify >=0.21.0, package version 0.2.0, and all rendering
algorithms remain unchanged. Pillow is a new base dependency; the optional `pdf`
extra adds pypdfium2. Ordinary files enter through Omeify's public array-image API,
not an alternative ViewSession or TIFF decoder.

### Executed checks

**281 tests passed**: 271 non-PTY tests and 10 POSIX pseudo-terminal tests, with
**94% reported statement coverage** across the combined runs. This includes all
188 baseline tests. The 10 forkpty deprecation warnings came from the terminal
harness under Python 3.13; they were not failures. Coverage does not instrument
the separately spawned terminal subprocesses, which have their own output and
restoration assertions.

The test environment was Linux, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
Pillow 12.3.0, pypdfium2 5.8.0, and tifffile 2026.5.15. Integration used the actual
Omeify 0.21.0 source (`6d9f5ff`) reconstructed from its repository snapshot. It
was not replaced by mocked Image/TIFF implementations or a local API imitation.
This does not verify the separately published Omeify wheel.

New checks cover actual PNG, JPEG, WebP, BMP, GIF, ICO, PPM and TGA codecs;
GIF/APNG/WebP frame selection; palette, alpha and grayscale transparency;
bilevel, grayscale and CMYK display; EXIF orientation; scalar 16-bit PNG precision;
RGB logical-channel/window semantics; filename-independent dispatch; missing
PDF extras; invalid cross-kind options; input/hard-link overwrite protection;
Pillow's large-image guard; and rejection of unsupported inputs without external
converter execution. A roughly 3.8-megapixel RGB PNG fits using its RAM pyramid
under a 60,000-sample per-view budget, then returns exact base pixels on zoom.

Pyramid levels are compared independently with area means taken directly from
base-array rectangles across integer/float, odd, skinny and RGB shapes. Tests
explicitly check finite/missing samples, partial-edge weighting, unchanged base
pixels, immutable ownership, explicit dyadic scales, and nonaccumulating integer
rounding. TIFF inputs are checked not to call Pillow or construct RAM pyramids.

PDF fixtures are independently assembled vector documents with the standard
library, not written by the decoder under test. Tests exercise page selection,
rotation, opaque mattes, RGB rather than BGR, real rasterization at 72/144/300 DPI,
owned pixel lifetime after native handles close, malformed PDFs, invalid pages,
excessive native dimensions, and snapshots. Both PNG and PDF were driven through
the actual prompt_toolkit application and through real CLI pseudo-terminals,
including zoom/pan, contrast, automatic half-block selection, and restored
alternate-screen, mouse, and cursor state on exit.

Visual checks used a local sample photograph encoded as PNG/JPEG/WebP and placed
in a generated PDF page. The actual TextImage half-block foreground/background
colors were rasterized and inspected at a shared 100-column budget. These were
cell-color reconstructions, not screenshots of a physical terminal emulator.
The photograph, generated PDF, inspection images and local fonts are not included
in the patch or package.

### Scope and remaining checks

The full source and tests parse with Python 3.11 grammar. Wheel and sdist builds
were checked with the installed setuptools backend. The array renderer was
exercised under an import guard that prohibits file-decoder and terminal imports.
No converter executable, GUI, display server, GPU, or disk image cache is needed.

A Python 3.11 runtime, minimum-version Pillow/PDFium combinations, macOS/Windows
terminals, SSH/tmux combinations, large production slides, and every PDF feature
were not exercised. Ruff was unavailable in this environment. Pillow builds
without WebP support skip the WebP codec tests; this test environment supported
WebP and executed them. Optional PDF integration tests skip without `ttwhy[pdf]`;
they were installed and executed here. To run both lanes in a checkout:

```bash
python -m pip install -e '.[dev,pdf]'
python -m pytest --cov=ttwhy --cov-report=term-missing
ruff check .
python -m build
```

Pillow remains the ordinary image decoder, with its supported sample depths and
large-image protection. Higher-bit-depth RGB previews can be reduced to RGB8;
scalar 16-bit PNG keeps raw uint16. PDF pages are fixed-resolution raster previews,
not a text/search interface or unbounded vector zoom. Native PDFium calls within
ttwhy are serialized; direct PDFium use elsewhere in an embedding application
must coordinate separately. RAM storage follows Omeify's array ownership policy
and is not governed by the per-view source-sample budget.

## PDF page / OME series navigation and progressive detail (October 2026)

This patch targets the supplied `c43a5e18b66bd1e608c883f24308471a6fbc0cb0` snapshot.
No dependency, Python minimum, package version, or text-rendering algorithm changes.
The original 144-DPI/fixed-resolution validation above is historical; interactive
PDF viewing now refines up to `--pdf-max-dpi` (600 by default).

### Executed checks

**314 tests passed across the final split runs:** 302 non-PTY tests and 12 POSIX
pseudo-terminal tests. This includes the existing 281 tests, with CLI PDF-DPI
expectations updated for the intentional flag replacement. The non-PTY run reports
**92% statement coverage**. Coverage does not instrument spawned PDF or terminal
processes; their actual behavior has explicit integration assertions. The PTY tests
report Python 3.13 `forkpty` deprecation warnings, not application failures.

Environment: Linux, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0,
pypdfium2 5.8.0, tifffile 2026.5.15, prompt_toolkit 3.0.52, pytest 9.0.2.
Tests use the actual Omeify 0.21.0 source (`6d9f5ff`), not an imitation of its API.
All reconstructed baseline ttwhy file byte sizes were checked against the supplied
snapshot. The patch is checked against a fresh baseline reconstruction. Wheel and
source-distribution builds succeed with the installed setuptools backend.

New checks exercise direct page/series entry, invalid destinations, cancellation,
PageUp/PageDown, bracket navigation, clickable navigation buttons, RGB/multichannel
series changes, and OME navigation with whole-image materialization prohibited.
Actual terminal subprocesses cover both PDF and OME destination entry, resize,
exit status, alternate-screen restoration, mouse state and cursor restoration.
The normal PDF terminal test quits with real refinement work outstanding.

Actual spawned PDF jobs test native rendering, RGB byte order, owned/read-only
pyramids, worker death, process cancellation, receiver cleanup and no image-cache
files. A real file-viewer test reaches 600 DPI, changes pages, rapidly retargets,
then quits and verifies that child processes are reaped. Controlled-completion
tests additionally exercise 600-before-300 results, current-page priority before
neighbor work, cached-page reuse, moving-cache eviction, released distant images,
coalesced source loads, and source handoffs while a display read is still running.

Fractional PDF page extents test page-relative camera preservation rather than an
assumed exact DPI ratio. Contrast survives upgrades. A custom read-budget test
verifies that automatic refinement cannot move the camera: a rejected upgrade
stays cached, then succeeds after a closer zoom. Snapshot coordinates are checked
against the same initial-preview grid. High-detail failures preserve a usable
lower-resolution source and display an error.

The terminal tests exposed a cancellation-lifetime race during development. The
final scheduler uses Python 3.11 TaskGroup to await every sibling finalizer. A
regression test deliberately delays one finalizer while the other finishes and
verifies that shutdown does not close the receivers early.

### Full-page and visual spot checks

One synthetic letter-size vector PDF was opened at 144 DPI, then rasterized at
300 and 600 DPI in parallel with the actual worker implementation. In this one
run, quick-preview rasterization/pyramid preparation took about 0.76 seconds;
refinement/transfer took about 3.24 and 3.84 seconds from submission, respectively.
These are environment- and workload-specific spot checks, not benchmark promises.

The resulting 600-DPI raster was 5100 by 6601 RGB pixels (PDFium rounds native
raster extents); its retained pyramid was about **128.4 MiB**. The largest child's
reported peak RSS was about **1097 MiB**, illustrating that construction and IPC
scratch are much larger than retained raster storage. This does not measure or
cap total process-tree RSS, and does not imply every PDF page is safe at 600 DPI.

Actual half-block output for the same close-up of six-point lettering and fine
vector lines at 144/600 DPI was rasterized and visually inspected. These were
reconstructions of TextImage foreground/background colors, not photographs or
screenshots of a physical terminal. Fixtures, inspection images and fonts are not
bundled in the patch or package.

### Remaining validation limits

Python 3.11 grammar checks pass; an actual Python 3.11 runtime was not available.
Native macOS/Windows terminals, SSH/tmux transport, minimum supported PDFium/Pillow
versions, large production OME series and complex real-world PDFs were not tested.
The standalone Ruff executable was unavailable. Snapshot and array-renderer
regressions, source lifetime tests and actual synthetic PDF/OME integrations were
run. No claim of diagnostic color management or unrestricted vector zoom is made.

## Rectangle selection, PDF text and clip delivery (October 2026)

This patch targets the supplied `1faa6c04ce11433736dff2e4678a0ca694338530`
snapshot. All 48 reconstructed baseline files matched their declared byte sizes.
The PDF default remains 600 DPI. Dependencies, Python minimum, package version,
image sampling/pyramid algorithms and normal rendering algorithms are unchanged.

### Executed checks

**394 tests passed across split runs: 380 non-PTY tests and 14 POSIX terminal
tests.** The final non-PTY run reports **92% statement coverage**. Tests include
all original 314 tests plus 80 new tests/parameterized cases. Terminal tests were
split into shorter invocations for the execution environment's command deadline;
coverage does not instrument the spawned terminal/PDF processes. Python 3.13's
forkpty warnings come from the multithreaded test harness, not application errors.

New checks exercise exact Unicode/space-preserving TextImage crops; immutable
foreground/background ownership; drag direction and inclusive endpoints; keyboard
anchors, Shift+arrows, corner swapping and clamping; border exclusion; actual
mouse/keyboard interaction; clickable buttons; Ctrl-C copy rather than quit in
selection mode; stable Save prompts; terminal resize invalidation; PDF refinement
handoffs deferred until selection ends; copied content pinned while an async save
runs; one outstanding clip operation; and immediate-quit save completion.

Independent two-page standard-font PDF fixtures cover 0/90/180/270-degree page
rotation, nonzero origins, fractional page dimensions, CropBox offsets and the
144/300/600 DPI coordinate grids. Rectangle extraction uses real PDFium, compared
with independently calculated coordinate transforms and literal expected words.
Tests verify page changes, missing text layers, explicit screen-glyph copying,
changed-on-disk rejection, and no extra raster/cache files during extraction.

File-delivery tests cover UTF-8/LF output, preserved rectangular spaces, atomic
replacement, no-overwrite, input and hard-link aliases, output symlinks, symlinked
directory components, nonregular files, missing directories, and rechecking a
destination that becomes an input symlink during publication. Clipboard tests
check exact base64 payloads, write-only OSC 52, byte limits without truncation,
helper selection and encoding, no shell evaluation, timeout fallback, remote/
multiplexer routing, and successful file saves despite clipboard failure.
The native-helper unit tests simulate process invocation; they do not change a
real macOS, Windows, X11 or Wayland clipboard.

Actual CLI PTY tests open PNG and PDF files, select and copy rectangles, decode
the emitted OSC 52 to compare the exact clip, verify explicit text and ANSI files,
and check clean exit/alternate-screen/mouse/cursor restoration. These prove emitted
protocol bytes and files, not acceptance by a particular desktop terminal.

### Existing shutdown race reproduced and fixed

The baseline non-PTY run reproduced a failure in the existing real PDF navigation/
rapid-quit test: a PDF process could remain tracked after its normal successful
cleanup was cancelled. TaskGroup alone did not prevent interruption inside an
awaited finalizer. This patch shields the entire process/pipe finalizer and waits
through repeated cancellation. A new actual-worker regression test cancels twice
while successful cleanup is deliberately paused, then verifies the child is reaped.
The existing real PDF navigation/rapid-quit test passes with this fix.

### Visual and packaging checks

A generated PDF page and the actual prompt_toolkit rendered screen were inspected
as PNGs. The screen inspection reconstructed the application's final character/
style grid using a local monospace font and exact half-block colors. The outline,
active corner, selection help and buttons were checked. Selecting the first text
region exported `ALPHA one` while excluding the other line and right-hand text.
This is a reconstruction of actual UI output, not a native terminal screenshot.
Fixtures, inspection PNGs and fonts are not bundled in the patch or package.

All 44 Python source/test/example modules pass Python 3.11 grammar parsing.
Wheel and source-distribution construction use the installed setuptools backend.
The final patch is also applied and tested against a fresh baseline reconstruction.

### Environment and limits

Linux/Python 3.13.5; NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0, pypdfium2 5.8.0,
tifffile 2026.5.15, prompt_toolkit 3.0.52, pytest 9.0.2, setuptools 82.0.1.
Integration uses the actual Omeify 0.21.0 source from the available `6d9f5ff`
snapshot, not an imitation of its Image API. All reconstructed Omeify Python
source files used here matched their declared byte sizes; this does not verify
its separately published wheel.

An actual Python 3.11 runtime, minimum supported PDFium version, native desktop
clipboard, live SSH/tmux/Screen transport, and real macOS/Windows terminals were
not exercised. Ruff and the `build` CLI frontend were unavailable. No OCR or
layout-analysis capability is claimed. Invisible/badly encoded/OCR text layers
can differ from visible lettering; extraction preserves PDFium's text-layer order,
not a guaranteed visual reading order. OSC 52 remains terminal-policy-dependent
and unacknowledged; file saving is the portable explicit fallback.

## Shareable selected PNG for chat apps (October 2026)

This incremental patch applies after the rectangle-selection patch on `1faa6c0`.
It retains the viewer's PDF detail default of 600 DPI and adds no dependencies or
Python-version increase. PNG bytes are rendered from the frozen `TextImage`, not
from a new PDF/image raster read. In half-block mode every upper/lower half uses
the exact RGB foreground/background array values, without terminal fonts or
embedded ANSI escape codes. Braille dots are reconstructed geometrically;
other glyphs depend on available local fonts and can differ from terminal fonts.

**Executed validation:** 402 tests passed in five separate runs: 388 non-PTY
and 14 actual PTY tests. A combined one-process suite run exceeded the test
runner's timeout without an assertion failure; the five smaller runs cover
all 402 collected tests. Linux/Python 3.13.5, real Omeify 0.21.0 source,
Pillow 12.3.0, pypdfium2 5.8.0. The new tests check per-half pixel colors
in decoded PNGs, fixed cell geometry, Braille/ASCII visibility, export size
limits, atomic PNG save, input/hardlink protection, path collisions, local
clipboard-helper command payloads, explicit remote/OSC 52 failures, and keyboard
selection followed by image copy/save. Existing PDF navigation, all 14 terminal
PTY tests, and the original text/ANSI selection actions pass unchanged.

Local helper invocation on macOS, Wayland, and Windows was simulated; actual
macOS/Windows/Wayland/X11 image-paste acceptance by Discord has **not** been
measured. Native helpers are optional; unavailable/remote environments can
save PNGs explicitly. On macOS, the helper briefly uses a private temporary
PNG file for AppleScript clipboard ingestion and deletes it immediately. This
is not an intermediate image pyramid or persistent cache. On SSH/tmux, OSC 52
is text-only and no attempt is made to transmit PNG clipboard data through it.
