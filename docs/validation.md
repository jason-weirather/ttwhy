# ttwhy initial implementation: validation record

## Result

**135 tests passed.** Measured statement coverage was **93.28%** (1,083 of 1,161 statements). This is statement coverage, not branch coverage. The full suite also passed from the built source distribution and after applying the patch to a fresh reconstruction of the uploaded baseline.

The patch targets the supplied initial ttwhy snapshot, commit `4186c3e62f3d45a1588ea2e5320a7950795c37b3`. Its original LICENSE and .gitignore are unchanged. It introduces ttwhy 0.1.0.

## Tested environment

Linux x86-64; Python 3.13.5. Dependencies used:

| Component | Version |
|---|---|
| Omeify | 0.16.1, source commit `529bf0a82269b123960e3554d2f83ab31b5f31f5` |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| tifffile | 2026.5.15 |
| prompt_toolkit | 3.0.52 |
| Click | 8.1.8 |
| wcwidth | 0.7.0 |
| pytest | 9.0.2 |
| pytest-asyncio | 1.3.0 |
| pytest-cov | 7.0.0 |
| pexpect | 4.9.0 |
| setuptools | 82.0.1 |

Omeify was loaded from the actual source snapshot for the checked commit, not replaced by a stub or reimplemented decoder. Reconstructed source files were checked against their snapshot byte counts. Tests write their TIFF fixtures independently with tifffile, then read them through Omeify. No upstream Omeify patch is required.

## Checks actually run

**Numerical and text engine:** explicit clipping and gamma; unsigned, signed, and floating intensities; nonfinite and constant inputs; sparse preview windows; deterministic sampling; uint32 narrow-window precision through the complete file-to-text path; area resampling against an independent rectangle-intersection oracle; fractional viewports; uncovered/partially covered cells; RGB and multichannel composition; character aspect; exact ASCII ramp output; half-block foreground/background colors; ANSI reset behavior; terminal-control and double-width glyph rejection; owned/read-only outputs; nonmutation of source arrays.

**Real file boundary:** tiled and striped OME-TIFF; deflate and uncompressed storage; big- and little-endian files; uint8, uint16, uint32, signed int16, float32, and float64 workflows; RGB uint8/uint16; three-channel fluorescence; stored pyramids with odd dimensions; OME series selection; physical pixel anisotropy; missing calibration; missing/malformed/non-OME inputs; unsupported non-singleton Z; invalid channel/level/region inputs; resource closure and rejected reuse after close/reopen. A test makes full-page/full-series `asarray` calls fail to ensure the viewer uses regional decoding instead.

**Planning and caching:** stored-level selection; source-sample budget before pixel reads; RGB counting as three samples; explicit bounded-crop fallback; decoded-cache release on channel/level changes; preview LRU eviction and disabled caching; repeated rendering with new contrast but no additional TIFF reads; forged-plan rejection; decoder failures propagated without cached partial success.

**Actual full-screen application:** keyboard zoom/pan; sidebar visibility and narrow-terminal resize; RGB and fluorescence frames; channel solo/multiselect; fixed windows across camera movement; lower/upper/gamma editing; invalid input retention; cancellation; letter input not triggering global shortcuts inside numeric entry; auto limits preserving gamma; ASCII/half-block/color controls; SGR mouse wheel with coordinate anchoring; mouse drag; sidebar clicks; coalescing rapid requests; stale-frame rejection; worker error propagation; exit waiting for an in-flight decode before source closure.

**Real POSIX pseudo-terminal:** spawned the CLI in an actual PTY, exercised input and resize, observed colored half-block output, and checked alternate-screen exit, mouse-protocol disable, cursor restoration, normal quit, and Ctrl-C. Detailed state assertions also use injected terminal input against the same real prompt_toolkit application, not a mocked GUI.

**CLI and distribution:** eager version/help; plain and ANSI snapshots; repeated channel options; explicit raw limits; output preflight; atomic publication and no-overwrite; input-path/hard-link protection; non-TTY guidance; Python compilation; wheel and source-distribution builds through the PEP 517 setuptools backend; wheel installation with `--no-deps --no-index`; installed version from distribution metadata; file-free rendering without importing Omeify or prompt_toolkit; installed CLI snapshot; wheel license and `py.typed`; source distribution including test fixtures, docs, environment file, and demo; full test suite from the extracted source distribution.

**Patch:** `git diff --cached --check`, `git apply --check`, application to the exact uploaded three-file baseline, byte-for-byte comparison of the applied tree, and the full test suite from that applied tree.

## Visual check and small timing experiment

The included `examples/make_demo.py` generated two deterministic 768 by 768 tissue-like fixtures, RGB and three-channel fluorescence, with five stored pyramid levels. These are synthetic ring/nucleus/marker patterns, not patient data and not a biological simulation. Their rendered ASCII and half-block cell grids were visually inspected. The accompanying preview image depicts actual TextImage cells, not a screenshot of the terminal UI.

For a 100-column by 36-row frame at stored level 3, the following were measured in the test container:

| Fixture and mode | Source samples requested | Initial sample + render | Cached sample + render, median of 12 |
|---|---:|---:|---:|
| RGB ASCII | 27,648 | 9.42 ms | 0.53 ms |
| RGB half-blocks | 27,648 | 2.49 ms | 0.48 ms |
| Fluorescence ASCII, two channels | 18,432 | 4.16 ms | 0.69 ms |
| Fluorescence half-blocks, two channels | 18,432 | 4.13 ms | 0.73 ms |

These numbers are only a small synthetic rendering check. They do not include terminal transport/repainting, do not establish frames per second, do not control the operating-system file cache, and do not predict performance on a large slide over institutional network storage. The cached path is expected to be cheap because it reuses already sampled preview intensities.

## Limits of this validation

No real patient slide, multi-gigabyte production whole-slide image, institutional filesystem, macOS terminal, Windows terminal, or actual tmux session was tested here. No claim is made that all terminal emulator/font/SSH/tmux combinations work identically. Tests cover the direct application and POSIX terminal protocol; actual deployment remains a useful next check.

The available environment did not contain imagecodecs, so JPEG, JPEG 2000, and LZW decoder combinations were not exercised. The viewer delegates codec support and errors to Omeify. Deflate and uncompressed TIFF reads were exercised using real codecs.

Ruff was not installed and was not run. The `build` command-line frontend was also unavailable; both distribution artifacts were successfully built by calling the project's actual PEP 517 backend, `setuptools.build_meta`. The wheel installation check used existing scientific dependencies rather than a fresh network dependency resolution. The declared minimum dependency versions and other Python versions were not matrix-tested.

The two PTY tests emitted Python 3.13's `forkpty()` deprecation warning about spawning from a multi-threaded test process. Both tests completed successfully. These warnings came from the test harness, not exceptions in the viewer.

This remains an alpha text display, not a validated diagnostic viewer. Source-sample budgets and preview-cache limits are not a hard process-RSS bound; decoder buffers, oversized strips, resampling intermediates, and terminal presentation also use memory. RGB rendering is not ICC-managed, ASCII density is font-dependent, and the live viewer handles one OME series at a time.
