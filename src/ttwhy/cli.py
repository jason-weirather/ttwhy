"""Click front door for the same sampling and rendering library used by the TUI."""

import asyncio
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import click

from . import __version__
from .image_input import image_kind, open_image, pdf_raster_shape
from .pdf_workers import DEFAULT_PDF_MAX_DPI, pdf_dpi_steps
from .intensity_window import IntensityWindow
from .terminal_capabilities import available_color_depth, resolve_auto_mode
from .text_renderer import RENDER_MODES, TextRenderer
from .text_clip import CLIPBOARD_MODES, CLIP_SOURCES
from .view_session import ViewSession
from .runtime_overview import DEFAULT_OVERVIEW_CACHE_MIB, DEFAULT_OVERVIEW_MAX_SOURCE_PIXELS
from .startup_overview import (DEFAULT_STARTUP_OVERVIEW_SECONDS, prepare_fast_overview,
                               startup_fit)
from .viewport import Viewport


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Commit a complete UTF-8 snapshot without leaving partial output on failure."""
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Exclusive publication also protects against a destination created
            # after the preflight check. Both paths are on the same filesystem.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@click.command(context_settings={"show_default": True})
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.version_option(__version__)
@click.option("--series", type=click.IntRange(min=0), default=0,
              help="Zero-based TIFF image series (OME or SVS); one series at a time.")
@click.option("--page", type=click.IntRange(min=1),
              help="One-based PDF page to display (default: 1).")
@click.option("--frame", "frame_index", type=click.IntRange(min=0),
              help="Zero-based raster frame, for GIF/APNG/WebP (default: 0; no playback).")
@click.option("--pdf-max-dpi", type=click.FloatRange(min=0, min_open=True),
              help="PDF detail ceiling (default: 600 DPI). Viewer opens at up to 144 DPI, "
                   "then refines through 300/600 and the ceiling. Snapshots use the ceiling.")
@click.option("--pdf-cache-pages", type=click.IntRange(min=1),
              help="PDF page pyramids retained in RAM (default: 3; 1 disables prefetch).")
@click.option("--background", type=click.Choice(["white", "black"]),
              help="Matte for raster transparency and PDF pages (default: white).")
@click.option("--channel", "channels", type=click.IntRange(min=0), multiple=True, default=(0,),
              help="Zero-based logical channel. Repeat to composite multiple markers.")
@click.option("--lower", type=float,
              help="Lower display limit in raw intensity units; pair with --upper.")
@click.option("--upper", type=float,
              help="Upper display limit in raw intensity units; pair with --lower.")
@click.option("--mode", type=click.Choice(["auto", *RENDER_MODES, "ascii"]), default="auto",
              help="auto: colored half-blocks when supported, simple otherwise; "
                   "detail/shape: text art; simple/ascii: legacy density; blocks: half-blocks.")
@click.option("--color/--no-color", default=None,
              help="Use RGB/pseudocolor (auto: on in the viewer, off for plain snapshots).")
@click.option("--invert/--no-invert", default=None,
              help="Reverse text ink density (auto: on for RGB, off for fluorescence).")
@click.option("--cell-aspect", type=click.FloatRange(min=0, min_open=True), default=2.0,
              help="Terminal cell height divided by width.")
@click.option("--region", type=float, nargs=4, metavar="X Y WIDTH HEIGHT",
              help="Initial viewport in base-image pixels instead of fitting. PDF coordinates use "
                   "the initial preview grid (up to 144 DPI), also for snapshots.")
@click.option("--cache-mib", type=click.FloatRange(min=0), default=32.0,
              help="Maximum retained resampled-preview cache size; decoder storage is separate.")
@click.option("--max-read-pixels", type=click.IntRange(min=1), default=4194304,
              help="Maximum source samples per request, counting selected channels/RGB samples.")
@click.option("--overview-cache-mib", type=click.FloatRange(min=0),
              default=DEFAULT_OVERVIEW_CACHE_MIB,
              help="RAM for completed/in-progress full-coverage display overviews; 0 disables. "
                   "Separate from preview cache; viewer and snapshots.")
@click.option("--overview-max-source-pixels", type=click.IntRange(min=0),
              default=DEFAULT_OVERVIEW_MAX_SOURCE_PIXELS,
              help="Total native samples allowed in an overview scan, including channels/RGB. "
                   "Separate from each chunk's --max-read-pixels limit; 0 disables.")
@click.option("--startup-overview-seconds", type=click.FloatRange(min=0),
              default=DEFAULT_STARTUP_OVERVIEW_SECONDS,
              help="Attempt a quick whole-slide cache before first view/snapshot (default: 2s); "
                   "0 skips startup scanning. Native reads can exceed a deadline.")
@click.option("--snapshot", is_flag=True, help="Render one text frame without starting the viewer.")
@click.option("--columns", type=click.IntRange(min=1), default=120,
              help="Snapshot width in characters.")
@click.option("--rows", type=click.IntRange(min=1), default=40,
              help="Snapshot height in characters.")
@click.option("--ansi", is_flag=True, help="Allow ANSI color escape sequences in a snapshot.")
@click.option("--output-path", "-o", type=click.Path(dir_okay=False, path_type=Path),
              help="Snapshot destination; omitted means stdout. Requires --snapshot.")
@click.option("--output-text-clip", type=click.Path(dir_okay=False, path_type=Path),
              help="Interactive Copy/Save destination for the latest text clip; no file until used.")
@click.option("--output-ansi-clip", type=click.Path(dir_okay=False, path_type=Path),
              help="Interactive ANSI artwork clip destination for terminals.")
@click.option("--output-png-clip", type=click.Path(dir_okay=False, path_type=Path),
              help="Interactive image clip destination; PNG preserves colored blocks for sharing.")
@click.option("--output-geojson-clip", type=click.Path(dir_okay=False, path_type=Path),
              help="Interactive GeoJSON destination for rectangle or polygon selections.")
@click.option("--clipboard", type=click.Choice(CLIPBOARD_MODES), default=None,
              help="Copy transport: auto (local helper or OSC 52), osc52, or off (save only).")
@click.option("--clip-source", type=click.Choice(CLIP_SOURCES), default=None,
              help="Text clips: auto (embedded PDF words, otherwise screen glyphs) or screen.")
@click.option("--no-overwrite", is_flag=True,
              help="Refuse to replace an existing snapshot or clip destination.")
def main(
    input_path: Path, series: int, page: int | None, frame_index: int | None,
    pdf_max_dpi: float | None, pdf_cache_pages: int | None,
    background: str | None, channels: tuple[int, ...], lower: float | None,
    upper: float | None, mode: str, color: bool | None, invert: bool | None,
    cell_aspect: float, region: tuple[float, ...] | None, cache_mib: float,
    max_read_pixels: int, overview_cache_mib: float, overview_max_source_pixels: int,
    startup_overview_seconds: float, snapshot: bool, columns: int, rows: int, ansi: bool,
    output_path: Path | None, no_overwrite: bool,
    output_text_clip: Path | None, output_ansi_clip: Path | None,
    output_png_clip: Path | None, output_geojson_clip: Path | None,
    clipboard: str | None, clip_source: str | None,
) -> None:
    """Explore OME-TIFF, Aperio SVS, ordinary images, or PDFs as terminal text.

    Wheel/+/- zoom, arrows or mouse drag pan, c toggles the channel panel.
    PgUp/PgDn or [/] flip PDF pages/OME series; p jumps to a specific page/series.
    l/u/g edit limits and gamma; m cycles colored modes; M cycles ASCII ramps; b toggles blocks; ? shows help.
    s selects a rectangle; v starts a polygon; j copies GeoJSON; J saves it.
    Selection: y/Y text, i/I PNG, t/T ANSI, j/J GeoJSON (copy/save).
    """
    if (lower is None) != (upper is None):
        raise click.UsageError("--lower and --upper must be supplied together")
    if not snapshot and (output_path is not None or ansi):
        raise click.UsageError("--output-path and --ansi require --snapshot")
    if snapshot and any(v is not None for v in
                        (output_text_clip, output_ansi_clip, output_png_clip,
                         output_geojson_clip, clipboard, clip_source)):
        raise click.UsageError("Clip options apply only to interactive viewing")
    if snapshot and pdf_cache_pages is not None:
        raise click.UsageError("--pdf-cache-pages applies only to interactive PDF viewing")
    if snapshot and no_overwrite and output_path is None:
        raise click.UsageError("--no-overwrite requires --output-path")
    if snapshot and color is True and not ansi:
        raise click.UsageError("Colored snapshots require --ansi")
    if snapshot and mode == "blocks" and not ansi:
        raise click.UsageError(
            "Half-block snapshots require --ansi; use detail, shape, or simple for plain text"
        )
    if not snapshot and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        raise click.UsageError(
            "Interactive viewing requires a terminal; use --snapshot for text output"
        )
    try:
        if output_path is not None:
            if output_path.resolve() == input_path.resolve() or (
                output_path.exists() and os.path.samefile(input_path, output_path)
            ):
                raise ValueError("The snapshot destination must not be the input image")
            if no_overwrite and output_path.exists():
                raise FileExistsError(f"Snapshot destination already exists: {output_path}")
            if not output_path.parent.is_dir():
                raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
        use_color = (not snapshot or ansi) if color is None else color
        if mode == "auto":
            mode = resolve_auto_mode(
                snapshot=snapshot, ansi=ansi, color=use_color,
                is_tty=sys.stdout.isatty(), encoding=sys.stdout.encoding,
                term=os.environ.get("TERM"),
                depth=None if snapshot else available_color_depth(),
            )
        window = None if lower is None else IntensityWindow(lower, upper)
        viewport = None if region is None else Viewport(*region)
        kind = image_kind(input_path)
        name = input_path.name
        if kind == "tiff":
            name += f" | series {series}"
        elif kind == "pdf":
            dpi = pdf_dpi_steps(DEFAULT_PDF_MAX_DPI if pdf_max_dpi is None else pdf_max_dpi)[-1]
            name += f" | page {page or 1} | {dpi:g} DPI raster"
        elif frame_index is not None:
            name += f" | frame {frame_index} (0-based)"
        matte = None if background is None else ((255, 255, 255) if background == "white"
                                                else (0, 0, 0))
        if not snapshot:
            from .viewer.file_viewer import view_file

            asyncio.run(view_file(
                input_path, series=series, page=page, frame=frame_index,
                pdf_max_dpi=pdf_max_dpi, pdf_cache_pages=pdf_cache_pages, background=matte,
                channels=channels, window=window, mode=mode, color=use_color, invert=invert,
                cell_aspect=cell_aspect, viewport=viewport, cache_mib=cache_mib,
                max_read_pixels=max_read_pixels, overview_cache_mib=overview_cache_mib,
                overview_max_source_pixels=overview_max_source_pixels,
                startup_overview_seconds=startup_overview_seconds,
                output_text_clip=output_text_clip,
                output_ansi_clip=output_ansi_clip, output_png_clip=output_png_clip,
                output_geojson_clip=output_geojson_clip, clipboard=clipboard or "auto",
                clip_source=clip_source or "auto", clip_overwrite=not no_overwrite,
            ))
            return
        if kind != "pdf" and pdf_max_dpi is not None:
            raise ValueError("--pdf-max-dpi applies only to PDF; TIFF/raster options do not apply")
        with (
            open_image(input_path, series=series, page=page, frame=frame_index,
                       pdf_dpi=dpi if kind == "pdf" else None, background=matte) as image,
            ViewSession(image, name=name, cache_mib=cache_mib,
                        max_read_pixels=max_read_pixels, overview_cache_mib=overview_cache_mib,
                        overview_max_source_pixels=overview_max_source_pixels) as session,
        ):
            if kind == "pdf" and viewport is not None:
                # --region stays on the quick-preview coordinate grid even when
                # a deterministic snapshot renders directly at the detail ceiling.
                initial_dpi = pdf_dpi_steps(dpi)[0]
                if initial_dpi != dpi:
                    height, width = pdf_raster_shape(input_path, page or 1, initial_dpi)
                    sy, sx = session.info.shape[0] / height, session.info.shape[1] / width
                    viewport = Viewport(viewport.x * sx, viewport.y * sy,
                                        viewport.width * sx, viewport.height * sy)
            selected = session._select_channels(channels)
            renderer = TextRenderer(
                mode=mode, color=use_color,
                invert=session.info.is_rgb if invert is None else invert, cell_aspect=cell_aspect,
            )
            whole_snapshot = viewport is None
            if whole_snapshot:
                # Snapshots depict the full extent by default, as does a quick
                # interactive start. A costly overview can use a labelled sparse
                # fallback rather than silently exporting a center crop.
                viewport, native = startup_fit(session, renderer, columns=columns,
                                               rows=rows, channels=selected)
                if not native:
                    prepare_fast_overview(session, selected, seconds=startup_overview_seconds)
            sample_rows, sample_columns = renderer.sample_shape(columns=columns, rows=rows)
            plan = session.plan(viewport, rows=sample_rows, columns=sample_columns,
                                channels=selected, allow_sampled_overview=whole_snapshot)
            if plan.level.approximate:
                click.echo("Pyramid scale unknown: using approximate extent-fit display mapping.",
                           err=True)
            samples = session.sample(plan)
            if samples.plan.sampled_overview is not None:
                click.echo(("Cached full-coverage overview: native coarse pixels were reduced "
                            "for this snapshot." if samples.cached_overview else
                            "Sampled overview: sparse navigational preview, not a complete "
                            "native reduction."), err=True)
            styles = session.prepare_styles(samples, windows=[window] * len(selected))
            frame = session.render(samples, styles=styles, renderer=renderer,
                                   columns=columns, rows=rows)
            text = frame.to_text(ansi=ansi) + "\n"
            if output_path is None:
                click.echo(text, nl=False, color=ansi)
            else:
                _write_text(output_path, text, overwrite=not no_overwrite)
    except MemoryError as error:
        raise click.ClickException(
            "Not enough RAM to open/render this image. Ordinary images retain a decoded "
            "raster and pyramid; for PDF, try a lower --pdf-max-dpi."
        ) from error
    except (ValueError, LookupError, OSError, RuntimeError, EOFError) as error:
        raise click.ClickException(str(error)) from error