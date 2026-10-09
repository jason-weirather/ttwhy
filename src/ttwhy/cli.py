"""Click front door for the same sampling and rendering library used by the TUI."""

import asyncio
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import click
from omeify import OMETiffReader

from . import __version__
from .intensity_window import IntensityWindow
from .text_renderer import RENDER_MODES, TextRenderer
from .view_session import ViewSession
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
              help="Zero-based OME image series; one series is displayed at a time.")
@click.option("--channel", "channels", type=click.IntRange(min=0), multiple=True, default=(0,),
              help="Zero-based logical channel. Repeat to composite multiple markers.")
@click.option("--lower", type=float,
              help="Lower display limit in raw intensity units; pair with --upper.")
@click.option("--upper", type=float,
              help="Upper display limit in raw intensity units; pair with --lower.")
@click.option("--mode", type=click.Choice([*RENDER_MODES, "ascii"]), default="detail",
              help="detail: spatial Braille; shape: stroke art; simple/ascii: legacy density; "
                   "blocks: legacy two-color half-blocks.")
@click.option("--color/--no-color", default=None,
              help="Use RGB/pseudocolor (auto: on in the viewer, off for plain snapshots).")
@click.option("--invert/--no-invert", default=None,
              help="Reverse text ink density (auto: on for RGB, off for fluorescence).")
@click.option("--cell-aspect", type=click.FloatRange(min=0, min_open=True), default=2.0,
              help="Terminal cell height divided by width.")
@click.option("--region", type=float, nargs=4, metavar="X Y WIDTH HEIGHT",
              help="Initial viewport in base-image pixels instead of fitting the image.")
@click.option("--cache-mib", type=click.FloatRange(min=0), default=32.0,
              help="Maximum retained resampled-preview cache size; decoder storage is separate.")
@click.option("--max-read-pixels", type=click.IntRange(min=1), default=4194304,
              help="Maximum source samples per request, counting selected channels/RGB samples.")
@click.option("--snapshot", is_flag=True, help="Render one text frame without starting the viewer.")
@click.option("--columns", type=click.IntRange(min=1), default=120,
              help="Snapshot width in characters.")
@click.option("--rows", type=click.IntRange(min=1), default=40,
              help="Snapshot height in characters.")
@click.option("--ansi", is_flag=True, help="Allow ANSI color escape sequences in a snapshot.")
@click.option("--output-path", "-o", type=click.Path(dir_okay=False, path_type=Path),
              help="Snapshot destination; omitted means stdout. Requires --snapshot.")
@click.option("--no-overwrite", is_flag=True,
              help="Refuse to replace an existing snapshot destination.")
def main(
    input_path: Path, series: int, channels: tuple[int, ...], lower: float | None,
    upper: float | None, mode: str, color: bool | None, invert: bool | None,
    cell_aspect: float, region: tuple[float, ...] | None, cache_mib: float,
    max_read_pixels: int, snapshot: bool, columns: int, rows: int, ansi: bool,
    output_path: Path | None, no_overwrite: bool,
) -> None:
    """Explore a pyramidal OME-TIFF as text, because sometimes the answer is yes.

    Wheel/+/- zoom, arrows or mouse drag pan, c toggles the channel panel.
    l/u/g edit limits and gamma; m cycles render modes; b toggles blocks; ? shows help.
    """
    if (lower is None) != (upper is None):
        raise click.UsageError("--lower and --upper must be supplied together")
    if not snapshot and (output_path is not None or ansi or no_overwrite):
        raise click.UsageError("--output-path, --ansi and --no-overwrite require --snapshot")
    if no_overwrite and output_path is None:
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
        window = None if lower is None else IntensityWindow(lower, upper)
        viewport = None if region is None else Viewport(*region)
        with (
            OMETiffReader(input_path, series=series) as image,
            ViewSession(image, name=f"{input_path.name} | series {series}",
                        cache_mib=cache_mib, max_read_pixels=max_read_pixels) as session,
        ):
            selected = session._select_channels(channels)
            renderer = TextRenderer(
                mode=mode, color=(not snapshot or ansi) if color is None else color,
                invert=session.info.is_rgb if invert is None else invert, cell_aspect=cell_aspect,
            )
            if snapshot:
                if viewport is None:
                    viewport, limited = session.fit(columns=columns, rows=rows, renderer=renderer,
                                                    channels=selected)
                    if limited:
                        click.echo(
                            "Read budget: rendering a bounded center crop, not the whole slide.",
                            err=True,
                        )
                sample_rows, sample_columns = renderer.sample_shape(columns=columns, rows=rows)
                plan = session.plan(viewport, rows=sample_rows, columns=sample_columns,
                                    channels=selected)
                if plan.level.approximate:
                    click.echo("Pyramid scale unknown: using approximate extent-fit display mapping.",
                               err=True)
                samples = session.sample(plan)
                styles = session.prepare_styles(samples, windows=[window] * len(selected))
                frame = session.render(samples, styles=styles, renderer=renderer,
                                       columns=columns, rows=rows)
                text = frame.to_text(ansi=ansi) + "\n"
                if output_path is None:
                    click.echo(text, nl=False, color=ansi)
                else:
                    _write_text(output_path, text, overwrite=not no_overwrite)
            else:
                from .viewer import Viewer

                asyncio.run(Viewer(session, channels=selected, window=window, renderer=renderer,
                                   viewport=viewport).run())
    except (ValueError, LookupError, OSError, RuntimeError, EOFError) as error:
        raise click.ClickException(str(error)) from error
